"""
Contrastive/ranking-based causal-LM training on UNFAIR-ToS (LexGLUE).

UNFAIR-ToS is a **multi-label** benchmark: each Terms of Service clause may
belong to zero or more of 8 unfairness categories.  Because the evaluator
scores each category independently via a binary Yes/No prompt, training
mirrors this by expanding each clause into per-*(clause, category)* binary
ranking instances:

  * Active category   → candidates [" Yes", " No"],  label = 0  (Yes correct)
  * Inactive category → candidates [" Yes", " No"],  label = 1  (No correct)

Training objective
------------------
For a clause *p* and category *k* with candidates {" Yes", " No"}:

    s_i = mean_t [ log P(candidate_i token_t | prompt, candidate_i tokens_<t) ]

Optimise s_correct > s_incorrect via one of:

  softmax  (default) : F.cross_entropy(scores, label)   — binary CE (C=2)
  hinge              : max(0, γ − s_c + s_i)            — max-margin
  logistic           : log(1 + exp(s_i − s_c))          — smooth margin

This is a **pure causal language model** objective.  No classification head
is added.  Prompt / candidate encoding matches ``evaluations/unfair_tos/evaluator.py``
so that training and inference use the same representation.

Class imbalance
---------------
UNFAIR-ToS has ~7 inactive categories per active one.  ``--num_negative_categories``
caps inactive-category instances per clause (default 4).  Set to 0 for
positives-only training or omit the flag to use all inactive categories.

NOTE: The ``eval_sampled_micro_f1`` reported during training measures F1 over
the sampled binary instances only, not over the full 8-category multi-label
evaluation.  For the true UNFAIR-ToS micro/macro F1 run:

    python run_all_evaluations.py \\
        --model_name_or_path <output_dir> \\
        --skip perplexity ledgar truthfulqa lambada style casehold

Usage
-----
python -m legal_reasoning.unfair_tos_contrastive.train \\
    --model_name_or_path gpt2 \\
    --output_dir checkpoints/unfair_tos_contrastive \\
    --ranking_loss softmax \\
    --num_negative_categories 4 \\
    --aux_ce_alpha 0.1 \\
    --max_length 512 \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 4 \\
    --gradient_accumulation_steps 4 \\
    --learning_rate 2e-5 \\
    --device mps
"""

import argparse
import logging
import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from transformers import EvalPrediction, TrainingArguments

from legal_reasoning.utils import load_model_for_training
from legal_reasoning.casehold_contrastive.trainer import RankingCollator, RankingTrainer, NegativeResamplerCallback
from .dataset import UnfairTosRankingDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device helper  (identical to casehold_contrastive/train.py)
# ---------------------------------------------------------------------------

def _apply_device_env(device: str) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    elif device == "mps":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    elif device == "cuda":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_sampled_micro_f1(eval_pred: EvalPrediction) -> dict:
    """
    Micro-F1 over the binary (Yes/No) sampled instances.

    This is a proxy metric for training monitoring; it is NOT the true
    8-category UNFAIR-ToS micro-F1.  Use ``run_all_evaluations.py`` for that.

    Each instance is a 2-way ranking: label=0 means Yes is correct,
    label=1 means No is correct.  A correct prediction means the model
    assigns a higher score to the correct candidate.
    """
    scores = eval_pred.predictions   # (N, 2) numpy array
    labels = eval_pred.label_ids     # (N,)
    preds  = scores.argmax(-1)       # (N,)
    # Binary F1 (pos label = 0, i.e. "active category predicted correctly").
    micro_f1 = float(f1_score(labels, preds, average="micro", zero_division=0))
    return {"sampled_micro_f1": micro_f1}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Contrastive/ranking causal-LM training on UNFAIR-ToS.\n"
            "Scores Yes/No candidates via mean log P(candidate | prompt) "
            "and optimises s_correct > s_incorrect — no classification head."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model & output ──────────────────────────────────────────────────
    parser.add_argument(
        "--model_name_or_path", required=True,
        help=(
            "HuggingFace model name (e.g. 'gpt2', 'gpt2-medium'), a local "
            "HuggingFace save_pretrained directory, or a project .pt adapter path."
        ),
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save the trained model and tokenizer.",
    )
    parser.add_argument(
        "--unfreeze_all", action="store_true",
        help="Unfreeze all parameters for full fine-tuning.",
    )
    parser.add_argument(
        "--unfreeze_lm_head", action="store_true",
        help=(
            "Unfreeze the LM head (and tied input embeddings) of an adapter "
            "checkpoint.  Has no effect when --unfreeze_all is set."
        ),
    )

    # ── Loss / objective ────────────────────────────────────────────────
    parser.add_argument(
        "--ranking_loss", type=str, default="softmax",
        choices=["softmax", "hinge", "logistic"],
        help=(
            "Ranking objective:\n"
            "  softmax  — InfoNCE cross-entropy over scores (default; binary CE for C=2)\n"
            "  hinge    — pairwise max-margin with margin --margin\n"
            "  logistic — smooth pairwise margin (no hyperparameter)"
        ),
    )
    parser.add_argument(
        "--margin", type=float, default=1.0,
        help="Hinge margin γ (only used when --ranking_loss hinge).",
    )
    parser.add_argument(
        "--aux_ce_alpha", type=float, default=0.0,
        help=(
            "Weight of auxiliary CE loss on the correct candidate tokens. "
            "0.0 = pure ranking loss. Shares the main forward pass."
        ),
    )

    # ── Negative sampling ────────────────────────────────────────────────
    parser.add_argument(
        "--num_negative_categories", type=int, default=4,
        help=(
            "Number of inactive (negative) categories to sample per clause. "
            "Controls the active/inactive class balance. "
            "Default 4 reduces the ~7:1 imbalance in UNFAIR-ToS. "
            "Set to a large number (e.g. 8) to use all inactive categories."
        ),
    )

    # ── Data ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--max_length", type=int, default=512,
        help=(
            "Maximum total sequence length per (prompt + candidate) sequence. "
            "Each instance holds 2 sequences (Yes and No)."
        ),
    )
    parser.add_argument(
        "--max_train_examples", type=int, default=None,
        help="Cap on training clauses (default: all ~5.5 K).",
    )
    parser.add_argument(
        "--max_eval_examples", type=int, default=None,
        help="Cap on validation clauses (default: all ~2.0 K).",
    )

    # ── Training hyperparameters ─────────────────────────────────────────
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=4,
        help=(
            "Batch size per device.  Each instance contains 2 sequences "
            "(Yes and No).  Can be larger than LEDGAR since C=2."
        ),
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ── Mixed precision ──────────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fp16", action="store_true",
                       help="float16 mixed-precision (CUDA only).")
    group.add_argument("--bf16", action="store_true",
                       help="bfloat16 mixed-precision (Ampere+ GPU or recent MPS).")

    # ── Evaluation & checkpointing ───────────────────────────────────────
    parser.add_argument("--eval_strategy", type=str, default="epoch",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--no_load_best_model_at_end", action="store_true")

    # ── Device ───────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])

    # ── Misc ─────────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Arguments: %s", args)

    load_best = not args.no_load_best_model_at_end
    if load_best and args.save_strategy != args.eval_strategy:
        logger.warning(
            "--save_strategy (%s) != --eval_strategy (%s); "
            "disabling load_best_model_at_end.",
            args.save_strategy, args.eval_strategy,
        )
        load_best = False

    logger.info("Device: %s", args.device)
    _apply_device_env(args.device)

    # ------------------------------------------------------------------
    # Model & tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading model: %s", args.model_name_or_path)
    model, tokenizer, adapter_save_fn = load_model_for_training(
        args.model_name_or_path,
        unfreeze_all=args.unfreeze_all,
        unfreeze_lm_head=args.unfreeze_lm_head,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %d / %d (%.1f%%)",
        trainable, total, 100 * trainable / total if total else 0.0,
    )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Loading training set…")
    train_dataset = UnfairTosRankingDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_negative_categories=args.num_negative_categories,
        max_examples=args.max_train_examples,
        seed=args.seed,
    )
    logger.info(
        "Training instances: %d  "
        "(clauses × ~(active + %d inactive) categories per clause)",
        len(train_dataset), args.num_negative_categories,
    )

    logger.info("Loading validation set…")
    eval_dataset = UnfairTosRankingDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_negative_categories=args.num_negative_categories,
        max_examples=args.max_eval_examples,
        seed=args.seed + 1,  # different seed for val negatives
    )
    logger.info("Validation instances: %d", len(eval_dataset))

    # ------------------------------------------------------------------
    # Data collator
    # ------------------------------------------------------------------
    collator = RankingCollator(pad_token_id=tokenizer.pad_token_id)

    # ------------------------------------------------------------------
    # TrainingArguments
    # ------------------------------------------------------------------
    ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_sampled_micro_f1",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
    )
    if args.eval_strategy == "steps":
        ta_kwargs["eval_steps"] = args.eval_steps
    if args.save_strategy == "steps":
        ta_kwargs["save_steps"] = args.save_steps

    training_args = TrainingArguments(**ta_kwargs)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = RankingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=_compute_sampled_micro_f1,
        ranking_loss=args.ranking_loss,
        margin=args.margin,
        aux_ce_alpha=args.aux_ce_alpha,
    )
    trainer.add_callback(NegativeResamplerCallback(train_dataset, base_seed=args.seed))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info(
        "Starting training "
        "(ranking_loss=%s, num_negative_categories=%d, margin=%.2f, aux_ce_alpha=%.3f)…",
        args.ranking_loss, args.num_negative_categories, args.margin, args.aux_ce_alpha,
    )
    trainer.train()

    # ------------------------------------------------------------------
    # Save final model and tokenizer
    # ------------------------------------------------------------------
    logger.info("Saving model and tokenizer to: %s", args.output_dir)
    if adapter_save_fn is not None:
        pt_path = adapter_save_fn(model, args.output_dir, tokenizer)
        logger.info(
            "Adapter checkpoint saved.  To evaluate, pass:\n"
            "  --model_name_or_path %s", pt_path,
        )
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
