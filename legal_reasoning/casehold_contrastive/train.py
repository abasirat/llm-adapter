"""
Contrastive/ranking-based causal-LM training on CaseHOLD (LexGLUE).

Training objective
------------------
For a given context *c* and the five candidate holdings h_0…h_4, the model
scores each holding as:

    s_i = mean_t [ log P(h_i token_t | c, h_i tokens_<t) ]

and is optimised so that s_correct > s_incorrect via one of:

  softmax  (default) : F.cross_entropy(scores, label)   — InfoNCE / NCE
  hinge              : Σ max(0, γ − s_c + s_i)          — max-margin pairwise
  logistic           : Σ log(1 + exp(s_i − s_c))        — smooth margin

This is a **pure causal language model** objective.  No classification head
is added.  The model architecture is unchanged.

An optional auxiliary CE loss on the correct holding is available via
``--aux_ce_alpha`` (default 0.0 — pure ranking).

Usage
-----
python -m legal_reasoning.casehold_contrastive.train \\
    --model_name_or_path gpt2 \\
    --output_dir checkpoints/casehold_contrastive \\
    --ranking_loss softmax \\
    --aux_ce_alpha 0.1 \\
    --max_length 512 \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 16 \\
    --learning_rate 2e-5 \\
    --device mps

Evaluating the trained model
-----------------------------
The saved checkpoint is a standard HuggingFace model directory (for base GPT-2)
or a .pt adapter file (for adapter checkpoints).  Pass it to the evaluator:

    python run_all_evaluations.py \\
        --model_name_or_path checkpoints/casehold_contrastive \\
        ...
"""

import argparse
import logging
import os
import sys

import numpy as np
from transformers import EvalPrediction, TrainingArguments

from legal_reasoning.utils import load_model_for_training
from .dataset import CaseHoldRankingDataset
from .trainer import RankingCollator, RankingTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device helper  (identical to casehold_continuation/train.py)
# ---------------------------------------------------------------------------

def _apply_device_env(device: str) -> None:
    """
    Steer the Trainer to the requested device without relying on removed
    TrainingArguments kwargs (``no_cuda`` / ``use_mps_device`` were dropped in
    transformers 4.42+).

    ``cpu``  : set ``CUDA_VISIBLE_DEVICES=""`` so all GPUs are hidden
    ``mps``  : enable the MPS fallback env var for unsupported ops
    ``cuda`` : clear any CPU-forcing override that may already be set
    ``auto`` : do nothing — Trainer auto-detects (cuda > mps > cpu)
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    elif device == "mps":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    elif device == "cuda":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    # 'auto': leave environment untouched


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_accuracy(eval_pred: EvalPrediction) -> dict:
    """
    Compute CaseHOLD accuracy from accumulated ``(N, 5)`` score matrices.

    ``eval_pred.predictions``  — ``(N, 5)`` per-candidate mean log-prob scores
    ``eval_pred.label_ids``    — ``(N,)``   gold label indices
    """
    scores = eval_pred.predictions   # (N, 5)  numpy array
    labels = eval_pred.label_ids     # (N,)
    preds = scores.argmax(-1)        # (N,)
    acc = float((preds == labels).mean())
    return {"casehold_accuracy": acc}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Contrastive/ranking causal-LM training on CaseHOLD.\n"
            "Optimises s_correct > s_incorrect using differentiable "
            "likelihood scoring — no classification head."
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
        help=(
            "Unfreeze all parameters for full fine-tuning. "
            "Default: respect the checkpoint's frozen/trainable setup."
        ),
    )

    # ── Loss / objective ────────────────────────────────────────────────
    parser.add_argument(
        "--ranking_loss", type=str, default="softmax",
        choices=["softmax", "hinge", "logistic"],
        help=(
            "Ranking objective:\n"
            "  softmax  — InfoNCE cross-entropy over scores (default)\n"
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
            "Weight of auxiliary CE loss on the correct holding tokens. "
            "0.0 = pure ranking loss. Shares the main forward pass."
        ),
    )

    # ── Data ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--max_length", type=int, default=512,
        help=(
            "Maximum total sequence length per (prompt + candidate) sequence. "
            "Default 512 is lower than the continuation trainer's 1024 because "
            "each example holds 5 sequences; keep this in mind when setting "
            "--per_device_train_batch_size."
        ),
    )
    parser.add_argument(
        "--max_train_examples", type=int, default=None,
        help="Cap on training examples (default: all ~45 K).",
    )
    parser.add_argument(
        "--max_eval_examples", type=int, default=None,
        help="Cap on validation examples (default: all ~3.9 K).",
    )

    # ── Training hyperparameters ─────────────────────────────────────────
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=1,
        help=(
            "Batch size per device.  Each example contains 5 sequences, so "
            "effective GPU memory is ~5× a comparable CE run; start with 1."
        ),
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.05,
        help="Fraction of total training steps used for LR warmup.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ── Mixed precision ──────────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fp16", action="store_true",
        help="float16 mixed-precision (CUDA only).",
    )
    group.add_argument(
        "--bf16", action="store_true",
        help="bfloat16 mixed-precision (Ampere+ GPU or recent MPS).",
    )

    # ── Evaluation & checkpointing ───────────────────────────────────────
    parser.add_argument(
        "--eval_strategy", type=str, default="epoch",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500,
        help="Validate every N steps (only when --eval_strategy steps).",
    )
    parser.add_argument(
        "--save_strategy", type=str, default="epoch",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Save every N steps (only when --save_strategy steps).",
    )
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--no_load_best_model_at_end", action="store_true",
        help="Do NOT restore the best checkpoint at the end of training.",
    )

    # ── Device ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )

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

    # load_best_model_at_end requires save_strategy == eval_strategy.
    load_best = not args.no_load_best_model_at_end
    if load_best and args.save_strategy != args.eval_strategy:
        logger.warning(
            "--save_strategy (%s) != --eval_strategy (%s); "
            "disabling load_best_model_at_end to avoid a Trainer error.",
            args.save_strategy,
            args.eval_strategy,
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
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %d / %d (%.1f%%)",
        trainable,
        total,
        100 * trainable / total if total else 0.0,
    )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Loading training set…")
    train_dataset = CaseHoldRankingDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_examples=args.max_train_examples,
    )
    logger.info("Training examples: %d", len(train_dataset))

    logger.info("Loading validation set…")
    eval_dataset = CaseHoldRankingDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_examples=args.max_eval_examples,
    )
    logger.info("Validation examples: %d", len(eval_dataset))

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
        # Best model selected by accuracy (higher is better).
        metric_for_best_model="eval_casehold_accuracy",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        # The dataset returns tensors with exact names; prevent the Trainer
        # from stripping them before passing to compute_loss.
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
        compute_metrics=_compute_accuracy,
        # Ranking-specific
        ranking_loss=args.ranking_loss,
        margin=args.margin,
        aux_ce_alpha=args.aux_ce_alpha,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info(
        "Starting training (ranking_loss=%s, margin=%.2f, aux_ce_alpha=%.3f)…",
        args.ranking_loss,
        args.margin,
        args.aux_ce_alpha,
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
            "  --model_name_or_path %s",
            pt_path,
        )
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
