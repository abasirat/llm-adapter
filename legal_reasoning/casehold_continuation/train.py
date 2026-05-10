"""
Supervised causal-LM continuation training on CaseHOLD (LexGLUE).

Training objective
------------------
Each example is a single sequence:

    [context]\\nHolding: [correct holding]

Loss is computed only on the holding tokens; the context is masked with -100.
This is pure causal-LM fine-tuning — no classification head is added.

Usage
-----
python -m legal_reasoning.casehold_continuation.train \\
    --model_name_or_path gpt2 \\
    --output_dir checkpoints/casehold_continuation \\
    --max_length 1024 \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 2 \\
    --gradient_accumulation_steps 8 \\
    --learning_rate 2e-5 \\
    --fp16 \\
    --max_train_examples 10000 \\
    --max_eval_examples 1000

Evaluating the trained model
-----------------------------
The saved output_dir is a standard HuggingFace model directory.
Pass it directly to the existing CaseHOLD evaluator:

    python run_all_evaluations.py \\
        --model_name_or_path checkpoints/casehold_continuation \\
        ...
"""

import argparse
import logging
import math
import os
import sys

from transformers import Trainer, TrainerCallback, TrainingArguments

from .dataset import CaseHoldContinuationDataset
from .utils import CausalLMCollator, load_model_for_training

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _apply_device_env(device: str) -> None:
    """
    Steer the Trainer to the requested device without relying on removed
    TrainingArguments kwargs (no_cuda / use_mps_device were dropped in
    transformers 4.42+).

    'cpu'  : set CUDA_VISIBLE_DEVICES="" so all GPUs are hidden
    'mps'  : enable the MPS fallback env var for unsupported ops
    'cuda' : clear any CPU-forcing override that may already be set
    'auto' : do nothing — Trainer auto-detects (cuda > mps > cpu)
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
# Perplexity callback
# ---------------------------------------------------------------------------

class PerplexityCallback(TrainerCallback):
    """
    Log validation perplexity after each evaluation step.

    Because all prompt token labels are set to -100, the Trainer's eval_loss
    is computed exclusively on the holding (target) tokens.  Perplexity is
    simply exp(that loss).
    """

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        if "eval_loss" in metrics:
            try:
                ppl = math.exp(metrics["eval_loss"])
            except OverflowError:
                ppl = float("inf")
            metrics["eval_perplexity"] = round(ppl, 4)
            logger.info("eval_perplexity: %.4f", ppl)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a causal LM on CaseHOLD via continuation training.\n"
            "Loss is computed only on the correct holding tokens."
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

    # ── Data ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="Maximum total sequence length (prompt + target tokens).",
    )
    parser.add_argument(
        "--max_train_examples", type=int, default=None,
        help="Cap on training examples (default: use all ~45 K).",
    )
    parser.add_argument(
        "--max_eval_examples", type=int, default=None,
        help="Cap on validation examples (default: use all ~3.9 K).",
    )

    # ── Training hyperparameters ─────────────────────────────────────────
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
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
        help="When to run validation.",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500,
        help="Validate every N steps (only when --eval_strategy steps).",
    )
    parser.add_argument(
        "--save_strategy", type=str, default="epoch",
        choices=["no", "steps", "epoch"],
        help="When to save checkpoints.",
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Save every N steps (only when --save_strategy steps).",
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=2,
        help="Maximum number of checkpoints to retain.",
    )
    parser.add_argument(
        "--no_load_best_model_at_end", action="store_true",
        help="Do NOT restore the best checkpoint at the end of training.",
    )

    # ── Device ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to train on. 'auto' lets the Trainer pick (cuda > mps > cpu).",
    )

    # ── Misc ─────────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=0,
        help="Subprocesses for data loading. 0 = main process only (safest).",
    )

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
            args.save_strategy, args.eval_strategy,
        )
        load_best = False

    logger.info("Device: %s", args.device)
    _apply_device_env(args.device)

    # ------------------------------------------------------------------
    # Model & tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading model: %s", args.model_name_or_path)
    model, tokenizer = load_model_for_training(
        args.model_name_or_path,
        unfreeze_all=args.unfreeze_all,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %d / %d (%.1f%%)",
        trainable, total, 100 * trainable / total if total else 0,
    )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Loading training set…")
    train_dataset = CaseHoldContinuationDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_examples=args.max_train_examples,
    )
    logger.info("Training examples: %d", len(train_dataset))

    logger.info("Loading validation set…")
    eval_dataset = CaseHoldContinuationDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_examples=args.max_eval_examples,
    )
    logger.info("Validation examples: %d", len(eval_dataset))

    # ------------------------------------------------------------------
    # Data collator
    # ------------------------------------------------------------------
    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)

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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        # Our dataset returns tensors named exactly as the model expects;
        # disabling column removal prevents the Trainer from stripping them.
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[PerplexityCallback()],
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting training…")
    trainer.train()

    # ------------------------------------------------------------------
    # Save final model and tokenizer
    # ------------------------------------------------------------------
    logger.info("Saving model and tokenizer to: %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
