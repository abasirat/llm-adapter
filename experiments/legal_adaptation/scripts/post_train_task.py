"""
post_train_task.py – Task-specific post-training of adapter parameters.

Binarises a task's HuggingFace training split (via prepare_dataset) then
fine-tunes the adapter on it using the existing train() loop.

Can be used as a standalone script or imported from eval.py:

    from post_train_task import run_post_training
    checkpoint = run_post_training(
        model_path="outputs/models/my_adapter.pt",
        training_cfg=training_cfg_dict,
        data_cfg=data_cfg_dict,
        output_dir="outputs/post_trained/casehold",
        device="auto",
    )
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------------
# Make train.py importable when this module is loaded from eval.py, which may
# be run from a different working directory.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from train import (          # noqa: E402 – must follow sys.path adjustment
    MixedDataset,
    TokenBinDataset,
    set_seed,
    token_bin_collate,
    train,
)

import wandb                 # noqa: E402
from llm_adapter import load_model as _llm_load_model  # noqa: E402
from llm_adapter import setup_model as _llm_setup_model  # noqa: E402
from evaluations.ledgar.evaluator import LEDGARGenerationDataset, supervised_lm_collate  # noqa: E402


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_post_training(
    model_path: str,
    training_cfg: dict,
    data_cfg: dict,
    output_dir: str,
    device: str = "auto",
) -> str:
    """Fine-tune adapter parameters on a task training split.

    Parameters
    ----------
    model_path   : path to the domain-adapted adapter checkpoint (.pt file)
    training_cfg : dict with training hyperparameters (mirrors task_ft.yaml)
    data_cfg     : dict with dataset configuration (mirrors data/casehold.yaml)
    output_dir   : directory where post-trained checkpoints are written
    device       : device string – "auto", "cuda", "mps", or "cpu"

    Returns
    -------
    str  Absolute path to the best checkpoint produced by training
         (``<output_dir>/checkpoint-best``).  Falls back to the trace
         checkpoint if no validation-driven improvement was recorded.
    """
    from evaluations.utils import select_device  # lazy import – avoids circular deps

    # Ensure prepare_dataset is importable.
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from prepare_dataset import run_prepare_dataset  # noqa: E402

    device_name = select_device(device)
    device_obj = torch.device(device_name)
    seed = training_cfg.get("seed", 42)
    set_seed(seed)

    # ------------------------------------------------------------------
    # 1. Binarise task training data (idempotent – skipped when .bin exists)
    # ------------------------------------------------------------------
    task_name = data_cfg.get("task")

    if task_name not in {"ledgar"}:
        bin_path = os.path.join(output_dir, "data.bin")
        if not os.path.exists(bin_path) or data_cfg.get("force_prepare", True):
            print(f"\n[post_training] Tokenising task data → {bin_path}")
            data_cfg["output_dir"] = output_dir  # prepare_dataset expects an output_dir key
            run_prepare_dataset(data_cfg)
        else:
            print(f"\n[post_training] Reusing tokenised data: {bin_path}")
    else:
        print(f"\n[post_training] Skipping data preparation for task {task_name!r} "
              "(dataset will be loaded directly from HuggingFace).")
        bin_path = None  # LEDGARGenerationDataset doesn't use a bin file

    # ------------------------------------------------------------------
    # 2. Load model — handle both .pt adapter checkpoints and plain HF model IDs
    # ------------------------------------------------------------------
    print(f"[post_training] Loading model from {model_path}")
    _is_pt_file = os.path.isfile(model_path)

    if _is_pt_file:
        model, _tokenizer, adapter_config = _llm_load_model(model_path)
        saved_meta = torch.load(model_path, map_location="cpu", weights_only=False)
        adapter_type = saved_meta["adapter_type"]
    else:
        # model_path is a HuggingFace model ID (e.g. "gpt2") — no adapter file
        adapter_type   = "none"
        adapter_config = None

        # For plain HF model IDs, ensure post-training has trainable parameters.
        # Defaults here are task fine-tuning friendly and can be overridden
        # from task_ft.yaml.
        posttrain_num_tailor_layers = int(training_cfg.get("num_tailor_layers", 1))
        posttrain_freeze_lm_head = bool(training_cfg.get("freeze_lm_head", False))

        model, _tokenizer = _llm_setup_model(
            model_name=model_path,
            adapter_type="none",
            adapter_config=None,
            num_tailor_layers=posttrain_num_tailor_layers,
            freeze_lm_head=posttrain_freeze_lm_head,
        )

    if model is None:
        raise RuntimeError("Model setup failed: expected a model instance, got None.")

    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_param_count == 0:
        raise RuntimeError(
            "Post-training model has zero trainable parameters. "
            "If using a HF model ID, set training_cfg.num_tailor_layers > 0 "
            "or training_cfg.freeze_lm_head=false."
        )

    # Extract layer_adapter-specific settings from the stored adapter_config dict.
    def _get(key, default):
        if isinstance(adapter_config, dict):
            return adapter_config.get(key, default)
        return default

    variational_modeling      = _get("variational_modeling", False)
    aggregation_strategy      = _get("aggregation_strategy", "attention")
    shift_regularization      = _get("shift_regularization", False)
    layer_adapter_max_temp    = _get("attention_temperature", 2.0)
    layer_adapter_min_temp    = _get("min_attention_temperature", 0.8)

    # ------------------------------------------------------------------
    # 3. Build dataloaders from the binarised data
    # ------------------------------------------------------------------
    context_size = training_cfg.get("context_size", 512)
    batch_size   = training_cfg.get("batch_size", 8)
    val_fraction = training_cfg.get("val_fraction", 0.1)
    num_workers  = training_cfg.get("num_workers", 0)
    pin_memory   = device_obj.type == "cuda"

    # Print a summary of the training configuration for easy reference in logs:
    print("\n[post_training] Training configuration:")
    print(f"  model_path: {model_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  device: {device_obj}")
    print(f"  adapter_type: {adapter_type}")
    print(f"  variational_modeling: {variational_modeling}")
    print(f"  aggregation_strategy: {aggregation_strategy}")
    print(f"  shift_regularization: {shift_regularization}")
    print(f"  layer_adapter_max_temperature: {layer_adapter_max_temp}")
    print(f"  layer_adapter_min_temperature: {layer_adapter_min_temp}")
    print(f"  context_size: {context_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  val_fraction: {val_fraction}")

    #full_dataset = TokenBinDataset(bin_path, context_size=context_size)

    task_name = data_cfg.get("task")
    if task_name == "ledgar":
        full_dataset = LEDGARGenerationDataset(
            tokenizer=_tokenizer,
            split=data_cfg.get("split", "train"),
            max_length=training_cfg.get("context_size", 1024),
            max_label_length=data_cfg.get("max_label_length", 16),
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        collate_fn = partial(
            supervised_lm_collate,
            pad_token_id=_tokenizer.pad_token_id,
        )

    #elif task_name == "casehold":
    #    full_dataset = CaseHOLDGenerationDataset(
    #        tokenizer=_tokenizer,
    #        split=data_cfg.get("split", "train"),
    #        max_length=training_cfg.get("context_size", 1024),
    #    )
    #    collate_fn = partial(
    #        supervised_lm_collate,
    #        pad_token_id=_tokenizer.pad_token_id,
    #    )

    else:
        if bin_path is None:
            raise RuntimeError(
                f"Expected a token-bin path for task={task_name!r}, but got None."
            )
        full_dataset = TokenBinDataset(bin_path, context_size=context_size)
        collate_fn = partial(token_bin_collate, context_size=context_size)

    if len(full_dataset) == 0:
        raise RuntimeError(
            f"Dataset for task={task_name} is empty."
        )

    train_dataset = full_dataset
    val_dataset: Optional[torch.utils.data.Dataset] = None

    if val_fraction > 0.0 and len(full_dataset) > 1:
        total     = len(full_dataset)
        val_size  = max(1, int(total * val_fraction))
        train_size = total - val_size

        if train_size >= 1:
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed),
            )
            print(
                f"[post_training] Dataset split – train: {train_size} chunks, "
                f"val: {val_size} chunks."
            )

        #collate_fn = partial(token_bin_collate, context_size=context_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
        )

    # ------------------------------------------------------------------
    # 4. Run the training loop (reuses train() from train.py unchanged)
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "checkpoint")

    if not os.path.exists(model_save_path) or training_cfg.get("force_retrain", False):
        # Disable wandb for post-training runs; train() calls wandb.log() throughout.
        wandb.init(mode="disabled")
        try:
            train(
                model=model,
                train_dataloader=train_dataloader,
                device=device_obj,
                model_path=model_save_path,
                learning_rate=training_cfg.get("learning_rate", 1e-4),
                adapter_type=adapter_type,
                adapter_config=adapter_config,
                variational_modeling=variational_modeling,
                num_epochs=training_cfg.get("num_epochs", 3),
                adam_beta1=training_cfg.get("adam_beta1", 0.9),
                adam_beta2=training_cfg.get("adam_beta2", 0.999),
                weight_decay=training_cfg.get("weight_decay", 0.01),
                early_stopping_patience=training_cfg.get("early_stopping_patience", 3),
                early_stopping_min_delta=training_cfg.get("early_stopping_min_delta", 1e-4),
                val_dataloader=val_dataloader,
                progress_interval=training_cfg.get("progress_interval", 10),
                val_interval=training_cfg.get("val_interval", 50),
                kl_loss_weight=training_cfg.get("kl_loss_weight", 0.0),
                kl_warmup_fraction=training_cfg.get("kl_warmup_fraction", 0.0),
                kl_schedule=training_cfg.get("kl_schedule", "linear"),
                shift_regularization=shift_regularization,
                layer_adapter_max_temperature=layer_adapter_max_temp,
                layer_adapter_min_temperature=layer_adapter_min_temp,
                aggregation_strategy=aggregation_strategy,
            )
        finally:
            wandb.finish()
    else:
        print(f"[post_training] Checkpoint already exists at {model_save_path}. "
              "Use --force_retrain to override.")

    # train() saves the best val checkpoint to model_save_path + "-best"
    best_path  = model_save_path + "-best"
    trace_path = model_save_path + "-trace"

    if os.path.exists(best_path):
        print(f"[post_training] Best checkpoint: {best_path}")
        return best_path

    if os.path.exists(trace_path):
        print(
            f"[post_training] No val-improvement checkpoint found; "
            f"using trace checkpoint: {trace_path}"
        )
        return trace_path

    raise FileNotFoundError(
        f"Post-training did not produce any checkpoint under {output_dir!r}. "
        "Check that training ran for at least one validation interval."
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-train adapter parameters on a task's training split.",
    )
    parser.add_argument("--model_path",      required=True, help="Domain-adapted adapter checkpoint (.pt).")
    parser.add_argument("--training_config", required=True, help="Path to post-training YAML config (task_ft.yaml).")
    parser.add_argument("--data_config",     required=True, help="Path to task data YAML config (e.g. data/casehold.yaml).")
    parser.add_argument("--output_dir",      required=True, help="Directory for post-trained checkpoints.")
    parser.add_argument("--device",          default="auto", help="Device: auto, cuda, mps, cpu.")
    parser.add_argument("--force_retrain",   action="store_true",
                        help="Re-train even if a checkpoint already exists in output_dir.")
    args = parser.parse_args()

    best_path = os.path.join(args.output_dir, "checkpoint-best")
    if not args.force_retrain and os.path.exists(best_path):
        print(f"Checkpoint already exists at {best_path}. Use --force_retrain to override.")
        return

    training_cfg = _load_yaml(args.training_config)
    data_cfg     = _load_yaml(args.data_config)

    training_cfg["force_prepare"] = args.force_retrain  # ensure data is re-prepared when forcing retrain

    run_post_training(
        model_path=args.model_path,
        training_cfg=training_cfg,
        data_cfg=data_cfg,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
