"""
Shared utilities for CaseHOLD continuation training.

  build_prompt            – canonical prompt builder (must match the evaluator)
  load_model_for_training – load a causal LM + tokenizer for fine-tuning
  CausalLMCollator        – right-pad a batch of variable-length examples
"""

import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


# HuggingFace model names that can be loaded directly with AutoModel*.
_HF_CAUSAL_MODELS = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(context: str) -> str:
    """
    Build the causal-LM prompt for a CaseHOLD example.

    CaseHOLD contexts typically end with the literal token '<HOLDING>'.
    We remove it and append '\\nHolding:' to prime the model to generate
    the legal holding as a continuation.

    Example
    -------
    Input  : "... the court ruled <HOLDING>"
    Output : "... the court ruled\\nHolding:"

    NOTE: evaluations/casehold/evaluator.py is updated to call this function
    so that the inference-time prompt is identical to the training prompt.
    """
    context = context.strip()
    if context.endswith("<HOLDING>"):
        context = context[: -len("<HOLDING>")].rstrip()
    return context + "\nHolding:"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_training(
    model_name_or_path: str,
    unfreeze_all: bool = False,
) -> Tuple[torch.nn.Module, PreTrainedTokenizer, Optional[Callable]]:
    """
    Load a causal LM and tokenizer for fine-tuning.

    Handles three cases:
      1. Well-known HuggingFace names  (gpt2, gpt2-medium, …)
      2. Local HuggingFace model dirs  (saved with save_pretrained, have config.json)
      3. Project adapter checkpoints   (.pt files) – loaded via llm_adapter

    Args:
        model_name_or_path: HuggingFace name, directory, or adapter path.
        unfreeze_all: If True, make every parameter trainable regardless of
                      what the checkpoint set up.  Useful for full fine-tuning
                      of a base model or for unfreezing an adapter checkpoint.
    Returns:
        (model, tokenizer, save_fn)

        save_fn is None for plain HuggingFace models.
        For adapter checkpoints, save_fn(model, output_dir, tokenizer) saves
        via llm_adapter.save_model so the result can be reloaded by load_model.
        The adapter .pt file is written to <output_dir>/model.pt and the
        tokenizer files to <output_dir>/.
    """
    save_fn: Optional[Callable] = None

    if model_name_or_path in _HF_CAUSAL_MODELS or _is_hf_model_dir(model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    else:
        # Project adapter checkpoint (.pt) – delegate to the project loader.
        try:
            from llm_adapter import load_model as _load_adapter, save_model as _save_adapter
        except ImportError as exc:
            raise ImportError(
                "llm_adapter is not installed.  Run `pip install -e .` from the "
                "project root, or pass a standard HuggingFace model name."
            ) from exc

        # Read metadata before loading so the save_fn closure can reconstruct
        # the same payload without needing the original checkpoint path later.
        raw = torch.load(model_name_or_path, map_location="cpu", weights_only=False)
        _adapter_type   = raw["adapter_type"]
        _adapter_config = raw["adapter_config"]
        _wechsel_config = raw.get("wechsel_config", None)
        del raw  # free before loading the full model

        model, tokenizer, _ = _load_adapter(model_name_or_path)

        # Closure that mirrors llm_adapter.save_model but targets a directory:
        # writes <output_dir>/model.pt + tokenizer files to <output_dir>/.
        def save_fn(
            m: torch.nn.Module,
            output_dir: str,
            tok: PreTrainedTokenizer,
            *,
            adapter_type=_adapter_type,
            adapter_config=_adapter_config,
            wechsel_config=_wechsel_config,
        ) -> str:
            os.makedirs(output_dir, exist_ok=True)
            pt_path = os.path.join(output_dir, "model.pt")
            _save_adapter(
                m,
                adapter_type=adapter_type,
                adapter_config=adapter_config,
                save_path=pt_path,
                wechsel_config=wechsel_config,
                tokenizer=tok,
                tokenizer_path=output_dir,
            )
            return pt_path

    # GPT-2 family has no pad token by default; reuse eos_token for padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True

    return model, tokenizer, save_fn


def _is_hf_model_dir(path: str) -> bool:
    """Return True if *path* looks like a directory saved with save_pretrained."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class CausalLMCollator:
    """
    Collate variable-length causal-LM training examples into a padded batch.

    Strategy:
      - input_ids     : right-padded with pad_token_id
      - attention_mask: right-padded with 0
      - labels        : right-padded with -100  (ignored by CrossEntropyLoss)

    The dataset already sets prompt token positions to -100 in *labels*, so
    the loss is computed exclusively on the target (holding) tokens.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        max_len = max(e["input_ids"].shape[0] for e in examples)

        input_ids_out, attention_mask_out, labels_out = [], [], []

        for e in examples:
            seq_len = e["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids_out.append(torch.cat([
                e["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
            ]))
            attention_mask_out.append(torch.cat([
                e["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ]))
            labels_out.append(torch.cat([
                e["labels"],
                torch.full((pad_len,), -100, dtype=torch.long),  # ignore padding
            ]))

        return {
            "input_ids": torch.stack(input_ids_out),
            "attention_mask": torch.stack(attention_mask_out),
            "labels": torch.stack(labels_out),
        }