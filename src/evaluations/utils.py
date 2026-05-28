"""
Shared utilities for all evaluators.

Provides helpers for:
- Loading a HuggingFace or adapter model/tokenizer
- Device selection
- Safe JSON saving
"""

import json
import sys
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Standard HuggingFace causal model names that can be loaded directly.
_HF_CAUSAL_MODELS = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}


def _is_hf_model_dir(path: str) -> bool:
    """Return True if *path* is a directory saved with save_pretrained."""
    p = Path(path)
    return p.is_dir() and (p / "config.json").exists()


def select_device(device: str = "auto") -> str:
    """Return a concrete device string (cuda / mps / cpu)."""
    if device not in (None, "auto"):
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_name_or_path: str,
    device: str,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a causal language model and its tokenizer.

    For well-known HuggingFace model names (gpt2, gpt2-medium, …) this uses
    AutoModelForCausalLM directly.  For local adapter checkpoints it falls
    back to the project's llm_adapter.load_model helper.

    Returns:
        (model, tokenizer)  —  model is already moved to *device* and in eval mode.
    """
    if model_name_or_path in _HF_CAUSAL_MODELS or _is_hf_model_dir(model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        _set_pad_token(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    else:
        # Local adapter checkpoint – delegate to the project loader.
        try:
            from llm_adapter import load_model as load_adapter_model
        except ImportError as exc:
            raise ImportError(
                "llm_adapter is not installed.  Run `pip install -e .` from the "
                "project root, or pass a standard HuggingFace model name."
            ) from exc

        model, tokenizer, _ = load_adapter_model(model_name_or_path)
        _set_pad_token(tokenizer)

    model.to(device)
    model.eval()
    return model, tokenizer


def _set_pad_token(tokenizer: AutoTokenizer) -> None:
    """Set pad_token to eos_token if it is missing (common for GPT-2 family)."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def safe_save_json(data: dict, output_file: str) -> None:
    """Write *data* as pretty-printed JSON to *output_file*, creating dirs as needed."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
