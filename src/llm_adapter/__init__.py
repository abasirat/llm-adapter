"""
LLM Adapter - A library for adapting language models to new languages and tasks.

Modules:
- autoregressive_adapt: GPT-style models adaptation
- encoder_adapt: Encoder models (BERT-style) adaptation
"""

# Autoregressive Adapt Imports
from .autoregressive_adapt.model_setup import (
    setup_model,
    load_learnable_params,
    save_learnable_params,
    print_trainable_parameters,
    train_tokenizer,
)
from .autoregressive_adapt.layer_adapter import LayerAdapter
from .autoregressive_adapt.language_tailor import LanguageAdapter

# Encoder Adapt Imports
from .encoder_adapt.adapter import Adapter, Tailor

__all__ = [
    # Autoregressive Adapt
    "setup_model",
    "load_learnable_params",
    "save_learnable_params",
    "print_trainable_parameters",
    "train_tokenizer",
    "LayerAdapter",
    "LanguageAdapter",
    # Encoder Adapt
    "Adapter",
    "Tailor",
]

__version__ = "0.1.0"
