"""
LAMBADA evaluator.

Tests whether a causal language model can predict the final word of a passage
given its context.  Uses exact-match after simple normalization.

Public interface:
    evaluate(model_name_or_path, device, **kwargs) -> dict
"""

import re
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from evaluations.utils import load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_context_target(text: str):
    """Return (context, target) where target is the final word of the passage."""
    text = text.strip()
    words = text.split()
    if len(words) < 2:
        return None, None
    return " ".join(words[:-1]), words[-1]


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for exact-match comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


@torch.no_grad()
def _predict_next_word(model, tokenizer, context: str, device: str) -> str:
    """
    Predict the most likely next token using a single forward pass.

    The logit at the last context position is read and argmax is taken over
    the full vocabulary.  Equivalent to greedy one-step decoding but avoids
    the overhead of model.generate().
    """
    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    logits = model(context_ids).logits        # (1, seq_len, vocab)
    next_token_id = int(logits[0, -1, :].argmax())
    predicted = tokenizer.decode([next_token_id], skip_special_tokens=True).strip()
    return predicted.strip(".,!?;:\"'()[]{}")


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

def _run_evaluation(
    model,
    tokenizer,
    device: str,
    split: str = "test",
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    dataset = load_dataset("lambada", split=split, streaming=True)

    results = []
    correct = 0

    for i, ex in enumerate(tqdm(dataset, desc="LAMBADA")):
        if max_examples is not None and i >= max_examples:
            break

        context, target = _split_context_target(ex["text"])
        if context is None:
            continue

        prediction = _predict_next_word(model, tokenizer, context, device)
        is_correct = _normalize(prediction) == _normalize(target)
        correct += int(is_correct)

        results.append(
            {
                "context": context,
                "target": target,
                "prediction": prediction,
                "correct": is_correct,
            }
        )

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "num_examples": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate(
    model_name_or_path: str,
    device: str,
    split: str = "test",
    max_examples: Optional[int] = None,
    **kwargs,
) -> dict:
    """
    Evaluate a causal LM on the LAMBADA last-word prediction benchmark.

    Args:
        model_name_or_path: HuggingFace model name or local adapter checkpoint path.
        device: Torch device string, e.g. "cuda", "mps", "cpu".
        split: Dataset split to use (default "test").
        max_examples: Optional cap on number of examples evaluated.

    Returns:
        dict with scalar/printable metrics:
            accuracy      – fraction of passages with correct last-word prediction
            num_examples  – number of passages evaluated
    """
    model, tokenizer = load_model(model_name_or_path, device)
    output = _run_evaluation(
        model, tokenizer, device,
        split=split,
        max_examples=max_examples,
    )

    return {
        "accuracy": output["accuracy"],
        "num_examples": output["num_examples"],
        "_raw": output,
    }
