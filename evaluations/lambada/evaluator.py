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
def _predict_next_word(model, tokenizer, context: str, device: str, max_new_tokens: int = 5) -> str:
    """Generate a short continuation and return its first word."""
    inputs = tokenizer(context, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    if not generated:
        return ""

    # Strip punctuation that may be attached to the word.
    word = generated.split()[0]
    return word.strip(".,!?;:\"'()[]{}")


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

def _run_evaluation(
    model,
    tokenizer,
    device: str,
    split: str = "test",
    max_examples: Optional[int] = None,
    max_new_tokens: int = 5,
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

        prediction = _predict_next_word(model, tokenizer, context, device, max_new_tokens)
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
    max_new_tokens: int = 5,
    **kwargs,
) -> dict:
    """
    Evaluate a causal LM on the LAMBADA last-word prediction benchmark.

    Args:
        model_name_or_path: HuggingFace model name or local adapter checkpoint path.
        device: Torch device string, e.g. "cuda", "mps", "cpu".
        split: Dataset split to use (default "test").
        max_examples: Optional cap on number of examples evaluated.
        max_new_tokens: Tokens to generate when predicting the next word.

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
        max_new_tokens=max_new_tokens,
    )

    return {
        "accuracy": output["accuracy"],
        "num_examples": output["num_examples"],
        "_raw": output,
    }
