"""
Perplexity evaluator.

Measures average perplexity of a causal language model on three language
corpora drawn from XNLI:
  - English  (en)
  - Danish   (da)
  - Farsi    (fa)

Lower perplexity on a language corpus indicates better language-model fit.

Public interface:
    evaluate(model_name_or_path, device, **kwargs) -> dict
"""

from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from evaluations.utils import load_model


# ---------------------------------------------------------------------------
# Per-text perplexity
# ---------------------------------------------------------------------------

def _perplexity(model, tokenizer, text: str, device: str) -> float:
    """Return the perplexity of *text* under *model*."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss

    return torch.exp(loss).item()


def _avg_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    desc: str = "",
) -> Dict[str, Any]:
    """Compute average perplexity over a list of texts, skipping errors."""
    values: List[float] = []

    for text in tqdm(texts, desc=desc or "Perplexity"):
        try:
            values.append(_perplexity(model, tokenizer, text, device))
        except Exception as exc:
            print(f"  [perplexity] skipping sample: {exc}")

    if not values:
        return {"avg_perplexity": float("inf"), "num_samples": 0}

    return {
        "avg_perplexity": sum(values) / len(values),
        "min_perplexity": min(values),
        "max_perplexity": max(values),
        "num_samples": len(values),
    }


# ---------------------------------------------------------------------------
# Language corpus loaders
# ---------------------------------------------------------------------------

_XNLI_LANGS = {
    "en": "English",
    "da": "Danish",
    "fa": "Farsi",
}


def _load_xnli_texts(lang: str, max_samples: int = 500) -> List[str]:
    """
    Load up to *max_samples* premise strings from the XNLI validation split
    for *lang*.  Returns an empty list if the language is unavailable.
    """
    try:
        ds = load_dataset("xnli", lang, split="validation")
        texts = [ex["premise"][:500] for ex in ds][:max_samples]
        return texts
    except Exception as exc:
        print(f"  [perplexity] could not load XNLI '{lang}': {exc}")
        return []


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

def _run_evaluation(
    model,
    tokenizer,
    device: str,
    languages: List[str],
    max_samples: int = 500,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    for lang in languages:
        lang_name = _XNLI_LANGS.get(lang, lang)
        texts = _load_xnli_texts(lang, max_samples)

        if not texts:
            results[lang] = {"error": f"no samples available for '{lang}'"}
            continue

        stats = _avg_perplexity(
            model, tokenizer, texts, device,
            desc=f"Perplexity ({lang_name})",
        )
        stats["language"] = lang_name
        stats["corpus"] = f"XNLI ({lang_name})"
        results[lang] = stats

    return results


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate(
    model_name_or_path: str,
    device: str,
    languages: Optional[List[str]] = None,
    max_samples: int = 500,
    max_examples: Optional[int] = None,
    **kwargs,
) -> dict:
    """
    Evaluate perplexity of a causal LM on language-specific XNLI corpora.

    Args:
        model_name_or_path: HuggingFace model name or local adapter path.
        device: Torch device string.
        languages: List of XNLI language codes to evaluate.
                   Defaults to ["en", "da", "fa"].
        max_samples: Number of text snippets to evaluate per language.
        max_examples: Alias for max_samples (used by the central runner).

    Returns:
        dict with per-language scalar metrics prefixed by language code:
            en_avg_perplexity, da_avg_perplexity, fa_avg_perplexity, …
    """
    if languages is None:
        languages = ["en", "da", "fa"]

    n_samples = max_examples if max_examples is not None else max_samples

    model, tokenizer = load_model(model_name_or_path, device)
    raw = _run_evaluation(model, tokenizer, device, languages, n_samples)

    # Flatten to scalar metrics for the summary table.
    flat: Dict[str, Any] = {}
    for lang, stats in raw.items():
        if "error" in stats:
            flat[f"{lang}_avg_perplexity"] = None
        else:
            flat[f"{lang}_avg_perplexity"] = stats.get("avg_perplexity")
            flat[f"{lang}_num_samples"] = stats.get("num_samples")

    flat["_raw"] = raw
    return flat
