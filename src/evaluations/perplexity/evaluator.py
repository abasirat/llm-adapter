"""
Perplexity evaluator.

Measures average perplexity of a causal language model on datasets defined
under the ``perplexity.datasets`` section of the evaluation config.

Each dataset entry requires:
  - name:  short alias (e.g. pile_of_law, wikipedia) or a HuggingFace path.
  - split: dataset split to load (e.g. test, validation, train).

Optional per-dataset fields:
  - subset:     HuggingFace dataset config/subset name.
  - text_field: column that contains the raw text (default: ``"text"``).

A built-in registry maps common aliases to their HuggingFace identifiers and
default text fields; unrecognised names are treated as HuggingFace paths
directly.

Perplexity is computed with a sliding-window approach so that documents longer
than the model's context window are handled correctly.

Public interface:
    evaluate(model_name_or_path, device, **kwargs) -> dict
"""

from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from evaluations.utils import load_model


# ---------------------------------------------------------------------------
# Built-in dataset registry
# Maps short alias -> default HuggingFace load_dataset kwargs + text_field.
# ---------------------------------------------------------------------------

_DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pile_of_law": {
        "path": "pile-of-law/pile-of-law",
        "name": "all",
        "text_field": "text",
    },
    "wikipedia": {
        "path": "wikipedia",
        "name": "20220301.en",
        "text_field": "text",
    },
}


# ---------------------------------------------------------------------------
# Sliding-window perplexity
# ---------------------------------------------------------------------------

def _perplexity(
    model,
    tokenizer,
    text: str,
    device: str,
    max_length: int = 1024,
    stride: int = 512,
) -> float:
    """
    Compute the perplexity of *text* using a sliding-window approach.

    For texts shorter than *max_length* tokens a single forward pass is used.
    For longer texts the window slides by *stride* tokens and the NLL is
    accumulated only over the non-overlapping (new) portion of each window,
    which is the standard method for evaluating LM perplexity on long docs.
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls: List[torch.Tensor] = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # tokens actually scored in this window

        window_ids = input_ids[:, begin:end]

        # Mask the overlapping prefix so it does not contribute to the loss.
        labels = window_ids.clone()
        labels[:, : window_ids.size(1) - target_len] = -100

        with torch.no_grad():
            loss = model(window_ids, labels=labels).loss

        nlls.append(loss * target_len)
        prev_end = end

        if end == seq_len:
            break

    if prev_end == 0:
        return float("inf")

    avg_nll = torch.stack(nlls).sum() / prev_end
    return torch.exp(avg_nll).item()


def _avg_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int = 1024,
    stride: int = 512,
    desc: str = "",
) -> Dict[str, Any]:
    """Compute average perplexity over *texts*, skipping samples that error."""
    values: List[float] = []

    for text in tqdm(texts, desc=desc or "Perplexity"):
        try:
            ppl = _perplexity(model, tokenizer, text, device, max_length, stride)
            if ppl != float("inf") and ppl == ppl:  # skip inf and NaN
                values.append(ppl)
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
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _resolve_dataset_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve a YAML dataset spec to ``load_dataset`` kwargs and a text field.

    The spec must contain at least ``name`` and ``split``.  Optional keys
    ``subset`` and ``text_field`` override registry defaults.
    """
    alias = spec["name"]
    split = spec["split"]

    registry_entry = dict(_DATASET_REGISTRY.get(alias, {"path": alias}))
    text_field = registry_entry.pop("text_field", "text")

    if "subset" in spec:
        registry_entry["name"] = spec["subset"]
    if "text_field" in spec:
        text_field = spec["text_field"]

    load_kwargs = {**registry_entry, "split": split}
    return {"load_kwargs": load_kwargs, "text_field": text_field, "alias": alias}


def _load_texts(
    load_kwargs: Dict[str, Any],
    text_field: str,
    max_samples: int,
) -> List[str]:
    """Stream up to *max_samples* non-empty texts from a HuggingFace dataset."""
    try:
        ds = load_dataset(**load_kwargs, streaming=True, trust_remote_code=True)
        texts: List[str] = []
        for ex in ds:
            if len(texts) >= max_samples:
                break
            val = ex.get(text_field, "")
            if isinstance(val, str) and val.strip():
                texts.append(val)
        return texts
    except Exception as exc:
        print(f"  [perplexity] could not load dataset {load_kwargs!r}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate(
    model_name_or_path: str,
    device: str,
    datasets: Optional[List[Dict[str, Any]]] = None,
    max_examples: Optional[int] = None,
    max_length: int = 1024,
    stride: int = 512,
    **kwargs,
) -> dict:
    """
    Evaluate perplexity of a causal LM on each dataset in *datasets*.

    Args:
        model_name_or_path: HuggingFace model name or local adapter path.
        device: Torch device string.
        datasets: List of dataset spec dicts corresponding to the
                  ``perplexity.datasets`` list in the evaluation config.
                  Each dict must have ``name`` and ``split``; optional keys
                  are ``subset`` and ``text_field``.
        max_examples: Maximum number of examples per dataset.  Defaults to 500.
        max_length: Token window size for sliding-window perplexity.
        stride: Stride for the sliding window (must be <= *max_length*).

    Returns:
        dict with per-dataset scalar metrics (dataset name used as prefix)::

            <name>_avg_perplexity
            <name>_num_samples

        plus a ``"_raw"`` key containing full per-dataset statistics.
    """
    if not datasets:
        print("[perplexity] No datasets configured; skipping.")
        return {"_raw": {}}

    n_samples = max_examples if max_examples is not None else 500

    model, tokenizer = load_model(model_name_or_path, device)

    raw: Dict[str, Any] = {}

    for spec in datasets:
        resolved = _resolve_dataset_spec(spec)
        alias = resolved["alias"]
        subset = spec.get("subset")
        prefix = f"{alias}[{subset}]" if subset else alias
        label = f"{prefix}/{spec['split']}"

        print(f"\n[perplexity] Loading {label} ...")
        texts = _load_texts(resolved["load_kwargs"], resolved["text_field"], n_samples)

        if not texts:
            raw[prefix] = {"error": f"no samples available for '{label}'"}
            continue

        stats = _avg_perplexity(
            model, tokenizer, texts, device,
            max_length=max_length,
            stride=stride,
            desc=f"Perplexity ({label})",
        )
        stats["dataset"] = label
        raw[prefix] = stats

    # Flatten to scalar metrics for the summary table.
    flat: Dict[str, Any] = {}
    for spec in datasets:
        alias = spec["name"]
        subset = spec.get("subset")
        prefix = f"{alias}[{subset}]" if subset else alias
        stats = raw.get(prefix, {})
        if "error" in stats:
            flat[f"{prefix}_avg_perplexity"] = None
        else:
            flat[f"{prefix}_avg_perplexity"] = stats.get("avg_perplexity")
            flat[f"{prefix}_num_samples"] = stats.get("num_samples")

    flat["_raw"] = raw
    return flat
