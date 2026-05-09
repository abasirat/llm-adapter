"""
Style evaluator.

Measures legal-style and general-style scores for a causal LM in a single
pass.  Two prompt sets are used:

  legal_prompts   – domain-specific legal phrases
  general_prompts – everyday, non-legal phrases

For each set the model generates continuations that are then classified by a
pre-trained TF-IDF + logistic regression binary classifier (legal vs. general).
Both P(legal) and P(general) are reported for each condition, giving a clear
picture of how well the model's style adapts to context.

The classifier (legal_style_clf.joblib) lives alongside this file.

Public interface:
    evaluate(model_name_or_path, device, **kwargs) -> dict
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch
from tqdm import tqdm

from evaluations.utils import load_model

_CLF_PATH          = Path(__file__).parent / "legal_style_clf.joblib"
_LEGAL_PROMPTS_DIR  = Path(__file__).parent / "legal_prompts"
_GENERAL_PROMPTS_DIR = Path(__file__).parent / "general_prompts"

# Hard-coded fallbacks used only if the bundled prompt directories are
# somehow unavailable (e.g. during unit tests without the full repo).
_LEGAL_PROMPTS_FALLBACK = [
    "The court held that",
    "Pursuant to the agreement,",
    "The defendant argued that",
    "In accordance with applicable law,",
    "The statute provides that",
]

_GENERAL_PROMPTS_FALLBACK = [
    "How do you make pancakes?",
    "The weather today is",
    "Scientists recently discovered that",
    "The best way to learn a new language is",
    "Technology has changed the way people",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_prompts_from_lines(lines: List[str], source_suffix: str) -> List[str]:
    """Parse prompt strings from a list of raw text lines (txt or jsonl)."""
    prompts: List[str] = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        if source_suffix == ".jsonl" or raw.startswith("{"):
            obj = json.loads(raw)
            # Prefer a key whose name contains "prompt".
            value = next(
                (v for k, v in obj.items() if "prompt" in k.lower() and isinstance(v, str)),
                next((v for v in obj.values() if isinstance(v, str)), None),
            )
            if value:
                prompts.append(value)
        else:
            prompts.append(raw)
    return prompts


def _load_prompts_from_file(path: Path) -> List[str]:
    """
    Load prompts from a single plain-text (.txt) or JSONL (.jsonl) file.

    Plain-text: one prompt per non-empty line.
    JSONL: each line is a JSON object; the value of the first field whose key
           contains "prompt" is used, falling back to the first string value.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = _parse_prompts_from_lines(lines, path.suffix.lower())
    if not prompts:
        raise ValueError(f"No prompts found in file: {path}")
    return prompts


def _load_prompts_from_dir(directory: Path) -> List[str]:
    """
    Load prompts from all .txt and .jsonl files inside *directory*.

    Files are read in sorted order so results are reproducible.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Prompts directory not found: {directory}")
    files = sorted(directory.glob("*.txt")) + sorted(directory.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No .txt or .jsonl prompt files found in: {directory}")
    prompts: List[str] = []
    for f in files:
        prompts.extend(_parse_prompts_from_lines(
            f.read_text(encoding="utf-8").splitlines(), f.suffix.lower()
        ))
    if not prompts:
        raise ValueError(f"No prompts loaded from directory: {directory}")
    return prompts


def _resolve_prompts(
    prompts: Optional[List[str]],
    path_or_dir: Optional[str],
    default_dir: Path,
) -> List[str]:
    """
    Resolve the active prompt list with the following priority:
      1. Explicit list (``prompts`` argument)
      2. ``path_or_dir`` — a file path or directory path
      3. ``default_dir`` — the bundled prompts directory
    """
    if prompts is not None:
        return list(prompts)
    if path_or_dir is not None:
        p = Path(path_or_dir)
        if p.is_dir():
            return _load_prompts_from_dir(p)
        return _load_prompts_from_file(p)
    if default_dir.is_dir():
        return _load_prompts_from_dir(default_dir)
    # Bundled directory unavailable — use hard-coded fallbacks.
    fallback = (
        _LEGAL_PROMPTS_FALLBACK
        if "legal" in default_dir.name
        else _GENERAL_PROMPTS_FALLBACK
    )
    return list(fallback)


def _load_classifier(clf_path: Path):
    if not clf_path.exists():
        raise FileNotFoundError(
            f"Style classifier not found at {clf_path}.\n"
            "Train it first with:\n"
            "  python model_eval/doc_classification/legal_style_classifier_streaming.py train"
        )
    return joblib.load(clf_path)


@torch.no_grad()
def _generate_texts(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    min_new_tokens: int,
    max_new_tokens: int,
    desc: str = "",
) -> List[str]:
    texts = []
    for prompt in tqdm(prompts, desc=desc):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
        generated = tokenizer.decode(
            output_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        texts.append(generated)
    return texts


def _score_condition(
    clf,
    prompts: List[str],
    texts: List[str],
    condition: str,
) -> Dict[str, Any]:
    """Classify texts and return per-condition metrics plus per-example results."""
    probs = clf.predict_proba(texts)
    general_probs = probs[:, 0].tolist()
    legal_probs   = probs[:, 1].tolist()
    preds = clf.predict(texts).tolist()

    results = [
        {
            "prompt": p,
            "generated_text": t,
            "legal_style_score": float(lp),
            "general_style_score": float(gp),
            "predicted_label": "legal" if pred == 1 else "general",
        }
        for p, t, lp, gp, pred in zip(prompts, texts, legal_probs, general_probs, preds)
    ]

    return {
        "condition": condition,
        "mean_legal_style_score": float(np.mean(legal_probs)),
        "median_legal_style_score": float(np.median(legal_probs)),
        "mean_general_style_score": float(np.mean(general_probs)),
        "median_general_style_score": float(np.median(general_probs)),
        "predicted_legal_rate": float(np.mean(preds)),
        "num_examples": len(texts),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate(
    model_name_or_path: str,
    device: str,
    clf_path: Optional[str] = None,
    legal_prompts: Optional[List[str]] = None,
    legal_prompts_path: Optional[str] = None,
    general_prompts: Optional[List[str]] = None,
    general_prompts_path: Optional[str] = None,
    min_new_tokens: int = 20,
    max_new_tokens: int = 80,
    max_examples: Optional[int] = None,
    **kwargs,
) -> dict:
    """
    Measure legal-style and general-style scores of a causal LM.

    Two prompt conditions are evaluated in a single call:

      legal condition   – legal-domain prompts; high P(legal) is expected for
                          a well-adapted legal model.
      general condition – everyday prompts; high P(general) is expected for a
                          baseline or well-calibrated model.

    Args:
        model_name_or_path: HuggingFace model name or local adapter path.
        device: Torch device string.
        clf_path: Optional path to a .joblib classifier file.
        legal_prompts: Explicit list of legal prompts (highest priority).
        legal_prompts_path: Path to a .txt/.jsonl file **or** a directory of
                            such files containing legal prompts.  Defaults to
                            evaluations/style/legal_prompts/.
        general_prompts: Explicit list of general prompts (highest priority).
        general_prompts_path: Path to a .txt/.jsonl file **or** a directory of
                              such files containing general prompts.  Defaults
                              to evaluations/style/general_prompts/.
        max_new_tokens: Maximum tokens to generate per prompt.
        min_new_tokens: Minimum tokens to generate per prompt.
        max_examples: Cap on prompts per condition.

    Returns:
        dict with scalar/printable metrics (prefixed by condition):
            legal_mean_legal_style_score
            legal_mean_general_style_score
            legal_predicted_legal_rate
            legal_num_examples
            general_mean_legal_style_score
            general_mean_general_style_score
            general_predicted_general_rate
            general_num_examples
    """
    clf = _load_classifier(Path(clf_path) if clf_path else _CLF_PATH)
    model, tokenizer = load_model(model_name_or_path, device)

    # Prompt resolution: explicit list > file/dir argument > bundled directory.
    active_legal   = _resolve_prompts(legal_prompts,   legal_prompts_path,   _LEGAL_PROMPTS_DIR)
    active_general = _resolve_prompts(general_prompts, general_prompts_path, _GENERAL_PROMPTS_DIR)
    if max_examples is not None:
        active_legal   = active_legal[:max_examples]
        active_general = active_general[:max_examples]

    legal_texts   = _generate_texts(model, tokenizer, active_legal,   device, min_new_tokens, max_new_tokens, desc="Style (legal prompts)")
    general_texts = _generate_texts(model, tokenizer, active_general, device, min_new_tokens, max_new_tokens, desc="Style (general prompts)")

    legal_scores   = _score_condition(clf, active_legal,   legal_texts,   condition="legal")
    general_scores = _score_condition(clf, active_general, general_texts, condition="general")

    return {
        # Legal-prompt condition
        "legal_mean_legal_style_score":   legal_scores["mean_legal_style_score"],
        "legal_mean_general_style_score": legal_scores["mean_general_style_score"],
        "legal_predicted_legal_rate":     legal_scores["predicted_legal_rate"],
        "legal_num_examples":             legal_scores["num_examples"],
        # General-prompt condition
        "general_mean_legal_style_score":   general_scores["mean_legal_style_score"],
        "general_mean_general_style_score": general_scores["mean_general_style_score"],
        "general_predicted_general_rate":   float(1.0 - general_scores["predicted_legal_rate"]),
        "general_num_examples":             general_scores["num_examples"],
        "_raw": {
            "legal":   legal_scores,
            "general": general_scores,
        },
    }
