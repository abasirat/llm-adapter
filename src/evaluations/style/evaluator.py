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

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from evaluations.utils import load_model

_TRAINING_SCRIPT = Path(__file__).parent / "legal_style_classifier_streaming.py"

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
            "  python src/evaluations/style/legal_style_classifier_streaming.py train\n"
            "or call evaluations.style.evaluator.train_classifier()."
        )
    return joblib.load(clf_path)


def train_classifier(
    output_model: Optional[str] = None,
    legal_dataset: str = "pile-of-law/pile-of-law",
    legal_subset: str = "courtlistener_opinions",
    legal_split: str = "train",
    legal_text_field: str = "text",
    general_dataset: str = "HuggingFaceFW/fineweb",
    general_subset: str = "sample-10BT",
    general_split: str = "train",
    general_text_field: str = "text",
    max_samples: int = 50000,
    min_chars: int = 100,
    max_chars: Optional[int] = 2000,
    test_size: float = 0.2,
    seed: int = 42,
    max_ngram: int = 2,
    min_df: int = 3,
    max_df: float = 0.95,
    max_features: int = 100000,
    max_iter: int = 1000,
    top_k: int = 30,
) -> str:
    """
    Train the TF-IDF + logistic regression legal/general style classifier.

    Delegates to src/evaluations/style/legal_style_classifier_streaming.py
    so there is no code duplication.

    Args:
        output_model: Destination path for the .joblib file.  Defaults to the
                      bundled classifier path (evaluations/style/legal_style_clf.joblib).
        legal_dataset/legal_subset/legal_split/legal_text_field:
            Source for legal training samples (streaming).
        general_dataset/general_subset/general_split/general_text_field:
            Source for general training samples (streaming).
        max_samples:  Maximum examples to stream per class.
        min_chars:    Minimum character length to accept a document.
        max_chars:    Truncate documents to this many characters (None = no limit).
        test_size:    Fraction held out for validation.
        seed:         Random seed.
        max_ngram:    Upper n-gram bound for TF-IDF.
        min_df / max_df / max_features: TF-IDF vocabulary parameters.
        max_iter:     Logistic regression max iterations.
        top_k:        Top features to print after training.

    Returns:
        Absolute path to the saved .joblib classifier file.
    """
    if not _TRAINING_SCRIPT.exists():
        raise FileNotFoundError(
            f"Training script not found at {_TRAINING_SCRIPT}. "
            "Ensure the src/evaluations/style directory is present."
        )

    out_path = Path(output_model) if output_model else _CLF_PATH

    spec = importlib.util.spec_from_file_location(
        "legal_style_classifier_streaming", _TRAINING_SCRIPT
    )
    training_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training_module)

    args = SimpleNamespace(
        legal_dataset=legal_dataset,
        legal_subset=legal_subset,
        legal_split=legal_split,
        legal_text_field=legal_text_field,
        general_dataset=general_dataset,
        general_subset=general_subset,
        general_split=general_split,
        general_text_field=general_text_field,
        max_samples=max_samples,
        min_chars=min_chars,
        max_chars=max_chars,
        test_size=test_size,
        seed=seed,
        max_ngram=max_ngram,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        max_iter=max_iter,
        top_k=top_k,
        output_model=str(out_path),
    )

    training_module.train_classifier(args)
    return str(out_path)


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
    expected_label: int,
) -> Dict[str, Any]:
    """Classify texts and return per-condition metrics plus per-example results.

    Args:
        expected_label: The ground-truth class for all texts in this condition.
                        1 for the legal condition, 0 for the general condition.
                        Used to compute precision, recall, and F1.
    """
    probs = clf.predict_proba(texts)
    general_probs = probs[:, 0].tolist()
    legal_probs   = probs[:, 1].tolist()
    preds = clf.predict(texts).tolist()

    true_labels = [expected_label] * len(preds)

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
        "precision": float(precision_score(true_labels, preds, pos_label=expected_label, zero_division=0)),
        "recall":    float(recall_score(   true_labels, preds, pos_label=expected_label, zero_division=0)),
        "f1":        float(f1_score(       true_labels, preds, pos_label=expected_label, zero_division=0)),
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
            legal_precision    – precision for legal class on legal-prompt outputs
            legal_recall       – recall for legal class on legal-prompt outputs
            legal_f1           – F1 for legal class on legal-prompt outputs
            legal_num_examples
            general_mean_legal_style_score
            general_mean_general_style_score
            general_predicted_general_rate
            general_precision  – precision for general class on general-prompt outputs
            general_recall     – recall for general class on general-prompt outputs
            general_f1         – F1 for general class on general-prompt outputs
            general_num_examples
    """
    should_train = kwargs.pop("train_classifier", False)
    training_kwargs = kwargs.pop("classifier_training", {}) or {}
    if should_train:
        _out_path = training_kwargs.pop("output_model", None) or (
            str(Path(clf_path)) if clf_path else str(_CLF_PATH)
        )
        train_classifier(output_model=_out_path, **training_kwargs)

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

    legal_scores   = _score_condition(clf, active_legal,   legal_texts,   condition="legal",   expected_label=1)
    general_scores = _score_condition(clf, active_general, general_texts, condition="general", expected_label=0)

    return {
        # Legal-prompt condition (ground truth = legal, pos_label=1)
        "legal_mean_legal_style_score":   legal_scores["mean_legal_style_score"],
        "legal_mean_general_style_score": legal_scores["mean_general_style_score"],
        "legal_predicted_legal_rate":     legal_scores["predicted_legal_rate"],
        "legal_precision":                legal_scores["precision"],
        "legal_recall":                   legal_scores["recall"],
        "legal_f1":                       legal_scores["f1"],
        "legal_num_examples":             legal_scores["num_examples"],
        # General-prompt condition (ground truth = general, pos_label=0)
        "general_mean_legal_style_score":   general_scores["mean_legal_style_score"],
        "general_mean_general_style_score": general_scores["mean_general_style_score"],
        "general_predicted_general_rate":   float(1.0 - general_scores["predicted_legal_rate"]),
        "general_precision":                general_scores["precision"],
        "general_recall":                   general_scores["recall"],
        "general_f1":                       general_scores["f1"],
        "general_num_examples":             general_scores["num_examples"],
        "_raw": {
            "legal":   legal_scores,
            "general": general_scores,
        },
    }
