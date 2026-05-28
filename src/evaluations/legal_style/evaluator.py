"""
Legal-style evaluator.

Generates text from a causal LM using a set of neutral prompts and then
classifies each generation with a pre-trained TF-IDF + logistic regression
classifier to measure how "legal" the model's output style is.

The classifier (legal_style_clf.joblib) lives alongside this file.  If it is
missing the evaluator will attempt to train one on the fly from streaming data.

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluations.utils import load_model

# Path to the pre-trained sklearn classifier bundled with this evaluator.
_CLF_PATH = Path(__file__).parent / "legal_style_clf.joblib"

# Default prompts used to generate model outputs for scoring.
_DEFAULT_PROMPTS = [
    "The court held that",
    "Pursuant to the agreement,",
    "The defendant argued that",
    "In accordance with applicable law,",
    "The contract stipulates that",
    "The plaintiff submitted evidence that",
    "It is hereby ordered that",
    "The statute provides that",
    "As set forth in the opinion,",
    "The tribunal concluded that",
]


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

def _load_classifier(clf_path: Path):
    """Load classifier from disk; raises FileNotFoundError if missing."""
    if not clf_path.exists():
        raise FileNotFoundError(
            f"Legal-style classifier not found at {clf_path}.\n"
            "Train it first with:\n"
            "  python src/evaluations/style/legal_style_classifier_streaming.py train"
        )
    return joblib.load(clf_path)


# ---------------------------------------------------------------------------
# Text generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_texts(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int = 80,
) -> List[str]:
    """Generate one completion per prompt."""
    texts = []

    for prompt in tqdm(prompts, desc="Legal-style: generating"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
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

        # Include the prompt for richer classification context.
        texts.append(prompt + " " + generated)

    return texts


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

def _run_evaluation(
    model,
    tokenizer,
    device: str,
    clf,
    prompts: List[str],
    max_new_tokens: int = 80,
) -> Dict[str, Any]:
    texts = _generate_texts(model, tokenizer, prompts, device, max_new_tokens)

    legal_probs = clf.predict_proba(texts)[:, 1].tolist()
    preds = clf.predict(texts).tolist()

    results = [
        {
            "prompt": p,
            "generated_text": t,
            "legal_style_score": float(lp),
            "predicted_label": "legal" if pred == 1 else "general",
        }
        for p, t, lp, pred in zip(prompts, texts, legal_probs, preds)
    ]

    return {
        "mean_legal_style_score": float(np.mean(legal_probs)),
        "median_legal_style_score": float(np.median(legal_probs)),
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
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 80,
    max_examples: Optional[int] = None,
    **kwargs,
) -> dict:
    """
    Measure legal-style score of a causal LM's generated text.

    The model generates continuations for a set of neutral prompts; a
    pre-trained TF-IDF/logistic-regression classifier then scores each
    generation as "legal" vs. "general".

    Args:
        model_name_or_path: HuggingFace model name or local adapter path.
        device: Torch device string.
        clf_path: Optional path to a .joblib classifier file.  Defaults to
                  the one bundled in this directory.
        prompts: List of prompt strings.  Defaults to built-in legal prompts.
        max_new_tokens: Tokens to generate per prompt.
        max_examples: Cap on the number of prompts used.

    Returns:
        dict with scalar/printable metrics:
            mean_legal_style_score    – average P(legal) across generated texts
            median_legal_style_score  – median P(legal)
            predicted_legal_rate      – fraction classified as "legal"
            num_examples              – number of prompts used
    """
    clf = _load_classifier(Path(clf_path) if clf_path else _CLF_PATH)
    model, tokenizer = load_model(model_name_or_path, device)

    active_prompts = list(prompts or _DEFAULT_PROMPTS)
    if max_examples is not None:
        active_prompts = active_prompts[:max_examples]

    output = _run_evaluation(model, tokenizer, device, clf, active_prompts, max_new_tokens)

    return {
        "mean_legal_style_score": output["mean_legal_style_score"],
        "median_legal_style_score": output["median_legal_style_score"],
        "predicted_legal_rate": output["predicted_legal_rate"],
        "num_examples": output["num_examples"],
        "_raw": output,
    }
