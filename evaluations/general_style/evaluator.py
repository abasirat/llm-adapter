"""
General-style evaluator.

Generates text from a causal LM using a set of everyday/general prompts and
then classifies each generation with the pre-trained TF-IDF + logistic
regression classifier to measure how "general" (non-legal) the model's output
style is.

The classifier is the same one used by the legal_style evaluator; here we
report the *general* class probability instead of the legal one.

Public interface:
    evaluate(model_name_or_path, device, **kwargs) -> dict
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch
from tqdm import tqdm

from evaluations.utils import load_model

# Re-use the classifier from the legal_style directory.
_CLF_PATH = Path(__file__).parent.parent / "legal_style" / "legal_style_clf.joblib"

# Default prompts: everyday, neutral, non-legal language.
_DEFAULT_PROMPTS = [
    "How do you make pancakes?",
    "The weather today is",
    "Scientists recently discovered that",
    "A simple recipe for pasta is",
    "The best way to learn a new language is",
    "According to recent research,",
    "The history of the internet began when",
    "In order to stay healthy,",
    "The main character of the story",
    "Technology has changed the way people",
]


# ---------------------------------------------------------------------------
# Classifier helper
# ---------------------------------------------------------------------------

def _load_classifier(clf_path: Path):
    if not clf_path.exists():
        raise FileNotFoundError(
            f"Legal/general-style classifier not found at {clf_path}.\n"
            "Train it first with:\n"
            "  python model_eval/doc_classification/legal_style_classifier_streaming.py train"
        )
    return joblib.load(clf_path)


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_texts(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int = 80,
) -> List[str]:
    texts = []
    for prompt in tqdm(prompts, desc="General-style: generating"):
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

    probs = clf.predict_proba(texts)
    general_probs = probs[:, 0].tolist()   # class 0 = general
    legal_probs   = probs[:, 1].tolist()   # class 1 = legal
    preds = clf.predict(texts).tolist()

    results = [
        {
            "prompt": p,
            "generated_text": t,
            "general_style_score": float(gp),
            "legal_style_score": float(lp),
            "predicted_label": "legal" if pred == 1 else "general",
        }
        for p, t, gp, lp, pred in zip(prompts, texts, general_probs, legal_probs, preds)
    ]

    return {
        "mean_general_style_score": float(np.mean(general_probs)),
        "median_general_style_score": float(np.median(general_probs)),
        "predicted_general_rate": float(1.0 - np.mean(preds)),
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
    Measure general-style score of a causal LM's generated text.

    The model generates continuations for everyday/neutral prompts; the same
    TF-IDF/logistic-regression classifier used by the legal_style evaluator
    scores each generation.  Here we report the *general* class probability.

    Args:
        model_name_or_path: HuggingFace model name or local adapter path.
        device: Torch device string.
        clf_path: Optional path to a .joblib classifier file.  Defaults to
                  the classifier in evaluations/legal_style/.
        prompts: List of prompt strings.  Defaults to built-in general prompts.
        max_new_tokens: Tokens to generate per prompt.
        max_examples: Cap on the number of prompts used.

    Returns:
        dict with scalar/printable metrics:
            mean_general_style_score  – average P(general) across generated texts
            median_general_style_score
            predicted_general_rate    – fraction classified as "general"
            num_examples
    """
    clf = _load_classifier(Path(clf_path) if clf_path else _CLF_PATH)
    model, tokenizer = load_model(model_name_or_path, device)

    active_prompts = list(prompts or _DEFAULT_PROMPTS)
    if max_examples is not None:
        active_prompts = active_prompts[:max_examples]

    output = _run_evaluation(model, tokenizer, device, clf, active_prompts, max_new_tokens)

    return {
        "mean_general_style_score": output["mean_general_style_score"],
        "median_general_style_score": output["median_general_style_score"],
        "predicted_general_rate": output["predicted_general_rate"],
        "num_examples": output["num_examples"],
        "_raw": output,
    }
