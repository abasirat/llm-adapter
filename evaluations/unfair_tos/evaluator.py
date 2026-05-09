"""
UNFAIR-ToS evaluator  (LexGLUE).

UNFAIR-ToS is a **multi-label** classification benchmark over Terms of Service
clauses.  Each clause may simultaneously belong to zero or more of 8 unfairness
categories:

    0  Limitation of liability
    1  Unilateral termination
    2  Unilateral change
    3  Content removal
    4  Contract by using
    5  Choice of law
    6  Jurisdiction
    7  Arbitration

Scoring method
--------------
Because GPT-2 is decoder-only we cannot attach a classification head.  Instead,
for each category we frame a binary prompt-based likelihood comparison:

    prompt  = "Terms of service clause: <text>\\nDoes this clause contain
               unfair terms related to <category_name>? Answer:"
    choice A = " Yes"
    choice B = " No"

The predicted label for category k is 1 if score(" Yes") > score(" No"),
else 0.

The KV cache of the shared prompt prefix (everything up to "Answer:") is
reused for the Yes/No comparison to avoid a redundant forward pass.

Metrics returned
----------------
  micro_f1          – micro-averaged F1 over all 8 binary decisions
  macro_f1          – macro-averaged F1 over the 8 categories
  per_label_f1      – dict {label_name: f1}  (stored in _raw only)
  num_examples      – number of examples evaluated

Public interface:
    evaluate(model_name_or_path, device, **kwargs) -> dict
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import f1_score
from tqdm import tqdm

from evaluations.utils import load_model


# ---------------------------------------------------------------------------
# Label verbalization
# ---------------------------------------------------------------------------

# Human-readable category names for the 8 unfairness classes (index-aligned).
_LABEL_NAMES = [
    "limitation of liability",
    "unilateral termination",
    "unilateral change",
    "content removal",
    "contract by using",
    "choice of law",
    "jurisdiction",
    "arbitration",
]

_YES_TEXT = " Yes"
_NO_TEXT  = " No"


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def _build_prompt(text: str, category: str) -> str:
    text = text.strip()[:1200]
    return (
        f"Terms of service clause: {text}\n"
        f"Does this clause contain unfair terms related to {category}? Answer:"
    )


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _binary_score(
    model,
    tokenizer,
    prompt: str,
    device: str,
) -> Tuple[float, float]:
    """
    Return (score_yes, score_no) as mean per-token log-probabilities.

    The prompt KV cache is computed once; Yes and No are each a single forward
    pass extending the cache.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    prompt_last_logit = out.logits[:, -1:, :]    # (1, 1, V)

    def _score_option(text: str) -> float:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if ids.shape[1] == 0:
            return float("-inf")
        option_out = model(ids, past_key_values=past, use_cache=False)
        all_logits = torch.cat([prompt_last_logit, option_out.logits[:, :-1, :]], dim=1)
        log_probs = F.log_softmax(all_logits, dim=-1)
        token_lp = log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        return token_lp.mean().item()

    return _score_option(_YES_TEXT), _score_option(_NO_TEXT)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _run_evaluation(
    model,
    tokenizer,
    device: str,
    split: str = "test",
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    dataset = load_dataset("lex_glue", "unfair_tos", split=split)

    # y_true / y_pred: List of n_examples binary vectors of length 8.
    y_true: List[List[int]] = []
    y_pred: List[List[int]] = []
    results: List[dict] = []

    for ex in tqdm(dataset, desc="UNFAIR-ToS", total=max_examples):
        if max_examples is not None and len(results) >= max_examples:
            break

        text: str = ex["text"]
        gold_labels: List[int] = ex["labels"]   # multi-hot list of active category indices

        gold_vec = [0] * len(_LABEL_NAMES)
        for idx in gold_labels:
            if 0 <= idx < len(_LABEL_NAMES):
                gold_vec[idx] = 1

        pred_vec = [0] * len(_LABEL_NAMES)
        scores_per_label: Dict[str, dict] = {}

        for k, category in enumerate(_LABEL_NAMES):
            prompt = _build_prompt(text, category)
            score_yes, score_no = _binary_score(model, tokenizer, prompt, device)
            pred_vec[k] = 1 if score_yes > score_no else 0
            scores_per_label[category] = {
                "score_yes": round(score_yes, 6),
                "score_no": round(score_no, 6),
                "pred": pred_vec[k],
                "gold": gold_vec[k],
            }

        y_true.append(gold_vec)
        y_pred.append(pred_vec)
        results.append(
            {
                "text": text[:200],
                "gold_labels": [_LABEL_NAMES[i] for i, v in enumerate(gold_vec) if v],
                "pred_labels": [_LABEL_NAMES[i] for i, v in enumerate(pred_vec) if v],
                "scores": scores_per_label,
            }
        )

    n = len(results)

    # sklearn expects shape (n_samples, n_classes)
    import numpy as np
    yt = np.array(y_true)
    yp = np.array(y_pred)

    micro_f1 = float(f1_score(yt, yp, average="micro", zero_division=0))
    macro_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    per_label_f1_arr = f1_score(yt, yp, average=None, zero_division=0)
    per_label_f1 = {
        _LABEL_NAMES[i]: round(float(per_label_f1_arr[i]), 4)
        for i in range(len(_LABEL_NAMES))
    }

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "num_examples": n,
        "per_label_f1": per_label_f1,
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
    Evaluate a causal LM on the UNFAIR-ToS benchmark (LexGLUE).

    Uses binary prompt-based likelihood scoring (Yes/No) independently for
    each of the 8 unfairness categories.  The prompt KV cache is shared
    across the two choices to halve the forward-pass cost per category.

    Args:
        model_name_or_path: HuggingFace model name or local adapter checkpoint.
        device: Torch device string.
        split: Dataset split — "train", "validation", or "test".
        max_examples: Optional cap on examples evaluated.

    Returns:
        dict with:
            micro_f1      – micro-averaged F1 over all 8 binary decisions
            macro_f1      – macro-averaged F1 over the 8 categories
            num_examples  – number of examples evaluated
    """
    model, tokenizer = load_model(model_name_or_path, device)
    output = _run_evaluation(model, tokenizer, device, split=split, max_examples=max_examples)

    return {
        "micro_f1": output["micro_f1"],
        "macro_f1": output["macro_f1"],
        "num_examples": output["num_examples"],
        "_raw": output,
    }
