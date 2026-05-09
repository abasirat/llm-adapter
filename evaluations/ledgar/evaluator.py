"""
LEDGAR evaluator  (LexGLUE).

LEDGAR is a contract-provision classification benchmark with 100 classes.
Each example is a provision text that must be assigned to its legal category
(e.g. "Indemnification", "Termination", "Governing Law", …).

Because GPT-2 is decoder-only, we use prompt-based likelihood scoring rather
than a classification head.

Scoring method
--------------
For each label L we construct:

    prompt  = "Contract provision: <text>\\nCategory:"
    scoring = " <label_string>"

and compute mean per-token log P(label_string | prompt).  The predicted class
is argmax over all 100 labels.

To amortise computation the prompt forward pass is cached via
past_key_values: we run the model once for the prompt and then score every
label with a single additional forward pass using the cached KV state.

Metrics returned
----------------
  accuracy        – exact-match accuracy
  macro_f1        – unweighted macro-averaged F1 across all classes
  per_class_f1    – dict {label_name: f1}  (stored in _raw only)
  num_examples    – number of examples evaluated

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

def _get_label_names(dataset) -> List[str]:
    """Return the ordered list of class label strings from the dataset features."""
    return dataset.features["label"].names


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def _build_prompt(text: str) -> str:
    # Truncate very long provisions to keep prompt within model context.
    text = text.strip()[:1500]
    return f"Contract provision: {text}\nCategory:"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_labels_with_cache(
    model,
    tokenizer,
    prompt: str,
    label_names: List[str],
    device: str,
) -> List[float]:
    """
    Score every label for a single prompt, reusing the prompt's KV cache.

    Returns a list of mean per-token log-probabilities, one per label.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Single forward pass for the prompt — capture the KV cache.
    out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    prompt_last_logit = out.logits[:, -1:, :]   # (1, 1, V)  — logit for token after prompt

    scores: List[float] = []

    for label in label_names:
        label_text = " " + label.strip()
        label_ids = tokenizer(
            label_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)                                     # (1, n)

        if label_ids.shape[1] == 0:
            scores.append(float("-inf"))
            continue

        # Concatenate prompt's last logit with labels' logits (using cache).
        # We need logits for all label token positions:
        # - position 0 of label: predicted by prompt_last_logit
        # - positions 1..n-1: predicted by the model with cache extended by label[:-1]

        # Run label tokens through model with KV cache.
        label_out = model(label_ids, past_key_values=past, use_cache=False)
        # label_out.logits shape: (1, n, V) — logit[i] predicts label_id[i+1]
        # logit[-1] would predict the next token after the label, which we don't need.

        # Full logit sequence for scoring label tokens:
        #   prompt_last_logit  → predicts label_ids[:, 0]
        #   label_out.logits[:, :-1, :] → predicts label_ids[:, 1:]
        all_logits = torch.cat(
            [prompt_last_logit, label_out.logits[:, :-1, :]], dim=1
        )                                                           # (1, n, V)

        log_probs = F.log_softmax(all_logits, dim=-1)
        token_lp = log_probs.gather(
            -1, label_ids.unsqueeze(-1)
        ).squeeze(-1)                                               # (1, n) — wait, label_ids is (1,n)

        # gather needs indices shaped (1, n, 1)
        token_lp = log_probs.gather(
            -1, label_ids.unsqueeze(-1)
        ).squeeze(-1)                                               # (1, n)

        scores.append(token_lp.mean().item())

    return scores


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
    dataset = load_dataset("lex_glue", "ledgar", split=split)
    label_names = _get_label_names(dataset)
    n_labels = len(label_names)

    results = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for ex in tqdm(dataset, desc="LEDGAR", total=max_examples):
        if max_examples is not None and len(results) >= max_examples:
            break

        text: str = ex["text"]
        gold: int = int(ex["label"])

        prompt = _build_prompt(text)
        scores = _score_labels_with_cache(model, tokenizer, prompt, label_names, device)

        pred = int(torch.tensor(scores).argmax().item())
        is_correct = pred == gold

        y_true.append(gold)
        y_pred.append(pred)

        results.append(
            {
                "text": text[:200],
                "gold": gold,
                "gold_label": label_names[gold],
                "pred": pred,
                "pred_label": label_names[pred],
                "correct": is_correct,
            }
        )

    n = len(results)
    accuracy = sum(r["correct"] for r in results) / n if n else 0.0

    labels_present = sorted(set(y_true))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=labels_present, average="macro", zero_division=0)
    )
    per_class_f1_arr = f1_score(
        y_true, y_pred, labels=list(range(n_labels)), average=None, zero_division=0
    )
    per_class_f1 = {
        label_names[i]: round(float(per_class_f1_arr[i]), 4)
        for i in range(n_labels)
    }

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_examples": n,
        "per_class_f1": per_class_f1,
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
    Evaluate a causal LM on the LEDGAR benchmark (LexGLUE).

    Uses prompt-based likelihood scoring with the full label set (100 classes).
    The KV cache of the prompt is reused across all label candidates to avoid
    redundant computation.

    Args:
        model_name_or_path: HuggingFace model name or local adapter checkpoint.
        device: Torch device string.
        split: Dataset split — "train", "validation", or "test".
        max_examples: Optional cap on examples evaluated.

    Returns:
        dict with:
            accuracy      – exact-match accuracy
            macro_f1      – macro-averaged F1 across all classes
            num_examples  – number of examples evaluated
    """
    model, tokenizer = load_model(model_name_or_path, device)
    output = _run_evaluation(model, tokenizer, device, split=split, max_examples=max_examples)

    return {
        "accuracy": output["accuracy"],
        "macro_f1": output["macro_f1"],
        "num_examples": output["num_examples"],
        "_raw": output,
    }
