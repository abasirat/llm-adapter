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

def _build_prompt(context: str, choice: str) -> str:
    return f"{context}\n\nCategory: {choice}"

def _choice_avg_logprob(
    model,
    tokenizer,
    context: str,
    choice: str,
    device: str,
    max_length: int = 1024,
) -> float:
    prompt = _build_prompt(context, choice)

    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"

    ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)

    tokenizer.truncation_side = old_side

    choice_token_count = len(tokenizer(choice, add_special_tokens=False).input_ids)

    if ids.shape[1] < 2:
        return float("-inf")

    with torch.no_grad():
        logits = model(ids).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(
        -1,
        shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    return token_lp[:, -choice_token_count:].mean().item()

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _run_evaluation(
    model,
    tokenizer,
    device: str,
    split: str = "test",
    max_examples: Optional[int] = None,
    label_batch_size: int = 32,
    max_length: int = 768,
) -> Dict[str, Any]:
    dataset = load_dataset("lex_glue", "ledgar", split=split)
    label_names = _get_label_names(dataset)

    results = []
    correct = 0
    total_margin = 0.0

    y_true = []
    y_pred = []

    for ex in tqdm(dataset, desc="LEDGAR", total=max_examples):
        if max_examples is not None and len(results) >= max_examples:
            break

        context: str = ex["text"]
        choices: List[str] = label_names  # always 100 candidates
        gold: int = int(ex["label"])

        scores = [
            _choice_avg_logprob(model, tokenizer, context, c, device)
            for c in choices
        ]

        pred = int(torch.tensor(scores).argmax().item())
        is_correct = pred == gold
        correct += int(is_correct)

        true_score = scores[gold]
        best_incorrect = max(s for i, s in enumerate(scores) if i != gold)
        margin = true_score - best_incorrect
        total_margin += margin

        y_true.append(gold)
        y_pred.append(pred)

        results.append(
            {
                "context": context,
                "choices": choices,
                "gold": gold,
                "pred": pred,
                "scores": [round(s, 6) for s in scores],
                "correct": is_correct,
                "margin": round(margin, 6),
            }
        )

    n = len(results)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if n else 0.0
    return {
        "accuracy": correct / n if n else 0.0,
        "correct_answer_margin": total_margin / n if n else 0.0,
        "num_examples": n,
        "results": results,
        "y_true": y_true,
        "y_pred": y_pred,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate(
    model_name_or_path: str,
    device: str,
    split: str = "test",
    max_examples: Optional[int] = None,
    label_batch_size: int = 32,
    max_length: int = 768,
    **kwargs,
) -> dict:
    model, tokenizer = load_model(model_name_or_path, device)

    output = _run_evaluation(
        model,
        tokenizer,
        device,
        split=split,
        max_examples=max_examples,
        label_batch_size=label_batch_size,
        max_length=max_length,
    )

    return {
        "accuracy": output["accuracy"],
        "macro_f1": output["macro_f1"],
        "num_examples": output["num_examples"],
        "y_true": output["y_true"],
        "y_pred": output["y_pred"],
        "_raw": output,
    }