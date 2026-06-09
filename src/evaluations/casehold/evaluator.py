"""
CaseHOLD evaluator  (LexGLUE).

CaseHOLD is a multiple-choice legal benchmark: given a judicial holding context
and five candidate holdings, select the one that correctly completes the
citation in the excerpt.

Scoring method
--------------
For each candidate holding we compute the **average per-token log-probability**
of the holding text conditioned on the context prompt:

    score_i = mean_t [ log P(token_t | context, holding_tokens_<t) ]

Using the mean (not sum) removes length bias.  The predicted label is
argmax_i(score_i).  Only holding tokens are scored; prompt tokens are excluded.

Metrics returned
----------------
  accuracy                – fraction of examples predicted correctly
  correct_answer_margin   – mean of (logP(correct) - max logP(incorrect))
                            over all examples; positive → model prefers correct
  num_examples            – number of examples evaluated

Public interface:
    evaluate(model_name_or_path, device, **kwargs) -> dict
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from evaluations.utils import load_model


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _build_prompt(context: str, choice: str) -> str:
    prefix, _ = context.split("<HOLDING>")
    return prefix + choice

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
) -> Dict[str, Any]:
    dataset = load_dataset("lex_glue", "case_hold", split=split)

    results = []
    correct = 0
    total_margin = 0.0

    for ex in tqdm(dataset, desc="CaseHOLD", total=max_examples):
        if max_examples is not None and len(results) >= max_examples:
            break

        context: str = ex["context"]
        choices: List[str] = ex["endings"]   # always 5 candidates
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
    return {
        "accuracy": correct / n if n else 0.0,
        "correct_answer_margin": total_margin / n if n else 0.0,
        "num_examples": n,
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
    Evaluate a causal LM on the CaseHOLD benchmark (LexGLUE).

    Args:
        model_name_or_path: HuggingFace model name or local adapter checkpoint.
        device: Torch device string.
        split: Dataset split — "train", "validation", or "test".
        max_examples: Optional cap on examples evaluated.

    Returns:
        dict with:
            accuracy               – fraction of examples predicted correctly
            correct_answer_margin  – mean margin logP(correct) - max logP(wrong)
            num_examples           – number of examples evaluated
    """
    model, tokenizer = load_model(model_name_or_path, device)
    output = _run_evaluation(model, tokenizer, device, split=split, max_examples=max_examples)

    return {
        "accuracy": output["accuracy"],
        "correct_answer_margin": output["correct_answer_margin"],
        "num_examples": output["num_examples"],
        "_raw": output,
    }
