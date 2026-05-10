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

def _choice_avg_logprob(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    choice: str,
    device: str,
) -> float:
    """
    Compute mean per-token log P(choice | prompt).

    prompt_ids : (1, prompt_len) already on *device*.
    choice     : raw string for the holding (a space is prepended to separate
                 it cleanly from the prompt).
    Returns    : scalar float (higher = more likely).
    """
    choice_ids = tokenizer(
        " " + choice.strip(),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    if choice_ids.shape[1] == 0:
        return float("-inf")

    full_ids = torch.cat([prompt_ids, choice_ids], dim=1)  # (1, L)

    with torch.no_grad():
        logits = model(full_ids).logits                     # (1, L, V)

    # Shift: logit[t] predicts token[t+1].
    shift_logits = logits[:, :-1, :]                        # (1, L-1, V)
    shift_labels = full_ids[:, 1:]                          # (1, L-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, L-1)

    # Only average over the choice tokens (prompt_len .. L-1 after shift).
    prompt_len = prompt_ids.shape[1]
    choice_lp = token_lp[:, prompt_len - 1:]               # (1, n_choice_tokens)

    if choice_lp.numel() == 0:
        return float("-inf")

    return choice_lp.mean().item()


#def _build_prompt(context: str) -> str:
#    """
#    Wrap the citation context in a clear instruction prefix.
#
#    CaseHOLD context strings already end with <HOLDING>.  We replace that
#    marker with a colon to form the prompt.
#    """
#    context = context.strip()
#    if context.endswith("<HOLDING>"):
#        context = context[: -len("<HOLDING>")].rstrip() + "\nHolding:"
#    else:
#        context = context + "\nHolding:"
#    return context
def _build_prompt(context: str) -> str:
    # Delegate to the canonical implementation used during continuation training
    # so that inference-time and training-time prompts are identical.
    from legal_reasoning.casehold_continuation.utils import build_prompt
    return build_prompt(context)


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

        prompt = _build_prompt(context)

        old_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left" # truncate from the left if needed to fit max_length. We want to keep the end of the prompt since the holding is there.
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=900
        ).input_ids.to(device)
        tokenizer.truncation_side = old_side

        scores = [
            _choice_avg_logprob(model, tokenizer, prompt_ids, c, device)
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
