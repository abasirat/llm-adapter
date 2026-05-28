"""
TruthfulQA evaluator.

Scores a causal language model on the TruthfulQA multiple-choice benchmark
(mc1_targets) using log-probability ranking.

Public interface:
    evaluate(model_name_or_path, device, **kwargs) -> dict
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from evaluations.utils import load_model


# ---------------------------------------------------------------------------
# Core scoring helpers
# ---------------------------------------------------------------------------

def _answer_logprob(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    device: str,
    normalize: bool = True,
) -> float:
    """
    Compute log P(answer | prompt).

    Only answer tokens are scored; if normalize=True returns the mean per-token
    log-probability (less sensitive to answer length).
    """
    full_text = prompt + answer
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits

    # GPT-style: logit at position t predicts token t+1.
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Answer tokens start at position prompt_len (offset by -1 due to shift).
    answer_lp = token_log_probs[:, prompt_len - 1 :]

    if answer_lp.numel() == 0:
        return float("-inf")

    return answer_lp.mean().item() if normalize else answer_lp.sum().item()


def _format_prompt(question: str) -> str:
    return f"Q: {question}\nA:"


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

def _run_evaluation(
    model,
    tokenizer,
    device: str,
    split: str = "validation",
    normalize: bool = True,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Internal evaluation loop; returns the full results dict."""
    dataset = load_dataset("truthful_qa", "multiple_choice")[split]

    results = []
    correct = 0

    for i, ex in enumerate(tqdm(dataset, desc="TruthfulQA")):
        if max_examples is not None and i >= max_examples:
            break

        question = ex["question"]
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        prompt = _format_prompt(question)

        scores = [
            _answer_logprob(model, tokenizer, prompt, " " + c, device, normalize)
            for c in choices
        ]

        pred_idx = int(torch.tensor(scores).argmax().item())
        gold_idx = labels.index(1)
        is_correct = pred_idx == gold_idx
        correct += int(is_correct)

        true_score = scores[gold_idx]
        best_false = max(s for j, s in enumerate(scores) if j != gold_idx)

        results.append(
            {
                "question": question,
                "choices": choices,
                "labels": labels,
                "scores": scores,
                "pred_idx": pred_idx,
                "gold_idx": gold_idx,
                "correct": is_correct,
                "truthfulness_margin": true_score - best_false,
            }
        )

    accuracy = correct / len(results) if results else 0.0
    return {
        "accuracy": accuracy,
        "num_examples": len(results),
        "normalize": normalize,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate(
    model_name_or_path: str,
    device: str,
    split: str = "validation",
    normalize: bool = True,
    max_examples: Optional[int] = None,
    **kwargs,
) -> dict:
    """
    Evaluate a causal LM on TruthfulQA (multiple-choice, mc1).

    Args:
        model_name_or_path: HuggingFace model name or local adapter checkpoint path.
        device: Torch device string, e.g. "cuda", "mps", "cpu".
        split: Dataset split to use (default "validation").
        normalize: Whether to normalise log-probs by answer length.
        max_examples: Optional cap on number of examples evaluated.

    Returns:
        dict with scalar/printable metrics:
            accuracy      – fraction of questions answered correctly
            num_examples  – number of questions evaluated
    """
    model, tokenizer = load_model(model_name_or_path, device)
    output = _run_evaluation(
        model, tokenizer, device,
        split=split,
        normalize=normalize,
        max_examples=max_examples,
    )

    # Return only scalar summary metrics; full per-example results are in
    # the raw output but we expose them under a separate key so the central
    # runner can still store them.
    return {
        "accuracy": output["accuracy"],
        "num_examples": output["num_examples"],
        # Kept for optional JSON export by the central runner.
        "_raw": output,
    }
