"""
CaseHOLD Continuation Dataset for causal-LM training.

Each training instance is a single token sequence:

    [BOS] <legal context> \\nHolding: <correct holding>
    |_________prompt___________|_________target___________|

Labels
------
    -100        for every prompt token   → excluded from cross-entropy loss
    token id    for every target token   → loss is computed here

Truncation
----------
If (prompt + target) exceeds max_length, the prompt is LEFT-TRUNCATED so
the target tokens are always kept intact.  The tail of the context (near the
original <HOLDING> marker) is more informative than the document beginning,
so left-truncating the prompt is the correct choice.
"""

from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .utils import build_prompt


class CaseHoldContinuationDataset(Dataset):
    """
    Torch Dataset wrapping the LexGLUE CaseHOLD split.

    Parameters
    ----------
    split        : "train", "validation", or "test"
    tokenizer    : Any HuggingFace tokenizer
    max_length   : Maximum total token count (prompt + target).
    max_examples : Optional cap on examples loaded from the split.
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        max_examples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        raw = load_dataset("lex_glue", "case_hold", split=split)
        if max_examples is not None:
            raw = raw.select(range(min(max_examples, len(raw))))

        self.examples: List[Dict[str, torch.Tensor]] = []
        skipped = 0
        for ex in raw:
            encoded = self._encode(ex)
            if encoded is not None:
                self.examples.append(encoded)
            else:
                skipped += 1

        if skipped:
            print(
                f"[CaseHoldContinuationDataset/{split}] Skipped {skipped} examples "
                f"where the target alone exceeded max_length={max_length}."
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode(self, ex: dict) -> Optional[Dict[str, torch.Tensor]]:
        """
        Tokenize one CaseHOLD example.

        Returns None if the correct holding alone is longer than max_length
        (these examples cannot be trained on meaningfully).
        """
        label_idx = int(ex["label"])
        correct_holding = ex["endings"][label_idx].strip()

        prompt_text = build_prompt(ex["context"])

        # A leading space ensures a clean BPE boundary between prompt and
        # target, matching how the evaluator scores continuations with
        # " " + choice.strip().
        target_text = " " + correct_holding

        # Tokenize the prompt with special tokens (BOS for GPT-2).
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt_text, add_special_tokens=True
        )
        # Tokenize the target without special tokens (it is a continuation).
        target_ids: List[int] = self.tokenizer.encode(
            target_text, add_special_tokens=False
        )

        # If the target alone will not fit, skip this example.
        if len(target_ids) >= self.max_length:
            return None

        # Left-truncate the prompt so the full target always fits.
        # We keep the END of the prompt because the excerpt immediately before
        # the <HOLDING> marker is the most relevant context for the holding.
        max_prompt_len = self.max_length - len(target_ids)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + target_ids
        prompt_len = len(prompt_ids)

        # ---------------------------------------------------------------
        # Label masking
        # Prompt positions → -100  (CrossEntropyLoss ignores these)
        # Target positions → actual token ids  (loss is computed here only)
        # ---------------------------------------------------------------
        labels = ([-100] * prompt_len) + target_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]
