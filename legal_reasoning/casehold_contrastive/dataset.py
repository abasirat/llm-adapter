"""
CaseHOLD Ranking Dataset for contrastive/ranking-based causal-LM training.

For each CaseHOLD example (context + 5 candidate holdings), all five candidate
sequences are encoded:

    [BOS] <legal context> \\nHolding: <candidate_i holding>

No label-masking is applied here.  Instead a ``choice_mask`` tensor marks which
token positions belong to the candidate holding; this mask is used inside
``RankingTrainer.compute_loss`` to extract per-candidate mean log-prob scores.

Shape of one example (before batch collation)
----------------------------------------------
    input_ids      : (5, L)   — L = max sequence length across the 5 candidates
    attention_mask : (5, L)
    choice_mask    : (5, L)   — 1 at candidate token positions, 0 elsewhere
    label          : scalar   — 0–4, index of the correct holding

Sequences are independently left-truncated per candidate so the candidate
tokens always fit within ``max_length``, matching the convention of
``CaseHoldContinuationDataset``.  Within each example the 5 sequences are
right-padded to the same length (``max(L_i)``).  Across-batch padding is
handled by ``RankingCollator`` in ``trainer.py``.
"""

from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from legal_reasoning.casehold_continuation.utils import build_prompt


class CaseHoldRankingDataset(Dataset):
    """
    Torch Dataset wrapping a LexGLUE CaseHOLD split for ranking/contrastive
    causal-LM training.

    Parameters
    ----------
    split        : ``"train"``, ``"validation"``, or ``"test"``
    tokenizer    : Any HuggingFace tokenizer.
    max_length   : Maximum total token count per (prompt + candidate) sequence.
                   Default 512 is lower than the continuation dataset's 1024
                   because each example holds 5 sequences simultaneously.
    max_examples : Optional cap on examples loaded from the split.
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
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
                f"[CaseHoldRankingDataset/{split}] Skipped {skipped} examples "
                f"where at least one candidate alone exceeded max_length={max_length}."
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_one(
        self,
        prompt_ids: List[int],
        candidate_ids: List[int],
    ) -> Optional[Dict[str, List[int]]]:
        """
        Encode a single (prompt, candidate) pair.

        Returns ``None`` if the candidate alone is >= ``max_length`` (cannot
        fit even with an empty prompt).

        Otherwise left-truncates the prompt so that the candidate always
        occupies its full length, and returns plain Python lists for
        ``input_ids``, ``attention_mask``, and ``choice_mask``.
        """
        if len(candidate_ids) >= self.max_length:
            return None

        max_prompt_len = self.max_length - len(candidate_ids)
        trunc_prompt = (
            prompt_ids[-max_prompt_len:]
            if len(prompt_ids) > max_prompt_len
            else prompt_ids
        )

        input_ids = trunc_prompt + candidate_ids
        prompt_len = len(trunc_prompt)
        cand_len = len(candidate_ids)

        # choice_mask: 0 at prompt positions, 1 at candidate positions.
        choice_mask = [0] * prompt_len + [1] * cand_len
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "choice_mask": choice_mask,
        }

    def _encode(self, ex: dict) -> Optional[Dict[str, torch.Tensor]]:
        """
        Encode all 5 candidates for one CaseHOLD example.

        Returns ``None`` if *any* candidate alone exceeds ``max_length``
        (we cannot rank fairly if a candidate cannot be scored).
        """
        prompt_text = build_prompt(ex["context"])

        # Tokenize with BOS (add_special_tokens=True for prompt only).
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt_text, add_special_tokens=True
        )

        encoded_candidates: List[Dict[str, List[int]]] = []
        for holding in ex["endings"]:
            # Prepend a space for a clean BPE boundary, matching the convention
            # in _choice_avg_logprob (evaluations/casehold/evaluator.py).
            candidate_ids: List[int] = self.tokenizer.encode(
                " " + holding.strip(), add_special_tokens=False
            )
            enc = self._encode_one(prompt_ids, candidate_ids)
            if enc is None:
                return None  # skip example entirely if any candidate is too long
            encoded_candidates.append(enc)

        # Pad all 5 sequences within this example to the longest one.
        max_len = max(len(e["input_ids"]) for e in encoded_candidates)
        pad_id = self.tokenizer.pad_token_id

        input_ids_list: List[List[int]] = []
        attn_mask_list: List[List[int]] = []
        choice_mask_list: List[List[int]] = []

        for e in encoded_candidates:
            pad_len = max_len - len(e["input_ids"])
            input_ids_list.append(e["input_ids"] + [pad_id] * pad_len)
            attn_mask_list.append(e["attention_mask"] + [0] * pad_len)
            choice_mask_list.append(e["choice_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),       # (5, L)
            "attention_mask": torch.tensor(attn_mask_list, dtype=torch.long),  # (5, L)
            "choice_mask": torch.tensor(choice_mask_list, dtype=torch.long),   # (5, L)
            "label": torch.tensor(int(ex["label"]), dtype=torch.long),         # scalar
        }

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]
