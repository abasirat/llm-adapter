"""
UNFAIR-ToS Ranking Dataset for contrastive/ranking-based causal-LM training.

UNFAIR-ToS (LexGLUE) is a **multi-label** classification task over Terms of
Service clauses.  Each clause may belong to zero or more of 8 unfairness
categories:

    0  limitation of liability
    1  unilateral termination
    2  unilateral change
    3  content removal
    4  contract by using
    5  choice of law
    6  jurisdiction
    7  arbitration

Training approach
-----------------
Because the evaluator scores each category independently via a binary
``" Yes"`` / ``" No"`` prompt, training mirrors this by expanding each clause
into per-*(clause, category_k)* **binary ranking instances**:

  * Active category   → candidates = [" Yes", " No"],  label = 0  (Yes is correct)
  * Inactive category → candidates = [" Yes", " No"],  label = 1  (No is correct)

This yields a 2-way ranking problem per instance that the existing
``RankingTrainer`` / ``RankingCollator`` (from
``legal_reasoning.casehold_contrastive.trainer``) handles without modification.

Class imbalance
---------------
UNFAIR-ToS is heavily skewed toward inactive labels (~7 inactive per active
category on average).  The ``num_negative_categories`` parameter caps the
number of inactive-category instances sampled per clause.  All active
(positive) instances are always included.  Set ``num_negative_categories=None``
to include all inactive categories without subsampling.

Prompt / label encoding  (matches ``evaluations/unfair_tos/evaluator.py``)
--------------------------------------------------------------------------
prompt   : ``"Terms of service clause: {text[:1200]}\\nDoes this clause contain
             unfair terms related to {category}? Answer:"``
candidate: ``" Yes"``  or  ``" No"``   (space prefix; add_special_tokens=False)

Shape of one instance (before batch collation)
----------------------------------------------
    input_ids      : (2, L)   — L = max of Yes/No sequence lengths
    attention_mask : (2, L)
    choice_mask    : (2, L)   — 1 at candidate token positions, 0 elsewhere
    label          : scalar   — 0 (Yes) for active categories, 1 (No) otherwise

Negative resampling
-------------------
Call ``resample_negatives(seed)`` at the start of each epoch (handled by
``NegativeResamplerCallback``) to expose the model to different inactive
categories across epochs.
"""

import random
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Constants — must match evaluations/unfair_tos/evaluator.py exactly
# ---------------------------------------------------------------------------

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

_PROMPT_TEXT_CLIP = 1200  # chars; mirrors the evaluator

_YES_TEXT = " Yes"
_NO_TEXT  = " No"


# ---------------------------------------------------------------------------
# Prompt helper  (matches evaluations/unfair_tos/evaluator._build_prompt)
# ---------------------------------------------------------------------------

def _build_prompt(text: str, category: str) -> str:
    text = text.strip()[:_PROMPT_TEXT_CLIP]
    return (
        f"Terms of service clause: {text}\n"
        f"Does this clause contain unfair terms related to {category}? Answer:"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UnfairTosRankingDataset(Dataset):
    """
    Torch Dataset wrapping a LexGLUE UNFAIR-ToS split for ranking/contrastive
    causal-LM training.

    Each clause is expanded into per-*(clause, category)* binary ranking
    instances so that training directly mirrors the Yes/No prompt scoring used
    by ``evaluations/unfair_tos/evaluator.py``.

    Parameters
    ----------
    split                  : ``"train"``, ``"validation"``, or ``"test"``
    tokenizer              : Any HuggingFace tokenizer.
    max_length             : Maximum total token count per (prompt + candidate)
                             sequence.  Default 512.
    num_negative_categories: Number of inactive categories to sample per clause.
                             ``None`` = all inactive categories (no subsampling).
                             Default 4 reduces the ~7:1 inactive/active imbalance.
    max_examples           : Optional cap on *clauses* loaded from the split
                             (instances per clause will vary by label density).
    seed                   : Random seed for reproducible negative sampling.
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        num_negative_categories: Optional[int] = 4,
        max_examples: Optional[int] = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negative_categories = num_negative_categories

        raw = load_dataset("lex_glue", "unfair_tos", split=split)
        if max_examples is not None:
            raw = raw.select(range(min(max_examples, len(raw))))

        # Pre-tokenize the shared Yes/No candidates once.
        self._yes_ids: List[int] = tokenizer.encode(
            _YES_TEXT, add_special_tokens=False
        )
        self._no_ids: List[int] = tokenizer.encode(
            _NO_TEXT, add_special_tokens=False
        )

        # Store raw clause texts and gold label vectors for resampling.
        self._texts: List[str] = []
        self._gold_vecs: List[List[int]] = []  # multi-hot binary vectors

        for ex in raw:
            gold_labels: List[int] = ex["labels"]  # active category indices
            gold_vec = [0] * len(_LABEL_NAMES)
            for idx in gold_labels:
                if 0 <= idx < len(_LABEL_NAMES):
                    gold_vec[idx] = 1
            self._texts.append(ex["text"])
            self._gold_vecs.append(gold_vec)

        rng = random.Random(seed)
        self.examples: List[Dict[str, torch.Tensor]] = self._build_examples(
            rng, split
        )

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample_negatives(self, seed: int) -> None:
        """Re-sample negative-category instances with a new seed."""
        rng = random.Random(seed)
        self.examples = self._build_examples(rng)

    def _build_examples(
        self,
        rng: random.Random,
        split: str = "",
    ) -> List[Dict[str, torch.Tensor]]:
        """Build flat list of binary ranking instances from stored clause data."""
        examples: List[Dict[str, torch.Tensor]] = []
        skipped_clauses = 0

        for text, gold_vec in zip(self._texts, self._gold_vecs):
            active_indices = [k for k, v in enumerate(gold_vec) if v == 1]
            inactive_indices = [k for k, v in enumerate(gold_vec) if v == 0]

            # Sample inactive categories.
            if (
                self.num_negative_categories is not None
                and len(inactive_indices) > self.num_negative_categories
            ):
                sampled_inactive = rng.sample(
                    inactive_indices, self.num_negative_categories
                )
            else:
                sampled_inactive = inactive_indices

            # Build one instance per selected (clause, category) pair.
            clause_had_instance = False
            for k in active_indices:
                enc = self._encode_instance(text, _LABEL_NAMES[k], label=0)
                if enc is not None:
                    examples.append(enc)
                    clause_had_instance = True

            for k in sampled_inactive:
                enc = self._encode_instance(text, _LABEL_NAMES[k], label=1)
                if enc is not None:
                    examples.append(enc)
                    clause_had_instance = True

            if not clause_had_instance:
                skipped_clauses += 1

        if skipped_clauses:
            tag = f"/{split}" if split else ""
            print(
                f"[UnfairTosRankingDataset{tag}] Skipped {skipped_clauses} clauses "
                f"where the prompt alone exceeded max_length={self.max_length}."
            )
        return examples

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

        Returns ``None`` if the candidate alone is >= ``max_length``.
        Left-truncates the prompt so the candidate always fits.
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

        choice_mask = [0] * prompt_len + [1] * cand_len
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "choice_mask": choice_mask,
        }

    def _encode_instance(
        self,
        text: str,
        category: str,
        label: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Encode a single (clause, category) binary ranking instance.

        Tokenizes the prompt once and scores it against both " Yes" and " No".
        Returns ``None`` if the prompt alone cannot fit within ``max_length``
        (both candidates would be dropped).

        Parameters
        ----------
        text     : Clause text.
        category : Human-readable category name from ``_LABEL_NAMES``.
        label    : 0 if Yes is correct (active category), 1 if No is correct.
        """
        prompt_text = _build_prompt(text, category)
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt_text, add_special_tokens=True
        )

        yes_enc = self._encode_one(prompt_ids, self._yes_ids)
        no_enc  = self._encode_one(prompt_ids, self._no_ids)

        if yes_enc is None or no_enc is None:
            return None

        # Pad both sequences to the longer of the two.
        max_len = max(len(yes_enc["input_ids"]), len(no_enc["input_ids"]))
        pad_id = self.tokenizer.pad_token_id

        def _pad(enc: Dict[str, List[int]]) -> Dict[str, List[int]]:
            pad_len = max_len - len(enc["input_ids"])
            return {
                "input_ids":      enc["input_ids"]      + [pad_id] * pad_len,
                "attention_mask": enc["attention_mask"]  + [0]      * pad_len,
                "choice_mask":    enc["choice_mask"]     + [0]      * pad_len,
            }

        yes_padded = _pad(yes_enc)
        no_padded  = _pad(no_enc)

        return {
            "input_ids":      torch.tensor(
                [yes_padded["input_ids"],      no_padded["input_ids"]],
                dtype=torch.long,
            ),  # (2, L)
            "attention_mask": torch.tensor(
                [yes_padded["attention_mask"], no_padded["attention_mask"]],
                dtype=torch.long,
            ),  # (2, L)
            "choice_mask":    torch.tensor(
                [yes_padded["choice_mask"],    no_padded["choice_mask"]],
                dtype=torch.long,
            ),  # (2, L)
            "label": torch.tensor(label, dtype=torch.long),  # scalar: 0 or 1
        }

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]
