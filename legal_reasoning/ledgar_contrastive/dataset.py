"""
LEDGAR Ranking Dataset for contrastive/ranking-based causal-LM training.

LEDGAR (LexGLUE) is a 100-class contract-provision classification task.
Each example is a contract provision text that must be assigned to one of
100 legal categories (e.g. "Indemnification", "Termination", "Governing Law").

Because there are 100 classes it is not feasible to encode all of them per
training example.  Instead, for each example we sample ``num_negatives``
label classes uniformly at random from the 99 incorrect classes.  The
correct label plus the sampled negatives form a ``(num_negatives + 1)``-way
ranking problem that can be optimised with the same ``RankingTrainer`` used
for CaseHOLD.

The correct candidate is inserted at a uniformly random position among the
negatives to avoid any positional bias; the ``label`` tensor records its index.

Sampling strategy
-----------------
Negatives are sampled once at dataset creation time (fixed per example).
Each epoch therefore sees the same negatives, which is sufficient for learning
because the model parameters change across epochs.  Pass ``--seed`` to control
reproducibility.

Prompt / label encoding  (matches ``evaluations/ledgar/evaluator.py``)
----------------------------------------------------------------------
prompt : ``"Contract provision: <text>\\nCategory:"``
         Text is clipped to 1500 characters before tokenisation (same as the
         evaluator) to avoid extreme sequence lengths.

label  : ``" " + label_name.replace("_", " ").replace("-", " ").strip()``
         The leading space ensures a clean BPE boundary between prompt and
         label, matching ``_score_labels_batched`` in the evaluator.

Shape of one example (before batch collation)
---------------------------------------------
    input_ids      : (num_negatives + 1, L)
    attention_mask : (num_negatives + 1, L)
    choice_mask    : (num_negatives + 1, L)  — 1 at label-token positions
    label          : scalar int              — index of the correct candidate
"""

import random
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Prompt / label helpers  (must match evaluations/ledgar/evaluator.py)
# ---------------------------------------------------------------------------

_PROMPT_TEXT_CLIP = 1500  # chars; mirrors the evaluator


def _build_prompt(text: str) -> str:
    return f"Contract provision: {text.strip()[:_PROMPT_TEXT_CLIP]}\nCategory:"


def _label_text(label_name: str) -> str:
    """Verbalise a LEDGAR label name in the same way as the evaluator."""
    return " " + label_name.replace("_", " ").replace("-", " ").strip()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LedgarRankingDataset(Dataset):
    """
    Torch Dataset wrapping a LexGLUE LEDGAR split for ranking/contrastive
    causal-LM training.

    Parameters
    ----------
    split          : ``"train"``, ``"validation"``, or ``"test"``
    tokenizer      : Any HuggingFace tokenizer.
    max_length     : Maximum total token count per (prompt + label) sequence.
    num_negatives  : Number of incorrect labels to sample per example.
                     Total candidates = num_negatives + 1 (the correct one).
    max_examples   : Optional cap on examples loaded from the split.
    seed           : Random seed for reproducible negative sampling.
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        num_negatives: int = 15,
        max_examples: Optional[int] = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives

        raw = load_dataset("lex_glue", "ledgar", split=split)
        # Fetch class names from dataset metadata (always 100 classes).
        self.label_names: List[str] = raw.features["label"].names

        if max_examples is not None:
            raw = raw.select(range(min(max_examples, len(raw))))

        # Pre-tokenize the label texts once — they are shared across all
        # examples and reused during negative sampling.
        self._label_ids: List[List[int]] = [
            tokenizer.encode(_label_text(name), add_special_tokens=False)
            for name in self.label_names
        ]

        # Store pre-tokenized prompts and gold labels so we can re-sample
        # negatives each epoch without re-downloading or re-tokenizing.
        self._prompt_ids_list: List[List[int]] = []
        self._gold_labels: List[int] = []
        for ex in raw:
            gold = int(ex["label"])
            prompt_ids = tokenizer.encode(
                _build_prompt(ex["text"]), add_special_tokens=True
            )
            self._prompt_ids_list.append(prompt_ids)
            self._gold_labels.append(gold)

        rng = random.Random(seed)
        self.examples: List[Dict[str, torch.Tensor]] = self._build_examples(rng, split)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample_negatives(self, seed: int) -> None:
        """Re-sample negatives with a new seed. Call at the start of each epoch."""
        rng = random.Random(seed)
        self.examples = self._build_examples(rng)

    def _build_examples(
        self,
        rng: random.Random,
        split: str = "",
    ) -> List[Dict[str, torch.Tensor]]:
        """Build ``self.examples`` from stored prompts and gold labels."""
        examples: List[Dict[str, torch.Tensor]] = []
        skipped = 0
        for prompt_ids, gold in zip(self._prompt_ids_list, self._gold_labels):
            encoded = self._encode_from_parts(prompt_ids, gold, rng)
            if encoded is not None:
                examples.append(encoded)
            else:
                skipped += 1
        if skipped:
            tag = f"/{split}" if split else ""
            print(
                f"[LedgarRankingDataset{tag}] Skipped {skipped} examples "
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
        Otherwise left-truncates the prompt so the candidate always fits.
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

    def _encode_from_parts(
        self,
        prompt_ids: List[int],
        gold: int,
        rng: random.Random,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Encode one LEDGAR example from pre-tokenized parts.

        The correct candidate is inserted at a random position among the
        negatives to prevent the model from exploiting positional bias.

        Returns ``None`` if the prompt alone exceeds ``max_length`` even
        without any label tokens appended.
        """
        # Sample negatives: all label indices except the correct one.
        negative_pool = [i for i in range(len(self.label_names)) if i != gold]
        neg_indices: List[int] = rng.sample(
            negative_pool, min(self.num_negatives, len(negative_pool))
        )

        correct_enc = self._encode_one(prompt_ids, self._label_ids[gold])
        if correct_enc is None:
            return None

        neg_encs: List[Dict[str, List[int]]] = []
        for ni in neg_indices:
            enc = self._encode_one(prompt_ids, self._label_ids[ni])
            if enc is None:
                # Extremely rare: label tokenises to >= max_length; use 1 token.
                enc = self._encode_one(prompt_ids, self._label_ids[ni][:1])
            neg_encs.append(enc)

        # Insert the correct candidate at a random position.
        insert_pos = rng.randint(0, len(neg_encs))
        all_encs: List[Dict[str, List[int]]] = (
            neg_encs[:insert_pos] + [correct_enc] + neg_encs[insert_pos:]
        )
        label_idx = insert_pos  # position of the correct candidate

        # Pad all candidates within this example to the longest sequence.
        max_len = max(len(e["input_ids"]) for e in all_encs)
        pad_id = self.tokenizer.pad_token_id

        input_ids_list: List[List[int]] = []
        attn_mask_list: List[List[int]] = []
        choice_mask_list: List[List[int]] = []

        for e in all_encs:
            pad_len = max_len - len(e["input_ids"])
            input_ids_list.append(e["input_ids"] + [pad_id] * pad_len)
            attn_mask_list.append(e["attention_mask"] + [0] * pad_len)
            choice_mask_list.append(e["choice_mask"] + [0] * pad_len)

        C = len(all_encs)  # num_negatives + 1
        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),       # (C, L)
            "attention_mask": torch.tensor(attn_mask_list, dtype=torch.long),  # (C, L)
            "choice_mask": torch.tensor(choice_mask_list, dtype=torch.long),   # (C, L)
            "label": torch.tensor(label_idx, dtype=torch.long),                # scalar
        }

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]
