"""
Collator and Trainer for CaseHOLD ranking/contrastive causal-LM training.

RankingCollator
---------------
Pads a batch of five-candidate examples produced by ``CaseHoldRankingDataset``
to a common length, yielding ``(B, 5, L_max)`` tensors.

RankingTrainer
--------------
HF Trainer subclass that implements differentiable ranking loss.

Scoring
~~~~~~~
For each candidate *i* the score *s_i* is the mean per-token log-probability
of the candidate tokens conditioned on the context — identical to the formula
used by ``_choice_avg_logprob`` in ``evaluations/casehold/evaluator.py``, but
computed with gradients so the model can be optimised:

    s_i = mean_t [ log P(token_t | context, candidate_tokens_<t) ]

All five candidates are scored in a **single forward pass** by flattening the
``(B, 5, L)`` batch to ``(B*5, L)``.

Loss variants
~~~~~~~~~~~~~
softmax  (default):  F.cross_entropy(scores, labels)
                     = − s_correct + log Σ exp(s_i)
                     (InfoNCE / noise-contrastive estimation)

hinge             :  (1/4) Σ_{i≠c}  max(0, γ − s_c + s_i)
                     Classic SVM-style pairwise max-margin.

logistic          :  (1/4) Σ_{i≠c}  log(1 + exp(s_i − s_c))
                     Smooth surrogate for the hinge, no margin hyperparameter.

Optional auxiliary CE
~~~~~~~~~~~~~~~~~~~~~
When ``aux_ce_alpha > 0`` a cross-entropy loss on the correct holding tokens
is added, sharing the **same forward pass** (no extra compute):

    total_loss = ranking_loss + aux_ce_alpha × CE(correct_holding | context)

This regularises the model's language-modelling ability and can prevent
catastrophic forgetting.
"""

from typing import Dict, List, Tuple

import logging

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV-cache expansion helper
# ---------------------------------------------------------------------------

def _expand_past_kv(past_kv, B: int, C: int):
    """
    Expand a KV cache from batch size B to B*C by repeating each entry C times.

    Handles three generations of the transformers KV-cache API:
      - Plain tuple-of-tuples (very old transformers): expanded tuple returned.
      - DynamicCache with key_cache/value_cache lists (transformers ~4.40+, 5.x):
        lists are expanded in-place on a new DynamicCache and returned.
      - Legacy DynamicCache with only to_legacy_cache/from_legacy_cache (4.36–4.39):
        round-tripped through the legacy tuple format.
    """
    def _expand(t: torch.Tensor) -> torch.Tensor:
        # (B, nh, P, hd)  →  (B*C, nh, P, hd)
        return t.unsqueeze(1).expand(-1, C, -1, -1, -1).reshape(B * C, *t.shape[1:])

    # --- Path 1: new-style DynamicCache with key_cache / value_cache lists -------
    # This covers transformers 5.x (and 4.40+) which requires a Cache object back.
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        try:
            from transformers.cache_utils import DynamicCache
            new_cache = DynamicCache()
            new_cache.key_cache   = [_expand(k) for k in past_kv.key_cache]
            new_cache.value_cache = [_expand(v) for v in past_kv.value_cache]
            # Keep _seen_tokens consistent so get_seq_length() returns the right value.
            if hasattr(new_cache, "_seen_tokens") and new_cache.key_cache:
                new_cache._seen_tokens = new_cache.key_cache[0].shape[-2]
            return new_cache
        except Exception:
            pass  # fall through to legacy path

    # --- Path 2: older DynamicCache with to_legacy_cache / from_legacy_cache -----
    if hasattr(past_kv, "to_legacy_cache"):
        legacy = past_kv.to_legacy_cache()
        expanded_legacy = tuple(
            (_expand(layer[0]), _expand(layer[1])) + tuple(layer[2:])
            for layer in legacy
        )
        try:
            from transformers.cache_utils import DynamicCache
            return DynamicCache.from_legacy_cache(expanded_legacy)
        except Exception:
            return expanded_legacy

    # --- Path 3: plain tuple-of-tuples (very old transformers) -------------------
    return tuple(
        (_expand(layer[0]), _expand(layer[1])) + tuple(layer[2:])
        for layer in past_kv
    )


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class RankingCollator:
    """
    Collate a batch of ``CaseHoldRankingDataset`` examples into padded tensors.

    Each example has shape ``(5, L_i)`` where ``L_i`` may differ across
    examples.  The collator right-pads every sequence to
    ``L_max = max(L_i)`` across the batch.

    Output tensors
    --------------
    input_ids      : right-padded with ``pad_token_id``  → ``(B, 5, L_max)``
    attention_mask : right-padded with ``0``              → ``(B, 5, L_max)``
    choice_mask    : right-padded with ``0``              → ``(B, 5, L_max)``
    labels         : stacked label indices                → ``(B,)``
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        max_len = max(e["input_ids"].shape[1] for e in examples)

        input_ids_out: List[torch.Tensor] = []
        attn_out: List[torch.Tensor] = []
        mask_out: List[torch.Tensor] = []
        labels_out: List[torch.Tensor] = []

        for e in examples:
            n_cands = e["input_ids"].shape[0]  # number of candidates (5 for CaseHOLD, variable for LEDGAR)
            seq_len = e["input_ids"].shape[1]
            pad_len = max_len - seq_len

            if pad_len > 0:
                pad_ids = torch.full(
                    (n_cands, pad_len), self.pad_token_id, dtype=torch.long
                )
                pad_zeros = torch.zeros((n_cands, pad_len), dtype=torch.long)
                input_ids_out.append(torch.cat([e["input_ids"], pad_ids], dim=1))
                attn_out.append(torch.cat([e["attention_mask"], pad_zeros], dim=1))
                mask_out.append(torch.cat([e["choice_mask"], pad_zeros], dim=1))
            else:
                input_ids_out.append(e["input_ids"])
                attn_out.append(e["attention_mask"])
                mask_out.append(e["choice_mask"])

            labels_out.append(e["label"])

        return {
            "input_ids": torch.stack(input_ids_out),       # (B, 5, L_max)
            "attention_mask": torch.stack(attn_out),        # (B, 5, L_max)
            "choice_mask": torch.stack(mask_out),           # (B, 5, L_max)
            "labels": torch.stack(labels_out),              # (B,)
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RankingTrainer(Trainer):
    """
    HF Trainer subclass that optimises a ranking/contrastive objective over
    the 5 CaseHOLD candidates using causal-LM likelihood scores.

    Parameters
    ----------
    ranking_loss  : ``"softmax"`` | ``"hinge"`` | ``"logistic"``
                    (default: ``"softmax"``)
    margin        : Hinge margin *γ* (default: ``1.0``; only used when
                    ``ranking_loss="hinge"``).
    aux_ce_alpha  : Weight of auxiliary CE loss on the correct holding
                    (default: ``0.0`` — pure ranking loss).

    All other positional/keyword arguments are forwarded to
    ``transformers.Trainer``.
    """

    def __init__(
        self,
        *args,
        ranking_loss: str = "softmax",
        margin: float = 1.0,
        aux_ce_alpha: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if ranking_loss not in ("softmax", "hinge", "logistic"):
            raise ValueError(
                f"ranking_loss must be 'softmax', 'hinge', or 'logistic', "
                f"got {ranking_loss!r}"
            )
        self.ranking_loss_type = ranking_loss
        self.margin = margin
        self.aux_ce_alpha = aux_ce_alpha

    # ------------------------------------------------------------------
    # Shared forward pass + scoring (used by both compute_loss and
    # prediction_step to avoid redundant computation)
    # ------------------------------------------------------------------

    def _score_from_logits(
        self,
        logits: torch.Tensor,        # (B*C, L, V)
        flat_ids: torch.Tensor,      # (B*C, L)
        flat_choice_mask: torch.Tensor,  # (B*C, L)  float
        B: int,
        C: int,
        L: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared scoring logic: returns (scores, shift_logits, shift_ids, shifted_mask)."""
        shift_logits = logits[:, :-1, :]   # (B*C, L-1, V)
        shift_ids    = flat_ids[:, 1:]     # (B*C, L-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp  = log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)  # (B*C, L-1)

        shifted_mask     = flat_choice_mask[:, 1:]                    # (B*C, L-1)
        masked_lp        = token_lp * shifted_mask
        candidate_counts = shifted_mask.sum(-1).clamp(min=1)          # (B*C,)
        scores           = (masked_lp.sum(-1) / candidate_counts).view(B, C)

        return scores, shift_logits, shift_ids, shifted_mask

    def _forward_and_score(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,   # scores         (B, C)
        object,         # model outputs (label pass)
        torch.Tensor,   # shift_logits   (B*C, L-1, V)
        torch.Tensor,   # shift_ids      (B*C, L-1)
        torch.Tensor,   # shifted_mask   (B*C, L-1)  float
        int, int, int,  # B, C, L
    ]:
        """
        Score all C candidates via causal-LM log-probabilities.

        Fast path: the prompt prefix shared by all C candidates is encoded
        ONCE with a KV cache, then all C label tails are scored in a single
        batched forward pass over B*C sequences of length (L - P) instead of
        B*C sequences of length L.  For typical LEDGAR sequences this reduces
        attention compute by ~8× and FFN compute by ~6×.

        Falls back to a flat B*C forward pass when the shared prefix cannot
        be determined (degenerate inputs).
        """
        B, C, L = inputs["input_ids"].shape

        # ------------------------------------------------------------------
        # Determine the shared prompt prefix length.
        # choice_mask[b,c,t] == 1 iff token t is a label (non-prompt) token.
        # All C candidates in one example share the same prompt; we find the
        # minimum first-label-token position across candidates and examples.
        # ------------------------------------------------------------------
        cm         = inputs["choice_mask"]                      # (B, C, L)
        has_label  = (cm > 0).any(dim=-1)                       # (B, C)
        first_lp   = (cm > 0).float().argmax(dim=-1)            # (B, C)
        first_lp[~has_label] = L                                # safety: no label token
        P = int(first_lp.min().item())                          # shared prompt length

        if P <= 0 or P >= L:
            # Degenerate — fall back to flat pass.
            return self._forward_and_score_flat(model, inputs, B, C, L)

        # ------------------------------------------------------------------
        # Step 1: encode the shared prompt prefix once per batch item.
        # We use candidate 0's prefix; all candidates share the same prompt
        # up to position P.
        # ------------------------------------------------------------------
        prompt_ids  = inputs["input_ids"][:, 0, :P]       # (B, P)
        prompt_attn = inputs["attention_mask"][:, 0, :P]  # (B, P)

        prompt_out    = model(prompt_ids, attention_mask=prompt_attn, use_cache=True)
        prompt_logits = prompt_out.logits                  # (B, P, V)
        past_kv       = prompt_out.past_key_values

        # Expand KV cache: (B, nh, P, hd) → (B*C, nh, P, hd).
        # Handles both DynamicCache (transformers >= 4.36) and legacy tuple-of-tuples.
        past_kv_expanded = _expand_past_kv(past_kv, B, C)

        # ------------------------------------------------------------------
        # Step 2: score all C label tails in one batched forward pass.
        # ------------------------------------------------------------------
        flat_label_ids = inputs["input_ids"][:, :, P:].reshape(B * C, L - P)   # (B*C, L-P)
        flat_full_attn = inputs["attention_mask"].reshape(B * C, L)             # (B*C, L)

        label_out    = model(flat_label_ids, attention_mask=flat_full_attn,
                             past_key_values=past_kv_expanded)
        label_logits = label_out.logits    # (B*C, L-P, V)

        # ------------------------------------------------------------------
        # Step 3: reconstruct full (B*C, L, V) logits and compute scores.
        # Prompt logits are identical for all C candidates — broadcast them.
        # ------------------------------------------------------------------
        V = prompt_logits.shape[-1]
        prompt_logits_flat = (
            prompt_logits.unsqueeze(1)
            .expand(-1, C, -1, -1)
            .reshape(B * C, P, V)
        )
        logits = torch.cat([prompt_logits_flat, label_logits], dim=1)  # (B*C, L, V)

        flat_ids  = inputs["input_ids"].view(B * C, L)
        flat_mask = inputs["choice_mask"].view(B * C, L).float()

        scores, shift_logits, shift_ids, shifted_mask = self._score_from_logits(
            logits, flat_ids, flat_mask, B, C, L
        )
        return scores, label_out, shift_logits, shift_ids, shifted_mask, B, C, L

    def _forward_and_score_flat(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        B: int,
        C: int,
        L: int,
    ) -> Tuple[
        torch.Tensor, object, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int
    ]:
        """Fallback: flat B*C forward pass (no prefix-cache optimisation)."""
        flat_ids  = inputs["input_ids"].view(B * C, L)
        flat_attn = inputs["attention_mask"].view(B * C, L)

        outputs = model(flat_ids, attention_mask=flat_attn)
        logits  = outputs.logits  # (B*C, L, V)

        flat_mask = inputs["choice_mask"].view(B * C, L).float()
        scores, shift_logits, shift_ids, shifted_mask = self._score_from_logits(
            logits, flat_ids, flat_mask, B, C, L
        )
        return scores, outputs, shift_logits, shift_ids, shifted_mask, B, C, L

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _ranking_loss(
        self,
        scores: torch.Tensor,  # (B, 5)
        labels: torch.Tensor,  # (B,)
        B: int,
        C: int,
    ) -> torch.Tensor:
        """Compute the configured ranking loss from ``(B, C)`` scores."""
        if self.ranking_loss_type == "softmax":
            # InfoNCE: −s_correct + log Σ exp(s_i)
            return F.cross_entropy(scores, labels)

        # Pairwise losses require separating correct vs. incorrect scores.
        batch_idx = torch.arange(B, device=scores.device)
        correct_scores = scores[batch_idx, labels].unsqueeze(1).expand(B, C)  # (B, 5)

        # Boolean mask: True at the 4 incorrect candidate positions.
        neg_mask = torch.ones(B, C, dtype=torch.bool, device=scores.device)
        neg_mask[batch_idx, labels] = False

        if self.ranking_loss_type == "hinge":
            # max(0, γ − s_correct + s_incorrect)
            margins = F.relu(self.margin - correct_scores + scores)  # (B, 5)
            return margins[neg_mask].mean()

        # logistic: log(1 + exp(s_incorrect − s_correct))
        #         = softplus(s_incorrect − s_correct)
        diffs = scores - correct_scores  # (B, 5); negative for correct candidate
        return F.softplus(diffs)[neg_mask].mean()

    def _aux_ce_loss(
        self,
        shift_logits: torch.Tensor,   # (B*5, L-1, V)
        shift_ids: torch.Tensor,       # (B*5, L-1)
        shifted_mask: torch.Tensor,    # (B*5, L-1)  float
        labels: torch.Tensor,          # (B,)
        B: int,
        C: int,
        L: int,
    ) -> torch.Tensor:
        """
        Cross-entropy loss on the correct holding tokens.

        Reuses the ``shift_logits`` and ``shift_ids`` tensors already computed
        in ``_forward_and_score`` — no extra forward pass.
        """
        V = shift_logits.shape[-1]
        Lm1 = L - 1
        device = labels.device

        idx = torch.arange(B, device=device)

        # Extract the correct candidate's shifted logits/ids/mask.
        correct_shift_logits = shift_logits.view(B, C, Lm1, V)[idx, labels]   # (B, Lm1, V)
        correct_shift_ids = shift_ids.view(B, C, Lm1)[idx, labels]             # (B, Lm1)
        correct_shifted_mask = shifted_mask.view(B, C, Lm1)[idx, labels]       # (B, Lm1) float

        # CE targets: token ids at candidate positions, -100 (ignored) elsewhere.
        ce_labels = correct_shift_ids.clone()
        ce_labels[correct_shifted_mask == 0] = -100

        return F.cross_entropy(
            correct_shift_logits.reshape(-1, V),
            ce_labels.reshape(-1),
            ignore_index=-100,
        )

    # ------------------------------------------------------------------
    # Training: compute_loss
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the ranking (± auxiliary CE) loss.

        A single forward pass scores all 5 candidates; the ranking loss
        and optional CE loss are derived from those scores.

        Adapter models reject unknown kwargs (e.g. ``num_items_in_batch``
        injected by Trainer ≥4.46 for loss scaling).  We pop it here,
        mirroring the same workaround in ``CausalLMTrainer``.
        """
        kwargs.pop("num_items_in_batch", None)

        scores, outputs, shift_logits, shift_ids, shifted_mask, B, C, L = (
            self._forward_and_score(model, inputs)
        )
        labels = inputs["labels"]

        loss = self._ranking_loss(scores, labels, B, C)

        if self.aux_ce_alpha > 0.0:
            aux = self._aux_ce_loss(
                shift_logits, shift_ids, shifted_mask, labels, B, C, L
            )
            loss = loss + self.aux_ce_alpha * aux

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Evaluation: prediction_step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Run inference and return ``(loss, scores, labels)`` so that
        ``compute_metrics`` can calculate CaseHOLD accuracy from the
        accumulated ``(B, 5)`` score matrices.
        """
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            scores, _, shift_logits, shift_ids, shifted_mask, B, C, L = (
                self._forward_and_score(model, inputs)
            )
            labels = inputs["labels"]

            loss = self._ranking_loss(scores, labels, B, C)
            if self.aux_ce_alpha > 0.0:
                aux = self._aux_ce_loss(
                    shift_logits, shift_ids, shifted_mask, labels, B, C, L
                )
                loss = loss + self.aux_ce_alpha * aux

        loss = loss.mean().detach()

        if prediction_loss_only:
            return loss, None, None

        return loss, scores.detach(), labels.detach()


# ---------------------------------------------------------------------------
# Callback: per-epoch negative resampling
# ---------------------------------------------------------------------------

class NegativeResamplerCallback(TrainerCallback):
    """
    Resample negative candidates at the start of each epoch.

    Pass an instance of this callback to ``RankingTrainer.add_callback``
    together with a dataset that implements ``resample_negatives(seed)``
    (e.g. ``LedgarRankingDataset``).

    A fresh seed is derived as ``base_seed + epoch + 1`` so the epoch-0
    resampling differs from the fixed seed used during dataset construction.
    """

    def __init__(self, train_dataset, base_seed: int = 42):
        self.train_dataset = train_dataset
        self.base_seed = base_seed

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not hasattr(self.train_dataset, "resample_negatives"):
            return
        epoch = int(state.epoch) if state.epoch is not None else 0
        new_seed = self.base_seed + epoch + 1
        self.train_dataset.resample_negatives(new_seed)
        logger.info(
            "NegativeResamplerCallback: resampled negatives with seed %d (epoch %d)",
            new_seed, epoch,
        )
