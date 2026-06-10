from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, cast

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llm_adapter import save_model


def verbalize_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").strip()


class LEDGARRankingDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        max_examples: int | None = None,
    ):
        self.dataset: Any = load_dataset("lex_glue", "ledgar", split=split)
        if max_examples is not None:
            self.dataset = self.dataset.select(range(min(max_examples, len(self.dataset))))
        self.label_names = self.dataset.features["label"].names

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ex = cast(Dict[str, Any], self.dataset[idx])
        return {
            "text": str(ex["text"]).strip(),
            "label_id": int(ex["label"]),
        }


def ranking_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "texts": [str(item["text"]) for item in batch],
        "label_ids": torch.tensor([int(cast(int, item["label_id"])) for item in batch], dtype=torch.long),
    }


def _as_int(value: object, default: int) -> int:
    return default if value is None else int(cast(int, value))


def _as_float(value: object, default: float) -> float:
    return default if value is None else float(cast(float, value))


def _as_str(value: object, default: str) -> str:
    return default if value is None else str(cast(str, value))


def _sample_negative_label_ids(
    gold_label_id: int,
    num_labels: int,
    num_negatives: int,
    rng: random.Random,
) -> List[int]:
    population = [label_id for label_id in range(num_labels) if label_id != gold_label_id]
    if not population:
        return []
    sample_size = min(num_negatives, len(population))
    return rng.sample(population, sample_size)


def _build_label_token_cache(
    tokenizer,
    label_names: Sequence[str],
    max_label_length: int,
) -> List[List[int]]:
    cached_label_token_ids: List[List[int]] = []
    fallback_token_id = tokenizer.eos_token_id
    if fallback_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for LEDGAR ranking training.")

    for label_name in label_names:
        label_text = " " + verbalize_label(label_name)
        label_token_ids = tokenizer(label_text, add_special_tokens=False).input_ids[:max_label_length]
        if not label_token_ids:
            label_token_ids = [fallback_token_id]
        cached_label_token_ids.append(label_token_ids)

    return cached_label_token_ids

# This could be improved by using the tokenizer's batch encoding and padding features
def _build_scoring_batch(
    tokenizer,
    texts: Sequence[str],
    candidate_label_ids: Sequence[Sequence[int]],
    label_token_cache: Sequence[Sequence[int]],
    max_length: int,
    truncation_side: str,
) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
    input_id_rows: List[List[int]] = []
    attention_rows: List[List[int]] = []
    label_lengths: List[int] = []

    assert len(texts) == len(candidate_label_ids), "Mismatched number of texts and candidate label sets."
    
    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side
    try:
        for text, candidates in zip(texts, candidate_label_ids):
            prompt = f"Contract provision:\n{text.strip()}\n\nCategory:"
            prompt_token_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=False,
            ).input_ids

            for label_id in candidates:
                label_token_ids = list(label_token_cache[label_id])

                max_prompt_len = max(1, max_length - len(label_token_ids))
                if len(prompt_token_ids) > max_prompt_len:
                    if truncation_side == "left":
                        prompt_ids = prompt_token_ids[-max_prompt_len:]
                    else:
                        prompt_ids = prompt_token_ids[:max_prompt_len]
                else:
                    prompt_ids = prompt_token_ids

                input_ids = prompt_ids + label_token_ids
                attention_mask = [1] * len(input_ids)

                input_id_rows.append(input_ids)
                attention_rows.append(attention_mask)
                label_lengths.append(len(label_token_ids))
    finally:
        tokenizer.truncation_side = old_side

    max_seq_len = max(len(row) for row in input_id_rows)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    padded_input_ids = []
    padded_attention = []
    for input_ids, attention_mask in zip(input_id_rows, attention_rows):
        pad_len = max_seq_len - len(input_ids)
        padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
        padded_attention.append(attention_mask + [0] * pad_len)

    return (
        torch.tensor(padded_input_ids, dtype=torch.long),
        torch.tensor(padded_attention, dtype=torch.long),
        label_lengths,
    )


def _score_candidates(
    model,
    tokenizer,
    texts: Sequence[str],
    candidate_label_ids: Sequence[Sequence[int]],
    label_token_cache: Sequence[Sequence[int]],
    device: torch.device,
    max_length: int,
    truncation_side: str,
) -> torch.Tensor:
    input_ids, attention_mask, label_lengths = _build_scoring_batch(
        tokenizer=tokenizer,
        texts=texts,
        candidate_label_ids=candidate_label_ids,
        label_token_cache=label_token_cache,
        max_length=max_length,
        truncation_side=truncation_side,
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    flat_scores: List[torch.Tensor] = []
    for row_idx, label_len in enumerate(label_lengths):
        valid_len = int(shift_mask[row_idx].sum().item())
        start_pos = valid_len - label_len
        if valid_len <= 0 or label_len <= 0 or start_pos < 0:
            flat_scores.append(token_log_probs.new_tensor(float("-inf")))
            continue
        flat_scores.append(token_log_probs[row_idx, start_pos:valid_len].mean())

    num_candidates = len(candidate_label_ids[0])
    return torch.stack(flat_scores).view(len(texts), num_candidates)


def _evaluate_full_label_accuracy(
    model,
    dataloader,
    tokenizer,
    label_names: Sequence[str],
    label_token_cache: Sequence[Sequence[int]],
    device: torch.device,
    max_length: int,
    truncation_side: str,
    label_batch_size: int,
) -> tuple[float, float]:
    """Evaluate the full LEDGAR label set in chunks.

    The returned loss is defined as 1 - accuracy so existing checkpoint logic
    can continue to minimize it.
    """
    model.eval()
    total_correct = 0
    total_examples = 0
    all_label_ids = list(range(len(label_names)))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Full-label validation"):
            texts = batch["texts"]
            gold_label_ids = batch["label_ids"].tolist()

            chunk_scores: List[torch.Tensor] = []
            for start in range(0, len(all_label_ids), label_batch_size):
                label_chunk = all_label_ids[start:start + label_batch_size]
                scores = _score_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    candidate_label_ids=[label_chunk for _ in texts],
                    label_token_cache=label_token_cache,
                    device=device,
                    max_length=max_length,
                    truncation_side=truncation_side,
                )
                chunk_scores.append(scores)

            scores = torch.cat(chunk_scores, dim=1)
            predictions = scores.argmax(dim=1)
            gold = torch.tensor(gold_label_ids, device=predictions.device)
            total_correct += int((predictions == gold).sum().item())
            total_examples += len(texts)

    accuracy = total_correct / max(total_examples, 1)
    loss = 1.0 - accuracy
    return loss, accuracy


def _evaluate_ranking_loss(
    model,
    dataloader,
    tokenizer,
    label_names: Sequence[str],
    label_token_cache: Sequence[Sequence[int]],
    device: torch.device,
    max_length: int,
    truncation_side: str,
    num_negatives: int,
    margin: float,
    seed: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    rng = random.Random(seed)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ranking validation"):
            texts = batch["texts"]
            gold_label_ids = batch["label_ids"].tolist()
            candidate_label_ids = [
                [gold_label_id] + _sample_negative_label_ids(gold_label_id, len(label_names), num_negatives, rng)
                for gold_label_id in gold_label_ids
            ]

            scores = _score_candidates(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                candidate_label_ids=candidate_label_ids,
                label_token_cache=label_token_cache,
                device=device,
                max_length=max_length,
                truncation_side=truncation_side,
            )

            positive_scores = scores[:, :1]
            negative_scores = scores[:, 1:]
            if negative_scores.numel() == 0:
                loss = -positive_scores.mean()
            else:
                loss = F.relu(margin - positive_scores + negative_scores).mean()

            total_loss += loss.item() * len(texts)
            total_correct += int((scores.argmax(dim=1) == 0).sum().item())
            total_examples += len(texts)

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return avg_loss, accuracy


def run_ledgar_label_ranking_post_training(
    model,
    tokenizer,
    adapter_type: str,
    adapter_config,
    training_cfg: Dict[str, object],
    data_cfg: Dict[str, object],
    output_dir: str,
    device: torch.device,
    *,
    shift_regularization: bool = False,
    variational_modeling: bool = False,
    aggregation_strategy: str = "attention",
    layer_adapter_max_temp: float = 2.0,
    layer_adapter_min_temp: float = 0.8,
) -> None:
    seed = _as_int(training_cfg.get("seed"), 42)
    batch_size = _as_int(training_cfg.get("batch_size"), 8)
    num_workers = _as_int(training_cfg.get("num_workers"), 0)
    num_epochs = _as_int(training_cfg.get("num_epochs"), 3)
    learning_rate = _as_float(training_cfg.get("learning_rate"), 1e-4)
    weight_decay = _as_float(training_cfg.get("weight_decay"), 0.01)
    adam_beta1 = _as_float(training_cfg.get("adam_beta1"), 0.9)
    adam_beta2 = _as_float(training_cfg.get("adam_beta2"), 0.999)
    progress_interval = _as_int(training_cfg.get("progress_interval"), 10)
    val_interval = _as_int(training_cfg.get("val_interval"), 50)
    early_stopping_patience = _as_int(training_cfg.get("early_stopping_patience"), 3)
    early_stopping_min_delta = _as_float(training_cfg.get("early_stopping_min_delta"), 1e-4)
    gradient_clip_norm = _as_float(training_cfg.get("gradient_clip_norm"), 1.0)
    use_amp = bool(training_cfg.get("use_amp", True)) and device.type == "cuda"
    kl_loss_weight = _as_float(training_cfg.get("kl_loss_weight"), 0.0)
    validation_label_batch_size = _as_int(training_cfg.get("validation_label_batch_size"), 32)

    max_length = _as_int(training_cfg.get("context_size"), 512)
    max_label_length = _as_int(data_cfg.get("max_label_length", training_cfg.get("max_label_length")), 24)
    truncation_side = _as_str(data_cfg.get("truncation_side", training_cfg.get("truncation_side")), "left")
    num_negatives = _as_int(training_cfg.get("ranking_num_negatives"), 7)
    margin = _as_float(training_cfg.get("ranking_margin"), 0.2)

    train_dataset = LEDGARRankingDataset(
        split=_as_str(data_cfg.get("train_split"), "train"),
        max_examples=cast(int | None, data_cfg.get("max_train_examples")),
    )
    if len(train_dataset) == 0:
        raise RuntimeError("LEDGAR ranking dataset is empty.")
    
    val_dataset = LEDGARRankingDataset(
        split=_as_str(data_cfg.get("validation_split"), "validation"),
        max_examples=cast(int | None, data_cfg.get("max_validation_examples")),
    ) if data_cfg.get("validation_split") else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=ranking_collate,
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=ranking_collate,
        )

    raw_model = model.to(device)
    trainable_params = [param for param in raw_model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters available for LEDGAR ranking post-training.")

    label_names = train_dataset.label_names
    label_token_cache = _build_label_token_cache(
        tokenizer=tokenizer,
        label_names=label_names,
        max_label_length=max_label_length,
    )

    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val_loss = float("inf")
    patience_counter = 0
    step = 0
    rng = random.Random(seed)

    os_path_trace = f"{output_dir}/checkpoint-trace"
    os_path_best = f"{output_dir}/checkpoint-best"

    print("[post_training] LEDGAR ranking configuration:")
    print(f"  negatives per example: {num_negatives}")
    print(f"  ranking margin: {margin}")
    print(f"  max_length: {max_length}")
    print(f"  max_label_length: {max_label_length}")
    print(f"  truncation_side: {truncation_side}")
    print(f"  validation_label_batch_size: {validation_label_batch_size}")

    for epoch in range(num_epochs):
        raw_model.train()
        epoch_loss = 0.0
        epoch_examples = 0
        progress_bar = tqdm(train_dataloader, desc=f"LEDGAR ranking epoch {epoch + 1}")
        current_temperature = None

        for batch in progress_bar:
            texts = batch["texts"]
            gold_label_ids = batch["label_ids"].tolist()
            candidate_label_ids = [
                [gold_label_id] + _sample_negative_label_ids(gold_label_id, len(label_names), num_negatives, rng)
                for gold_label_id in gold_label_ids
            ]

            optimizer.zero_grad(set_to_none=True)

            current_temperature = None
            if adapter_type == "layer_adapter" and aggregation_strategy == "attention":
                progress_ratio = min((step + 1) / max(1, len(train_dataloader) * num_epochs // 10), 1.0)
                current_temperature = layer_adapter_max_temp - progress_ratio * (
                    layer_adapter_max_temp - layer_adapter_min_temp
                )
                raw_model.transformer.encoder.set_attention_temperature(current_temperature)
                if isinstance(adapter_config, dict):
                    adapter_config["attention_temperature"] = current_temperature

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                scores = _score_candidates(
                    model=raw_model,
                    tokenizer=tokenizer,
                    texts=texts,
                    candidate_label_ids=candidate_label_ids,
                    label_token_cache=label_token_cache,
                    device=device,
                    max_length=max_length,
                    truncation_side=truncation_side,
                )

                positive_scores = scores[:, :1]
                negative_scores = scores[:, 1:]
                if negative_scores.numel() == 0:
                    loss = -positive_scores.mean()
                else:
                    loss = F.relu(margin - positive_scores + negative_scores).mean()

                if adapter_type == "layer_adapter":
                    shift_loss = raw_model.transformer.encoder.get_delta_loss()
                    if shift_regularization:
                        loss = loss + shift_loss

                if adapter_type == "layer_adapter" and variational_modeling:
                    loss = loss + kl_loss_weight * raw_model.transformer.encoder.get_kl_loss()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_norm)
                optimizer.step()

            batch_size_actual = len(texts)
            epoch_loss += loss.detach().float().item() * batch_size_actual
            epoch_examples += batch_size_actual
            step += 1

            if step % progress_interval == 0:
                sampled_accuracy = float((scores.argmax(dim=1) == 0).float().mean().item())
                progress_bar.set_postfix(
                    loss=f"{loss.detach().float().item():.4f}",
                    sampled_acc=f"{sampled_accuracy:.3f}",
                    temp=(f"{current_temperature:.2f}" if current_temperature is not None else "-")
                )

            if val_dataloader is not None and (step % val_interval == 0):
                val_loss, val_acc = _evaluate_full_label_accuracy(
                    model=raw_model,
                    dataloader=val_dataloader,
                    tokenizer=tokenizer,
                    label_names=label_names,
                    label_token_cache=label_token_cache,
                    device=device,
                    max_length=max_length,
                    truncation_side=truncation_side,
                    label_batch_size=validation_label_batch_size,
                )
                print(f"[post_training] Full-label val loss: {val_loss:.4f}, full-label val acc: {val_acc:.4f}")
                save_model(raw_model, adapter_type, adapter_config, os_path_trace)

                if val_loss < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_model(raw_model, adapter_type, adapter_config, os_path_best)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience and early_stopping_patience > 0:
                        print("[post_training] Early stopping triggered for LEDGAR ranking.")
                        return

        avg_epoch_loss = epoch_loss / max(epoch_examples, 1)
        print(f"[post_training] LEDGAR ranking epoch {epoch + 1}/{num_epochs} avg loss: {avg_epoch_loss:.4f}")
        save_model(raw_model, adapter_type, adapter_config, os_path_trace)

    if val_dataloader is None:
        save_model(raw_model, adapter_type, adapter_config, os_path_best)