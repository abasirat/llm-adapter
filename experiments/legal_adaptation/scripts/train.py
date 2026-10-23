import json
import hashlib
import numbers
import tempfile
from pathlib import Path
from torch.utils.data import BatchSampler, IterableDataset, RandomSampler, SequentialSampler, Subset
import numpy as np
import torch
import os
import warnings
import argparse
from functools import partial
from torch.utils.data import ConcatDataset, Dataset, random_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import math
import wandb
from datetime import datetime
from llm_adapter import setup_model, save_model, load_model
from peft import LoraConfig, TaskType, PromptEncoderConfig
import random
import yaml



def set_device(device_name=None):
    if device_name:
        return torch.device(device_name)
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # The following settings are due to some CUDA instability on A100
    if device == "cuda":
        #torch.backends.cuda.matmul.allow_tf32 = False
        #torch.backends.cudnn.allow_tf32 = False
        #torch.set_float32_matmul_precision("highest")

        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        #torch.use_deterministic_algorithms(True)
        pass

    return torch.device(device)

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TokenBinDataset(Dataset):
    def __init__(
        self,
        bin_path: str,
        context_size: int = 1024,
        max_tokens: int = None,
        start_token: int = 0,
        end_token: int = None,
    ):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Token-bin file not found: {bin_path}")

        meta_path = os.path.splitext(bin_path)[0] + ".json"

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            dtype_name = (
                meta.get("dtype")
                or meta.get("storage", {}).get("dtype")
                or "uint16"
            )
            dtype = np.dtype(dtype_name)

            tokenizer_meta = meta.get("tokenizer", meta)

            print(f"TokenBinDataset: loaded metadata from {meta_path}")
            print(f"  tokenizer : {tokenizer_meta.get('tokenizer_name', 'unknown')}")
            print(
                f"  EOS token : {tokenizer_meta.get('eos_token', 'unknown')} "
                f"(id={tokenizer_meta.get('eos_token_id', '?')})"
            )
        else:
            dtype = np.uint16
            print("TokenBinDataset: no metadata sidecar found, defaulting to uint16")

        self.context_size = context_size
        self.start_token = start_token

        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        if max_tokens is not None:
            self.data = self.data[:max_tokens]

        if end_token is None:
            end_token = len(self.data)

        self.end_token = min(end_token, len(self.data))
        self.num_tokens = max(0, self.end_token - self.start_token)

        # Drop incomplete final chunk.
        self.num_chunks = self.num_tokens // context_size

        print(f"  total tokens : {self.num_tokens:,}")
        print(f"  context size : {context_size}")
        print(f"  chunks       : {self.num_chunks:,}")
        print(f"  token range  : {self.start_token:,} - {self.end_token:,}")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.num_chunks + idx

        if idx < 0 or idx >= self.num_chunks:
            raise IndexError("Index out of range")

        start = self.start_token + idx * self.context_size
        end = start + self.context_size

        chunk = self.data[start:end].astype(np.int64)
        input_ids = torch.from_numpy(chunk)

        return {"input_ids": input_ids}

class MixedDataset(Dataset):
    def __init__(self, datasets, probabilities=None):
        self.datasets = datasets
        total_len = sum(len(d) for d in datasets)

        if total_len == 0:
            raise ValueError("MixedDataset received only empty datasets.")

        if probabilities is None:
            probabilities = [len(d) / total_len for d in datasets]

        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("Dataset sampling probabilities must sum to 1.")

        self.probabilities = probabilities
        self.total_len = total_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.datasets), p=self.probabilities)
        dataset = self.datasets[dataset_idx]
        item_idx = np.random.randint(0, len(dataset))
        return dataset[item_idx]
        
def build_token_bin_datasets(
    train_bin_paths,
    val_bin_paths,
    context_size,
    val_fraction=0.0,
    seed=42,
    max_train_tokens=None,
):
    """Prefer explicit held-out files; retain seeded chunk splitting as a fallback."""
    if not train_bin_paths:
        raise ValueError("At least one training token-bin file is required.")
    if context_size < 2:
        raise ValueError("context_size must be at least 2 for next-token prediction.")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if max_train_tokens is not None and max_train_tokens <= 0:
        raise ValueError("max_train_tokens must be positive.")

    train_bin_paths = list(train_bin_paths)
    val_bin_paths = list(val_bin_paths or [])
    for path in train_bin_paths + val_bin_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Token-bin file not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"Token-bin file is empty: {path}")
    for index, val_path in enumerate(val_bin_paths):
        if any(os.path.samefile(val_path, path) for path in train_bin_paths):
            raise ValueError(f"Validation file is also used for training: {val_path}")
        if any(os.path.samefile(val_path, path) for path in val_bin_paths[:index]):
            raise ValueError(f"Validation file is repeated: {val_path}")

    tokens_per_path = max_train_tokens // len(train_bin_paths) if max_train_tokens is not None else None
    if tokens_per_path is not None:
        if tokens_per_path < context_size:
            raise ValueError("The training token budget must allow a complete chunk per training file.")
        print(f"Limiting training to {max_train_tokens:,} tokens total ({tokens_per_path:,} per path).")
    if val_bin_paths:
        print(f"Using held-out validation files: {val_bin_paths}; val_fraction is ignored.")
    elif val_fraction > 0:
        warnings.warn(
            "Splitting packed training chunks for validation can share source documents; "
            "prefer explicit held-out validation files.",
            RuntimeWarning,
        )

    train_datasets = []
    val_datasets = []
    for path in train_bin_paths:
        dataset = TokenBinDataset(path, context_size=context_size, max_tokens=tokens_per_path)
        if not val_bin_paths and val_fraction > 0:
            val_size = int(len(dataset) * val_fraction)
            if val_size > 0:
                training, validation = random_split(
                    dataset,
                    [len(dataset) - val_size, val_size],
                    generator=torch.Generator().manual_seed(seed),
                )
                train_datasets.append(training)
                val_datasets.append(validation)
                continue
            warnings.warn(f"Validation fraction produced no complete chunks for {path}.", RuntimeWarning)
        train_datasets.append(dataset)

    for path in val_bin_paths:
        dataset = TokenBinDataset(path, context_size=context_size)
        if len(dataset) == 0:
            raise ValueError(f"Validation file has no complete {context_size}-token chunks: {path}")
        val_datasets.append(dataset)
    if not any(len(dataset) for dataset in train_datasets):
        raise ValueError("Training data has no complete chunks after applying the token budget or split.")

    training = MixedDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    validation = (
        ConcatDataset(val_datasets) if len(val_datasets) > 1
        else val_datasets[0] if val_datasets else None
    )
    return training, validation


def token_bin_collate(batch, context_size):
    examples = [x["input_ids"] for x in batch if len(x["input_ids"]) == context_size]

    if len(examples) == 0:
        raise ValueError("Empty batch after filtering short examples.")

    input_ids = torch.stack(examples)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }, None

class WarmUpCosineDecayScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, base_lr: float):
        """
        Warm-up with cosine decay scheduler.
        
        Args:
            optimizer (Optimizer): PyTorch optimizer.
            warmup_steps (int): Number of warm-up steps.
            total_steps (int): Total number of training steps.
            base_lr (float): Base learning rate.
        """
        if not isinstance(total_steps, numbers.Integral) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer.")
        if not isinstance(warmup_steps, numbers.Integral) or not 0 <= warmup_steps <= total_steps:
            raise ValueError("warmup_steps must be between zero and total_steps.")
        if not math.isfinite(base_lr) or base_lr < 0:
            raise ValueError("base_lr must be finite and nonnegative.")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.step_count = 0
        self._set_next_lr()

    def step(self, step_count:int=None):
        """Count completed optimizer updates and prepare the next update's rate."""
        increment = 1 if step_count is None else step_count
        if not isinstance(increment, numbers.Integral) or increment < 0:
            raise ValueError("step_count increment must be a nonnegative integer.")
        self.step_count += increment
        self._set_next_lr()

    def _set_next_lr(self):
        if self.step_count < self.warmup_steps:
            lr = self.base_lr * ((self.step_count + 1) / self.warmup_steps)
        else:
            # The warmup peak is used once. Decay reaches zero AFTER the final
            # update, so a one-update run with no warmup still changes weights.
            decay_start = max(0, self.warmup_steps - 1)
            progress = (self.step_count - decay_start) / (self.total_steps - decay_start)
            progress = min(max(progress, 0.0), 1.0)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """Return the rate prepared for the next optimizer update."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {"schema_version": 1, "step_count": self.step_count,
                "warmup_steps": self.warmup_steps, "total_steps": self.total_steps,
                "base_lr": self.base_lr}

    def load_state_dict(self, state):
        for key in ("schema_version", "warmup_steps", "total_steps", "base_lr"):
            expected = 1 if key == "schema_version" else getattr(self, key)
            if state.get(key) != expected:
                raise ValueError(f"Cannot resume: LR scheduler {key} does not match.")
        completed = state.get("step_count")
        if not isinstance(completed, numbers.Integral) or not 0 <= completed <= self.total_steps:
            raise ValueError("Cannot resume: invalid LR scheduler update count.")
        self.step_count = int(completed)
        self._set_next_lr()

class KLWeightScheduler:
    def __init__(
        self,
        max_kl_weight: float,
        warmup_steps: int,
        schedule_type: str = "linear",
        start_kl_weight: float = 0.0,
    ):
        self.max_kl_weight = max_kl_weight
        self.warmup_steps = max(1, warmup_steps)
        self.schedule_type = schedule_type
        self.start_kl_weight = start_kl_weight
        self.step_count = 0

    def step(self):
        self.step_count += 1

    def get_weight(self):
        progress = min(self.step_count / self.warmup_steps, 1.0)

        if self.schedule_type == "linear":
            weight = self.start_kl_weight + progress * (
                self.max_kl_weight - self.start_kl_weight
            )

        elif self.schedule_type == "sigmoid":
            # Smooth, slow start, faster middle, slow end
            x = 12 * (progress - 0.5)
            sigmoid = 1 / (1 + math.exp(-x))
            weight = self.start_kl_weight + sigmoid * (
                self.max_kl_weight - self.start_kl_weight
            )

        elif self.schedule_type == "cosine":
            weight = self.start_kl_weight + (
                0.5 * (1 - math.cos(math.pi * progress))
            ) * (self.max_kl_weight - self.start_kl_weight)

        else:
            raise ValueError(f"Unknown KL schedule type: {self.schedule_type}")

        return weight

class LinearTemperatureScheduler:
    def __init__(self, max_temperature: float, min_temperature: float, warmup_steps: int):
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.warmup_steps = max(1, warmup_steps)
        self.step_count = 0

    def step(self):
        self.step_count += 1

    def get_temperature(self):
        progress = min(self.step_count / self.warmup_steps, 1.0)
        temperature = self.max_temperature - progress * (self.max_temperature - self.min_temperature)
        return temperature
    
def validate(model, val_dataloader, device, device_type, use_amp, progress_interval=10):
    """
    Perform validation on a dataset and compute average loss.

    Args:
        model: PyTorch model to validate
        val_dataloader: DataLoader for validation data
        device: torch.device to run on
        device_type: device type ('cuda', 'mps', 'cpu')
        use_amp: whether to use automatic mixed precision
        progress_interval: how often to update the progress bar
    Returns:
        avg_loss: mean loss weighted by unmasked next-token targets
            (weighted by examples when labels are absent)
        num_batches: number of batches processed
    """
    was_training = model.training
    try:
        model.eval()
        val_loss = 0.0
        loss_weight = 0
        num_batches = 0

        progress_bar = tqdm(desc="Validating", total=len(val_dataloader))
        with torch.no_grad():
            for val_batch, _ in val_dataloader:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                labels = val_batch.get("labels")
                batch_weight = (
                    int(labels[..., 1:].ne(-100).sum().item())
                    if labels is not None else val_batch["input_ids"].size(0)
                )
                if batch_weight > 0:
                    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                        outputs = model(**val_batch)
                    batch_loss = outputs.loss.detach().float().item()
                    if not math.isfinite(batch_loss):
                        raise ValueError("Validation loss is not finite.")
                    val_loss += batch_loss * batch_weight
                    loss_weight += batch_weight
                num_batches += 1
                if num_batches % progress_interval == 0 or num_batches >= len(val_dataloader):
                    update = progress_interval if num_batches % progress_interval == 0 else (num_batches % progress_interval)
                    progress_bar.update(update)
            

        if loss_weight == 0:
            raise ValueError("Validation data contains no next-token targets to score.")
        avg_loss = val_loss / loss_weight
        return avg_loss, num_batches
    finally:
        model.train(was_training)


def _resume_seed(seed, epoch, index):
    key = f"{int(seed)}:{int(epoch)}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 63)


def dataset_resume_signature(dataset):
    """Describe supported data sources and their effective indexing settings."""
    name = f"{type(dataset).__module__}.{type(dataset).__qualname__}"
    result = {"class": name, "length": len(dataset)}
    if isinstance(dataset, Subset):
        indices = [int(index) for index in dataset.indices]
        result.update(dataset=dataset_resume_signature(dataset.dataset), indices_sha256=
                      hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest())
    elif isinstance(dataset, ConcatDataset) or (
        hasattr(dataset, "datasets") and hasattr(dataset, "probabilities")
    ):
        result["datasets"] = [dataset_resume_signature(item) for item in dataset.datasets]
        if hasattr(dataset, "probabilities"):
            result["probabilities"] = [float(value) for value in dataset.probabilities]
    elif hasattr(dataset, "data") and hasattr(dataset, "context_size"):
        data = dataset.data
        result.update(context_size=int(dataset.context_size),
                      start_token=int(getattr(dataset, "start_token", 0)),
                      end_token=int(getattr(dataset, "end_token", len(data))),
                      mapped_tokens=len(data), dtype=str(data.dtype))
        filename = getattr(data, "filename", None)
        if filename is not None:
            path = Path(filename).resolve(strict=True)
            stat = path.stat()
            result["file"] = {"path": str(path), "device": stat.st_dev,
                              "inode": stat.st_ino, "size": stat.st_size,
                              "mtime_ns": stat.st_mtime_ns}
            sidecar = path.with_suffix(".json")
            result["sidecar_sha256"] = (
                hashlib.sha256(sidecar.read_bytes()).hexdigest() if sidecar.is_file() else None
            )
    return result


class ResumableBatchSampler:
    """Track consumed batches independently of DataLoader's prefetch iterator."""
    def __init__(self, dataset_size, batch_size, shuffle=True, seed=42,
                 drop_last=False, signature=None):
        if not isinstance(dataset_size, numbers.Integral) or dataset_size < 0:
            raise ValueError("dataset_size must be a nonnegative integer.")
        if not isinstance(batch_size, numbers.Integral) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.dataset_size, self.batch_size = int(dataset_size), int(batch_size)
        self.shuffle, self.seed, self.drop_last = bool(shuffle), int(seed), bool(drop_last)
        self.signature = signature
        self.total_batches = (self.dataset_size // self.batch_size if self.drop_last
                              else math.ceil(self.dataset_size / self.batch_size))
        self.epoch, self.next_batch, self.order = -1, 0, []
        self.set_epoch(0)

    def __len__(self):
        # Scheduling must always use the full epoch, even during partial resume.
        return self.total_batches

    def set_epoch(self, epoch):
        if not isinstance(epoch, numbers.Integral) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer.")
        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch, self.next_batch = epoch, 0
            generator = torch.Generator().manual_seed(_resume_seed(self.seed, epoch, "order"))
            self.order = (torch.randperm(self.dataset_size, generator=generator).tolist()
                          if self.shuffle else list(range(self.dataset_size)))

    def __iter__(self):
        epoch, start = self.epoch, self.next_batch
        for batch in range(start, self.total_batches):
            indices = self.order[batch * self.batch_size:(batch + 1) * self.batch_size]
            yield [(epoch, index) for index in indices]

    def mark_consumed(self):
        if self.next_batch >= self.total_batches:
            raise RuntimeError("All batches in this epoch have already been consumed.")
        self.next_batch += 1

    def state_dict(self):
        return {"schema_version": 1, "dataset_size": self.dataset_size,
                "batch_size": self.batch_size, "shuffle": self.shuffle,
                "seed": self.seed, "drop_last": self.drop_last,
                "signature": self.signature, "epoch": self.epoch,
                "next_batch": self.next_batch, "order": torch.tensor(self.order, dtype=torch.int64)}

    def load_state_dict(self, state):
        for key in ("schema_version", "dataset_size", "batch_size", "shuffle",
                    "seed", "drop_last", "signature"):
            expected = 1 if key == "schema_version" else getattr(self, key)
            if key not in state or state[key] != expected:
                raise ValueError(f"Cannot resume: batch sampler {key} does not match.")
        epoch, cursor, order = state.get("epoch"), state.get("next_batch"), state.get("order")
        if isinstance(order, torch.Tensor):
            if order.ndim != 1 or order.dtype != torch.int64:
                raise ValueError("Cannot resume: malformed saved sample order.")
            order = order.tolist()
        if (not isinstance(epoch, numbers.Integral) or epoch < 0
                or not isinstance(cursor, numbers.Integral) or not 0 <= cursor <= self.total_batches
                or not isinstance(order, (list, tuple)) or len(order) != self.dataset_size
                or any(not isinstance(index, numbers.Integral) for index in order)
                or sorted(order) != list(range(self.dataset_size))):
            raise ValueError("Cannot resume: invalid epoch, consumed cursor, or sample permutation.")
        if not self.shuffle and list(order) != list(range(self.dataset_size)):
            raise ValueError("Cannot resume: sequential sampler order was changed.")
        self.epoch, self.next_batch, self.order = int(epoch), int(cursor), list(order)


class EpochSeededDataset(Dataset):
    """Seed one item without consuming the model process's CPU RNG streams."""
    def __init__(self, dataset, seed=42):
        self.dataset, self.seed = dataset, int(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, index = key
        seed = _resume_seed(self.seed, epoch, index)
        torch_state, numpy_state, python_state = (
            torch.get_rng_state(), np.random.get_state(), random.getstate()
        )
        try:
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            np.random.seed(seed % (2 ** 32))
            random.seed(seed)
            return self.dataset[index]
        finally:
            torch.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def _check_resume_loader(loader):
    if isinstance(loader.dataset, IterableDataset):
        raise ValueError("Exact resume requires a map-style dataset.")
    if type(loader.batch_sampler) is not BatchSampler or loader.batch_size is None:
        raise ValueError("Exact resume does not support custom batch samplers or unbatched loaders.")
    if type(loader.sampler) not in (SequentialSampler, RandomSampler):
        raise ValueError("Exact resume does not support custom or distributed samplers.")
    if isinstance(loader.sampler, RandomSampler) and (
        loader.sampler.replacement or loader.sampler.num_samples != len(loader.dataset)
    ):
        raise ValueError("Exact resume requires nonreplacement sampling over the full dataset.")
    if not getattr(loader, "in_order", True):
        raise ValueError("Exact resume requires DataLoader in_order=True.")
    if loader.worker_init_fn is not None:
        raise ValueError("Exact resume does not support custom worker_init_fn functions.")


def _clone_resume_loader(loader, dataset, generator, batch_sampler=None):
    kwargs = dict(dataset=dataset, num_workers=loader.num_workers,
                  collate_fn=loader.collate_fn, pin_memory=loader.pin_memory,
                  timeout=loader.timeout, generator=generator,
                  persistent_workers=loader.persistent_workers)
    if loader.num_workers > 0:
        kwargs.update(prefetch_factor=loader.prefetch_factor,
                      multiprocessing_context=loader.multiprocessing_context)
    if getattr(loader, "pin_memory_device", ""):
        kwargs["pin_memory_device"] = loader.pin_memory_device
    if hasattr(loader, "in_order"):
        kwargs["in_order"] = loader.in_order
    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        sampler = (RandomSampler(dataset, generator=generator)
                   if isinstance(loader.sampler, RandomSampler) else SequentialSampler(dataset))
        kwargs.update(batch_size=loader.batch_size, sampler=sampler, drop_last=loader.drop_last)
    return DataLoader(**kwargs)


def make_resumable_loader(loader, seed=42):
    """Return an isolated loader and its consumption-aware batch sampler."""
    _check_resume_loader(loader)
    sampler = ResumableBatchSampler(
        len(loader.dataset), loader.batch_size, isinstance(loader.sampler, RandomSampler),
        seed, loader.drop_last, dataset_resume_signature(loader.dataset),
    )
    cloned = _clone_resume_loader(loader, EpochSeededDataset(loader.dataset, seed),
                                  torch.Generator(), sampler)
    cloned._resume_loader_seed = int(seed)
    set_loader_epoch(cloned, 0)
    return cloned, sampler


def make_evaluation_loader(loader, seed=42):
    """Isolate sequential evaluation startup RNG without replacing the loader.

    Evaluation items/collation must be deterministic, as in the project's
    token-bin and supervised task datasets. Keeping the same loader also keeps
    its worker pool, collator, and any external references intact.
    """
    _check_resume_loader(loader)
    if type(loader.sampler) is not SequentialSampler:
        raise ValueError("Exact resume requires sequential validation/dev loaders.")
    loader.generator = torch.Generator()
    loader._resume_loader_seed = int(seed)
    set_loader_epoch(loader, 0)
    return loader


def set_loader_epoch(loader, epoch):
    """Call before EVERY iterator reconstruction, including a mid-epoch resume."""
    loader.generator.manual_seed(_resume_seed(loader._resume_loader_seed, epoch, "loader"))
    if isinstance(loader.batch_sampler, ResumableBatchSampler):
        loader.batch_sampler.set_epoch(epoch)


def _training_file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_training_write(path, writer):
    path = os.path.abspath(path)
    fd, temporary = tempfile.mkstemp(prefix=Path(path).name + ".", dir=os.path.dirname(path))
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_training_status(
    status_path,
    state_path,
    checkpoint_path,
    best_checkpoint_path,
    epoch,
    step,
    batch_index,
    best_val_loss,
    patience_counter,
    optimizer,
    scaler,
    scheduler,
    kl_scheduler,
    temp_scheduler,
    epoch_total_loss=0.0,
    epoch_num_batches=0,
    data_state=None,
    resume_config=None,
    progress_state=None,
    early_stopped=False,
    device_type=None,
):
    """Save resumable training state to a JSON status file and a .pt state file."""
    state = {
        "resume_version": 2,
        "data_state": data_state,
        "resume_config": resume_config,
        "progress_state": progress_state or {},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    if torch.cuda.is_available():
        state["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    if device_type == "mps":
        state["rng_state"]["mps"] = torch.mps.get_rng_state()
    _atomic_training_write(state_path, lambda path: torch.save(state, path))

    status = {
        "resume_version": 2,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": _training_file_sha256(checkpoint_path),
        "state_sha256": _training_file_sha256(state_path),
        "best_checkpoint_path": best_checkpoint_path,
        "state_path": os.path.abspath(state_path),
        "epoch": epoch,
        "step": step,
        "batch_index": batch_index,
        "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
        "patience_counter": patience_counter,
        "scheduler_step_count": scheduler.step_count if scheduler is not None else None,
        "kl_scheduler_step_count": kl_scheduler.step_count if kl_scheduler is not None else None,
        "temp_scheduler_step_count": temp_scheduler.step_count if temp_scheduler is not None else None,
        "epoch_total_loss": epoch_total_loss,
        "epoch_num_batches": epoch_num_batches,
        "early_stopped": early_stopped,
        "timestamp": datetime.now().isoformat(),
    }

    def write_status(path):
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(status, stream, indent=2)
    _atomic_training_write(status_path, write_status)
    print(f"Training status saved to {status_path}")


def load_training_status(status_path):
    """Load training status JSON and the accompanying .pt state file."""
    if not os.path.exists(status_path):
        raise FileNotFoundError(f"Training status file not found: {status_path}")
    with open(status_path, "r", encoding="utf-8") as f:
        status = json.load(f)

    if status.get("resume_version") != 2:
        raise ValueError(
            "Cannot resume legacy training status without saved sample order. "
            "Load its adapter checkpoint as initial weights for a new run instead."
        )
    for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"), ("state_path", "state_sha256")):
        path = status.get(path_key)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Training resume file not found: {path}")
        if _training_file_sha256(path) != status.get(hash_key):
            raise ValueError(f"Cannot resume: {path_key} does not match the saved status. "
                             "The checkpoint bundle may be incomplete or overwritten.")

    state_path = status.get("state_path")
    if not state_path or not os.path.exists(state_path):
        raise FileNotFoundError(f"Training state file not found: {state_path}")

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    status["_state"] = state

    if status.get("best_val_loss") is None:
        status["best_val_loss"] = float('inf')

    return status


def train(
    model,
    train_dataloader,
    device,
    model_path,
    learning_rate,
    adapter_type,
    adapter_config,
    variational_modeling=False,
    num_epochs=1,
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.0,
    early_stopping_patience=3,
    early_stopping_min_delta=1e-4,
    val_dataloader=None,
    progress_interval=100,
    val_interval=1000,
    dev_dataloaders=None,
    kl_loss_weight=1e-2,
    kl_warmup_fraction=0.2,
    kl_schedule="linear",
    shift_regularization=False,
    layer_adapter_max_temperature=1.0,
    layer_adapter_min_temperature=0.8,
    aggregation_strategy="attention",
    history_path=None,
    resume_state=None,
    seed=42,
):
    raw_model = model.to(device)
    trainable_params = [p for p in raw_model.parameters() if p.requires_grad]

    if not trainable_params:
        print("No trainable parameters found – skipping training loop.")
        return

    train_dataloader, batch_sampler = make_resumable_loader(train_dataloader, seed)
    if val_dataloader is not None:
        val_dataloader = make_evaluation_loader(val_dataloader, seed + 1)
    if dev_dataloaders is not None:
        dev_dataloaders = {
            name: make_evaluation_loader(loader, seed + 2 + index)
            for index, (name, loader) in enumerate(sorted(dev_dataloaders.items()))
        }
    if not len(train_dataloader):
        raise ValueError("Training loader contains no batches.")

    def evaluation_signature(loader):
        if loader is None:
            return None
        return {"dataset": dataset_resume_signature(loader.dataset),
                "batch_size": loader.batch_size, "drop_last": loader.drop_last}

    # These settings define optimizer, schedule and callback behavior. Changing any
    # of them requires starting a new run from weights, rather than an exact resume.
    resume_config = {
        "seed": int(seed), "num_epochs": num_epochs, "epoch_batches": len(train_dataloader),
        "lr_schedule_version": 2,
        "learning_rate": learning_rate, "adam_betas": [adam_beta1, adam_beta2],
        "weight_decay": weight_decay, "kl_loss_weight": kl_loss_weight,
        "kl_warmup_fraction": kl_warmup_fraction, "kl_schedule": kl_schedule,
        "max_temperature": layer_adapter_max_temperature,
        "min_temperature": layer_adapter_min_temperature,
        "adapter_type": adapter_type, "variational_modeling": variational_modeling,
        "base_model": getattr(raw_model.config, "name_or_path", None),
        "shift_regularization": shift_regularization, "aggregation_strategy": aggregation_strategy,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "progress_interval": progress_interval, "val_interval": val_interval,
        "device_type": device.type,
        "trainable_parameters": [(name, tuple(param.shape), str(param.dtype))
                                 for name, param in raw_model.named_parameters() if param.requires_grad],
        "validation": evaluation_signature(val_dataloader),
        "dev": {name: evaluation_signature(loader) for name, loader in (dev_dataloaders or {}).items()},
    }

    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=weight_decay,
    )

    try:
        dataloader_len = len(train_dataloader)
        total_steps = num_epochs * dataloader_len
    except (TypeError, AttributeError):
        # IterableDataset doesn't have a length
        dataloader_len = None
        total_steps = None

    if total_steps is not None:
        warmup_steps = int(.1 * total_steps)
        scheduler = WarmUpCosineDecayScheduler(optimizer, warmup_steps, total_steps, learning_rate)
        print(f"LR scheduler set up with {warmup_steps} warmup steps and {total_steps} total steps. Max LR: {learning_rate:.2e}.")

        kl_warmup_steps = int(kl_warmup_fraction * total_steps)
        kl_scheduler = KLWeightScheduler(
            max_kl_weight=kl_loss_weight,
            warmup_steps=kl_warmup_steps,
            schedule_type=kl_schedule,
            start_kl_weight=0.0,
        )
        print(f"KL scheduler set up with {kl_warmup_steps} warmup steps. Max KL weight: {kl_loss_weight:.2e}, schedule: {kl_schedule}")
    else:
        scheduler = None
        kl_scheduler = None
        print("LR scheduler disabled (unknown dataloader length)")
        print("KL scheduler disabled (unknown dataloader length)")
    
    layer_adapter_temp_scheduler = LinearTemperatureScheduler(
        max_temperature=layer_adapter_max_temperature,
        min_temperature=layer_adapter_min_temperature,
        warmup_steps=total_steps // 10 if total_steps else 100
    )

    device_type = device.type
    use_amp = device_type == "cuda" 
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Compile the model for faster forward passes.
    # Must happen BEFORE restoring RNG states: compilation may internally advance the
    # CUDA RNG, so restoring afterwards ensures the first resumed batch sees exactly
    # the same RNG state it would have seen in an uninterrupted run.
    try:
        model = torch.compile(raw_model, backend="auto", mode="default")
        print(f"Model compiled successfully on {device_type}")
    except Exception as e:
        print(f"torch.compile() not available or failed: {e}. Training with standard model.")
        model = raw_model
    #model = raw_model

    # -------------------------
    # Resume from checkpoint if provided
    # -------------------------
    start_epoch = 0
    start_batch_index = 0
    if resume_state is not None:
        start_epoch = resume_state.get("epoch", 0)
        start_batch_index = resume_state.get("batch_index", 0)

        _state = resume_state.get("_state", {})

        if not _state.get("data_state") or _state.get("resume_config") is None:
            raise ValueError("Cannot resume legacy training state without sampler and configuration state.")
        if _state["resume_config"] != resume_config:
            changed = [key for key in resume_config if _state["resume_config"].get(key) != resume_config[key]]
            raise ValueError(f"Cannot resume: training configuration changed: {', '.join(changed)}.")
        batch_sampler.load_state_dict(_state["data_state"]["sampler"])
        if (batch_sampler.epoch != start_epoch or batch_sampler.next_batch != start_batch_index
                or resume_state.get("step") != start_epoch * dataloader_len + start_batch_index):
            raise ValueError("Cannot resume: sampler cursor and training status disagree.")

        optimizer.load_state_dict(_state["optimizer_state_dict"])
        if use_amp and "scaler_state_dict" in _state:
            scaler.load_state_dict(_state["scaler_state_dict"])

        if scheduler is not None:
            scheduler_state = _state.get("scheduler_state_dict")
            if scheduler_state is None:
                raise ValueError("Cannot resume without saved LR scheduler state.")
            scheduler.load_state_dict(scheduler_state)
            if scheduler.step_count != resume_state.get("scheduler_step_count") or scheduler.step_count > resume_state["step"]:
                raise ValueError("Cannot resume: LR scheduler and training status disagree.")
        if kl_scheduler is not None and resume_state.get("kl_scheduler_step_count") is not None:
            kl_scheduler.step_count = resume_state["kl_scheduler_step_count"]
        if resume_state.get("temp_scheduler_step_count") is not None:
            layer_adapter_temp_scheduler.step_count = resume_state["temp_scheduler_step_count"]

        rng_state = _state.get("rng_state", {})
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        if device_type == "mps":
            if "mps" not in rng_state:
                raise ValueError("Cannot resume MPS training without its saved RNG state.")
            torch.mps.set_rng_state(rng_state["mps"])

        print(f"Resuming from epoch {start_epoch}, batch {start_batch_index}, global step {resume_state.get('step', 0)}")

    # Early stopping setup
    best_val_loss = resume_state["best_val_loss"] if resume_state is not None else float('inf')
    patience_counter = resume_state["patience_counter"] if resume_state is not None else 0
    if early_stopping_patience > 0:
        if val_dataloader is not None:
            print(f"Early stopping enabled (validation-based): patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        else:
            print(f"Early stopping enabled (training-based): patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    else:
        print("Early stopping disabled")

    step = resume_state["step"] if resume_state is not None else 0
    if resume_state is not None and resume_state.get("early_stopped", False):
        print("Training already ended through early stopping; no further batches to resume.")
        return raw_model
    _resume_total_loss = resume_state.get("epoch_total_loss", 0.0) if resume_state is not None else 0.0
    _resume_num_batches = resume_state.get("epoch_num_batches", 0) if resume_state is not None else 0
    saved_progress = resume_state.get("_state", {}).get("progress_state", {}) if resume_state else {}
    for epoch in range(start_epoch, num_epochs):
        set_loader_epoch(train_dataloader, epoch)
        model.train()
        total_loss = _resume_total_loss if epoch == start_epoch else 0.0
        num_batches = _resume_num_batches if epoch == start_epoch else 0
        pending = saved_progress if epoch == start_epoch else {}
        acc_loss = list(pending.get("loss", []))
        acc_kl_loss = list(pending.get("kl_loss", []))
        acc_shift_loss = list(pending.get("shift_loss", []))
        progress = int(pending.get("progress", 0))
        current_temperature = None
        early_stop_triggered = False

        # Use progress bar without total if length is unknown
        _pbar_initial = batch_sampler.next_batch
        progress_bar = tqdm(total=dataloader_len, initial=_pbar_initial - progress, desc="Processing")
        for i, (batch, _) in enumerate(train_dataloader, start=_pbar_initial):

            batch = {k:v.to(device) for k,v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            current_temperature = None
            if adapter_type == "layer_adapter" and aggregation_strategy == "attention":
                layer_adapter_temp_scheduler.step()
                current_temperature = layer_adapter_temp_scheduler.get_temperature()
                raw_model.transformer.encoder.set_attention_temperature(current_temperature)
                adapter_config["attention_temperature"] = current_temperature

            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)

            lm_loss = outputs.loss
            loss = lm_loss

            if adapter_type == 'layer_adapter':
                shift_loss = raw_model.transformer.encoder.get_delta_loss()
                if shift_regularization:
                    loss += shift_loss
                acc_shift_loss.append(shift_loss.detach().float().item())

            # If using layer_adapter with variational modeling, add KL divergence loss
            if adapter_type == 'layer_adapter' and variational_modeling:
                kl_loss = raw_model.transformer.encoder.get_kl_loss()
                current_kl_weight = (
                    kl_scheduler.get_weight()
                    if kl_scheduler is not None
                    else kl_loss_weight
                )
                
                loss += current_kl_weight * kl_loss 
                acc_kl_loss.append(kl_loss.detach().float().item())
                
            update_learning_rate = optimizer.param_groups[0]["lr"]
            optimizer_update_performed = True
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler reduces the scale when nonfinite gradients cause a
                # skipped update. A same/higher scale means the update succeeded.
                optimizer_update_performed = scaler.get_scale() >= previous_scale
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            if scheduler is not None and optimizer_update_performed:
                scheduler.step()
            
            if kl_scheduler is not None:
                kl_scheduler.step()

            batch_loss = loss.detach().float().item()
            batch_lm_loss = lm_loss.detach().float().item()
            total_loss += batch_loss
            num_batches += 1
            acc_loss.append(batch_loss)

            # Prefetched indices do not count as consumed until this update finishes.
            batch_sampler.mark_consumed()
            step += 1
            progress += 1
            #progress_bar.update(progress if progress else 1)
    
            if (i + 1) % progress_interval == 0 or (dataloader_len is not None and i + 1 >= dataloader_len):
                progress_bar.update(progress)
                progress = 0
                
                running_loss = total_loss / num_batches
                avg_acc_loss = sum(acc_loss) / len(acc_loss) if acc_loss else 0.0
                avg_acc_kl_loss = sum(acc_kl_loss) / len(acc_kl_loss) if acc_kl_loss else 0.0
                avg_acc_shift_loss = sum(acc_shift_loss) / len(acc_shift_loss) if acc_shift_loss else 0.0
                description = (
                    f"batch loss: {avg_acc_loss:.2f}"
                    + (f", LR: {update_learning_rate:.0e}" if scheduler else "")
                    + (f", best val loss: {best_val_loss:.2f}, patience: {patience_counter}/{early_stopping_patience}")
                    + (f", KL loss: {avg_acc_kl_loss:.4f}" if acc_kl_loss else "")
                    + (f", Shift loss: {avg_acc_shift_loss:.4f}" if acc_shift_loss else "")
                    + (f", Temp: {current_temperature:.2f}" if current_temperature is not None else "")
                )
                progress_bar.set_description(description)

                log_dict = {
                    "train/total_loss": avg_acc_loss,
                    "train/running_loss": running_loss,
                    "train/lm_loss": batch_lm_loss,
                    "train/optimizer_update_performed": optimizer_update_performed,
                }

                if scheduler is not None:
                    log_dict["train/learning_rate"] = update_learning_rate

                if adapter_type == 'layer_adapter' and aggregation_strategy == "attention":
                    #log_dict["residual_scaler"] = raw_model.transformer.encoder.adapter_scale.item()

                    layer_token_attentions = raw_model.transformer.encoder.layer_attention_metrics
                    ent_layer_attention = layer_token_attentions["entropy_of_layer_attention"].cpu().item()
                    log_dict["layer_attn/entropy"] = ent_layer_attention

                    log_dict["layer_attn/temperature"] = current_temperature

                if adapter_type == 'layer_adapter' and variational_modeling:
                    mu, std = raw_model.transformer.encoder.get_variational_stats()
                    logvar = 2.0 * torch.log(std.clamp_min(1e-8))

                    log_dict["variational/mu_mean"] = mu.mean().item()
                    log_dict["variational/mu_abs_mean"] = mu.abs().mean().item()
                    log_dict["variational/mu_rms"] = mu.pow(2).mean().sqrt().item()

                    log_dict["variational/std_mean"] = std.mean().item()
                    log_dict["variational/std_std"] = std.std().item()
                    log_dict["variational/std_min"] = std.min().item()
                    log_dict["variational/std_max"] = std.max().item()

                    kl_mu_term = 0.5 * mu.pow(2)
                    kl_var_term = 0.5 * (std.pow(2) - 1.0 - logvar)

                    log_dict["variational/kl_mu_term"] = kl_mu_term.mean().item()
                    log_dict["variational/kl_var_term"] = kl_var_term.mean().item()

                    log_dict["variational/batch_kl_loss"] = avg_acc_kl_loss 

                    log_dict["variational/kl_weight"] = (
                        kl_scheduler.get_weight() if kl_scheduler is not None else kl_loss_weight
                    )
                    #log_dict["variational/batch_shift_loss"] = avg_acc_shift_loss

                #---------------------------------
                # Log to WandB and optionally to history file
                #---------------------------------
                wandb.log(log_dict, step=step)
                row = {"step": step, **log_dict}
                if history_path:
                    with open(history_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                #---------------------------------

                if adapter_type == 'layer_adapter':
                    layer_token_attentions = raw_model.transformer.encoder.layer_attention_metrics
                    avg_layer_attention = layer_token_attentions["avg_attention_to_each_layer"].detach().cpu().numpy()

                    n_layer = raw_model.config.n_layer
                    nl = len(avg_layer_attention)

                    avg_layer_attention_data = [
                        [layer_idx, avg_layer_attention[layer_idx - n_layer + nl - 1]]
                        for layer_idx in range(n_layer, n_layer - nl, -1)
                    ]

                    avg_layer_attention_table = wandb.Table(
                        data=avg_layer_attention_data,
                        columns=["layer", "attention"]
                    )

                    wandb.log({
                        "layer_attn/bar": wandb.plot.bar(
                            avg_layer_attention_table,
                            "layer",
                            "attention",
                            title="Layer Attention Distribution"
                        ), 
                        "batch_shift_loss": avg_acc_shift_loss,
                    }, step=step)


                acc_loss = []
                acc_kl_loss = []
                acc_shift_loss = []
            if (i + 1) % val_interval == 0 or (i + 1) >= dataloader_len: 
                # Validate and select the best checkpoint even when early stopping is disabled.
                if val_dataloader is not None:
                    set_loader_epoch(val_dataloader, epoch)
                    avg_val_loss, _ = validate(raw_model, val_dataloader, device, device_type, use_amp, progress_interval) # Note: the loss does not include the KL divergence component, since that is not computed during validation.  This means that when using variational modeling, the absolute value of the validation loss may not be directly comparable to the training loss, but it can still be used for early stopping based on relative improvements.
                    wandb.log({"val_loss": avg_val_loss}, step=step)
                    print(f"Val Loss: {avg_val_loss:.4f}")

                    if avg_val_loss < best_val_loss - early_stopping_min_delta:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        save_model(raw_model, adapter_type, adapter_config, model_path+'-best')
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience and early_stopping_patience > 0:
                            print("Early stopping triggered. Ending training.")
                            early_stop_triggered = True

                    wandb.log({"best_val_loss": best_val_loss, "patience_counter": patience_counter}, step=step)

                    if adapter_type == 'layer_adapter' and variational_modeling:
                        mu, std = raw_model.transformer.encoder.get_variational_stats()
                        logvar = 2.0 * torch.log(std.clamp_min(1e-8))
                        mu_rms = mu.pow(2).mean().sqrt().item()
                        std_rms = std.pow(2).mean().sqrt().item()
                        kl_mu_term = 0.5 * mu.pow(2)
                        kl_var_term = 0.5 * (std.pow(2) - 1.0 - logvar)
                        print(f"Variational stats at validation - mu_rms: {mu_rms:.4f}, std_rms: {std_rms:.4f}, kl_mu_term: {kl_mu_term.mean().item():.4f}, kl_var_term: {kl_var_term.mean().item():.4f}")
                
                if dev_dataloaders is not None:
                    for dev_name, dev_loader in dev_dataloaders.items():
                        set_loader_epoch(dev_loader, epoch)
                        avg_dev_loss, _ = validate(raw_model, dev_loader, device, device_type, use_amp, progress_interval)
                        wandb.log({f"{dev_name}_loss": avg_dev_loss}, step=step)
                        print(f"{dev_name} Loss: {avg_dev_loss:.4f}")

                print(f"save parameters - progress {progress}")
                save_model(raw_model, adapter_type, adapter_config, model_path+'-trace')
                save_training_status(
                    status_path=model_path + '.training_status.json',
                    state_path=model_path + '-trace.state.pt',
                    checkpoint_path=model_path + '-trace',
                    best_checkpoint_path=model_path + '-best',
                    epoch=epoch,
                    step=step,
                    batch_index=i + 1,
                    best_val_loss=best_val_loss,
                    patience_counter=patience_counter,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    kl_scheduler=kl_scheduler,
                    temp_scheduler=layer_adapter_temp_scheduler,
                    epoch_total_loss=total_loss,
                    epoch_num_batches=num_batches,
                    data_state={"sampler": batch_sampler.state_dict()},
                    resume_config=resume_config,
                    progress_state={"loss": acc_loss, "kl_loss": acc_kl_loss,
                                    "shift_loss": acc_shift_loss, "progress": progress},
                    early_stopped=early_stop_triggered,
                    device_type=device_type,
                )
                if early_stop_triggered:
                    break

        if early_stop_triggered:
            progress_bar.close()
            break

        progress_bar.close()
        avg_loss = total_loss / max(num_batches, 1)
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss}, step=step)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

        batch_sampler.set_epoch(epoch + 1)
        # End-of-epoch status: batch_index=0 so resume starts the next epoch cleanly;
        # epoch_total_loss/num_batches reset to 0 since the next epoch starts fresh.
        print(f"End of epoch {epoch + 1}, saving model and training status...")
        save_model(raw_model, adapter_type, adapter_config, model_path+'-trace')
        save_training_status(
            status_path=model_path + '.training_status.json',
            state_path=model_path + '-trace.state.pt',
            checkpoint_path=model_path + '-trace',
            best_checkpoint_path=model_path + '-best',
            epoch=epoch + 1,
            step=step,
            batch_index=0,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            kl_scheduler=kl_scheduler,
            temp_scheduler=layer_adapter_temp_scheduler,
            epoch_total_loss=0.0,
            epoch_num_batches=0,
            data_state={"sampler": batch_sampler.state_dict()},
            resume_config=resume_config,
            device_type=device_type,
        )

        if patience_counter >= early_stopping_patience and early_stopping_patience > 0:
            print("Early stopping triggered. Ending training.")
            break

    return raw_model

def main():
    parser = argparse.ArgumentParser(description="Train a language model with adapters using YAML configs.")

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--adapter_config", type=str, required=True)
    parser.add_argument("--training_config", type=str, required=True)
    parser.add_argument("--data_config", type=str, required=False) 
    parser.add_argument("--train_bin_path", type=str, default=None, help="Path to training token-bin file (overrides data config)")
    parser.add_argument("--val_bin_path", type=str, default=None, help="Held-out validation token-bin file (overrides data config and val_fraction)")
    parser.add_argument("--dev_bin_path", type=str, default=None, help="Path to dev token-bin file (overrides data config)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to save the trained model (overrides training config)")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment for logging purposes (overrides training config)")
    parser.add_argument("--history_path", type=str, default=None, help="Path to save training history as JSONL (overrides training config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (overrides training config)")
    parser.add_argument("--max_train_tokens", type=int, default=None,
                        help="Maximum number of tokens used for training (overrides training config)")
    parser.add_argument("--continue-training", type=str, default=None, dest="continue_training",
                        help="Path to a .training_status.json file to resume an interrupted training run")

    args = parser.parse_args()

    def load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    model_cfg = load_yaml(args.model_config)
    adapter_cfg = load_yaml(args.adapter_config)
    training_cfg = load_yaml(args.training_config)
    data_cfg = load_yaml(args.data_config) if args.data_config else {}

    # -------------------------
    # Resolve config values
    # -------------------------

    seed = training_cfg.get("seed", 42)
    if args.seed is not None:
        seed = args.seed
    set_seed(seed)

    model_name = model_cfg["model_name"]
    tokenizer_path = model_cfg.get("tokenizer_path")
    freeze_lm_heads = model_cfg.get("freeze_lm_heads", False)
    num_tailor_layers = model_cfg.get("num_tailor_layers", 0)

    adapter_type = adapter_cfg.get("adapter_type", "none")

    chkpt = training_cfg.get("chkpt", "")

    num_epochs = training_cfg.get("num_epochs", 1)
    learning_rate = training_cfg.get("learning_rate", 5e-5)
    batch_size = training_cfg.get("batch_size", 1)
    dev_batch_size = training_cfg.get("dev_batch_size", 4)
    context_size = training_cfg.get("context_size", 1024)
    num_workers = training_cfg.get("num_workers", 0)

    adam_beta1 = training_cfg.get("adam_beta1", 0.9)
    adam_beta2 = training_cfg.get("adam_beta2", 0.999)
    weight_decay = training_cfg.get("weight_decay", 0.0)

    early_stopping_patience = training_cfg.get("early_stopping_patience", 3)
    early_stopping_min_delta = training_cfg.get("early_stopping_min_delta", 1e-4)
    val_fraction = training_cfg.get("val_fraction", 0.0)
    val_interval = training_cfg.get("val_interval", 1000)
    progress_interval = training_cfg.get("progress_interval", 100)

    max_train_tokens = training_cfg.get("max_train_tokens", None)
    if args.max_train_tokens is not None:
        max_train_tokens = args.max_train_tokens

    kl_loss_weight = training_cfg.get("kl_loss_weight", 1e-2)
    kl_warmup_fraction = training_cfg.get("kl_warmup_fraction", 0.2)
    kl_schedule = training_cfg.get("kl_schedule", "linear")

    wandb_cfg = training_cfg.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", True)
    project_name = wandb_cfg.get("project_name", "llm_adapter")
    wandb_log_dir = wandb_cfg.get("log_dir", None)

    train_bin_paths = data_cfg.get("train_bin_paths", [])
    if args.train_bin_path:
        train_bin_paths = [args.train_bin_path]

    val_bin_paths = data_cfg.get("val_bin_paths", [])
    if val_bin_paths is None:
        val_bin_paths = []
    if args.val_bin_path is not None:
        val_bin_paths = [args.val_bin_path]
    if isinstance(train_bin_paths, str):
        train_bin_paths = [train_bin_paths]
    if isinstance(val_bin_paths, str):
        val_bin_paths = [val_bin_paths]

    dev_bin_paths = data_cfg.get("dev_bin_paths", [])
    if args.dev_bin_path:
        dev_bin_paths = [args.dev_bin_path]

    dev_bin_names = data_cfg.get("dev_bin_names", None)

    if len(train_bin_paths) == 0:
        parser.error("At least one train_bin_path must be provided in the data config.")

    if dev_bin_names is not None:
        if len(dev_bin_names) == 0 and len(dev_bin_paths) > 0:
            parser.error("dev_bin_names requires one name per dev_bin_path.")
        if len(dev_bin_names) != len(dev_bin_paths):
            parser.error("dev_bin_names must have the same number of values as dev_bin_paths.")
        if len(set(dev_bin_names)) != len(dev_bin_names):
            parser.error("dev_bin_names values must be unique.")

    # -------------------------
    # Adapter values
    # -------------------------

    lora_r = adapter_cfg.get("lora_r", 4)
    lora_alpha = adapter_cfg.get("lora_alpha", 16)
    lora_dropout = adapter_cfg.get("lora_dropout", 0.1)
    lora_target_modules = adapter_cfg.get("lora_target_modules", ["attn.c_proj"])

    prompt_encoder_num_virtual_tokens = adapter_cfg.get("prompt_encoder_num_virtual_tokens", 20)

    if isinstance(lora_target_modules, str):
        lora_target_modules = [m.strip() for m in lora_target_modules.split(",")]

    num_aggregation_layers = adapter_cfg.get("num_aggregation_layers")
    prefix_length = adapter_cfg.get("prefix_length", 0)

    qk_dim = adapter_cfg.get("qk_dim", 32)
    v_dim = adapter_cfg.get("v_dim", 512)
    num_attention_heads = adapter_cfg.get("num_attention_heads", 4)
    attention_temperature = adapter_cfg.get("attention_temperature", 2.0)
    min_attention_temperature = adapter_cfg.get("min_attention_temperature", 0.8)

    agg_representation_type = adapter_cfg.get("agg_representation_type", "mid_mlp")
    agg_query_source = adapter_cfg.get("agg_query_source", "final_hidden")

    variational_modeling = adapter_cfg.get("variational_modeling", False)
    aggregation_strategy = adapter_cfg.get("aggregation_strategy", "attention")
    shift_regularization = adapter_cfg.get("shift_regularization", False)

    #-------------------------
    # Model path and experiment description
    #-------------------------

    if args.experiment_name:
        experiment_description = args.experiment_name
    else:
        experiment_description = f"{model_name}_{adapter_type}"
        if adapter_type == "layer_adapter":
            experiment_description += f"_agg-{aggregation_strategy}"
            if variational_modeling:
                experiment_description += "_variational"
            if shift_regularization:
                experiment_description += "_shiftreg"
            experiment_description += f"_qk{qk_dim}_v{v_dim}_heads{num_attention_heads}"
            
        elif adapter_type == "lora":
            experiment_description += f"_r{lora_r}_alpha{lora_alpha}_drop{lora_dropout}"
        elif adapter_type == "prefix_encoder":
            experiment_description += f"_virtual_tokens{prompt_encoder_num_virtual_tokens}"
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")
        experiment_description += f"_bs{batch_size}_lr{learning_rate:.0e}_ctx{context_size}_epochs{num_epochs}_seed{seed}"
    
    model_path = args.model_path if args.model_path else f"{training_cfg['output_dir']}/{experiment_description}.pt"

    history_dir = os.path.join(training_cfg.get("history_dir", "history"), experiment_description)
    history_path = f"{history_dir}/train_history.jsonl"
    if args.history_path:
        history_path = args.history_path
    os.makedirs(os.path.dirname(history_dir), exist_ok=True)

    print(f"Experiment description: {experiment_description}")
    print(f"Model checkpoints will be saved to: {model_path}")
    print(f"Training history will be saved to: {history_path}")


    # -------------------------
    # Logging
    # -------------------------

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    resolved_config = {
        "model": model_cfg,
        "adapter": adapter_cfg,
        "training": training_cfg,
        "data": data_cfg,
        "resolved": {
            "seed": seed,
            "model_name": model_name,
            "adapter_type": adapter_type,
            "model_path": model_path,
            "train_bin_paths": train_bin_paths,
            "val_bin_paths": val_bin_paths,
            "validation_source": (
                "explicit_files" if val_bin_paths
                else "random_training_split" if val_fraction > 0 else "none"
            ),
            "val_fraction": 0.0 if val_bin_paths else val_fraction,
            "context_size": context_size,
            "dev_bin_paths": dev_bin_paths,
            "history_path": history_path,
        },
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    resolved_config_path = f"{model_path}.resolved_config.yaml"
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False)

    if use_wandb:
        wandb_dir = None
        if wandb_log_dir:
            wandb_dir = os.path.join(wandb_log_dir, experiment_description, "wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = wandb_dir
            print(f"WandB log directory set to: {wandb_dir}")

        wandb.init(
            project=project_name,
            name=f"{experiment_description}_{current_date}",
            config=resolved_config,
            dir=wandb_dir,
        )
    else:
        wandb.init(mode="disabled")

    # -------------------------
    # Device
    # -------------------------

    device = set_device()
    print(f"The active device is {device}")

    # -------------------------
    # Model setup
    # -------------------------

    resume_state = None
    if args.continue_training:
        print(f"Resuming training from status file: {args.continue_training}")
        resume_state = load_training_status(args.continue_training)

        resume_epoch = resume_state.get("epoch", 0)
        resume_batch = resume_state.get("batch_index", 0)
        if resume_state.get("early_stopped", False):
            print("Training already ended through early stopping. Load the checkpoint as initial weights for a new run to train further.")
            return
        if resume_epoch >= num_epochs:
            print(
                f"Training already complete: status shows epoch {resume_epoch} "
                f"and num_epochs={num_epochs}. Nothing to do."
            )
            return

        chkpt = resume_state["checkpoint_path"]
        print(
            f"Resuming from epoch {resume_epoch}, batch {resume_batch}, "
            f"step {resume_state.get('step', 0)}"
        )

    if chkpt:
        print(f"Loading model from {chkpt}")
        model, tokenizer, adapter_config = load_model(chkpt, freeze_lm_head=freeze_lm_heads)

    else:
        if adapter_type == "none":
            adapter_config = None

        elif adapter_type == "layer_adapter":
            adapter_config = {
                "need_weights": adapter_cfg.get("need_weights", True),
                "dropout": adapter_cfg.get("dropout", 0.1),
                "num_aggregation_layers": num_aggregation_layers,
                "prefix_length": prefix_length,
                "qk_dim": qk_dim,
                "v_dim": v_dim,
                "num_attention_heads": num_attention_heads,
                "attention_temperature": attention_temperature,
                "representation_type": agg_representation_type,
                "query_source": agg_query_source,
                "variational_modeling": variational_modeling,
                "aggregation_strategy": aggregation_strategy,
                "shift_regularization": shift_regularization,
            }

        elif adapter_type == "lora":
            adapter_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                fan_in_fan_out=True, 
            )
        
        elif adapter_type == "prefix_encoder":
            adapter_config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=prompt_encoder_num_virtual_tokens,
            )

        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")
        print(f"Adapter config: {adapter_config}")

        print(f"Using pre-tokenized binary file(s): {train_bin_paths}")

        wechsel_config = None

        model, tokenizer = setup_model(
            model_name,
            adapter_type,
            adapter_config,
            num_tailor_layers,
            wechsel_config,
            tokenizer_path,
            freeze_lm_heads,
        )

    # -------------------------
    # Dataset setup
    # -------------------------

    print(f"Creating training dataset from pre-tokenized binary file(s): {train_bin_paths}")

    if prefix_length > 0 and adapter_type == "layer_adapter":
        if context_size + prefix_length > model.config.n_positions:
            print(
                f"Context size + prefix length ({context_size + prefix_length}) "
                f"exceeds model maximum positions ({model.config.n_positions})."
            )
            context_size -= prefix_length
            print(f"Adjusted context size: {context_size}")

    train_dataset, val_dataset = build_token_bin_datasets(
        train_bin_paths=train_bin_paths,
        val_bin_paths=val_bin_paths,
        context_size=context_size,
        val_fraction=val_fraction,
        seed=seed,
        max_train_tokens=max_train_tokens,
    )
    resolved_config["resolved"].update({
        "context_size": context_size,
        "validation_source": (
            "explicit_files" if val_bin_paths
            else "random_training_split" if val_dataset is not None else "none"
        ),
        "num_training_chunks": len(train_dataset),
        "num_validation_chunks": len(val_dataset) if val_dataset is not None else 0,
    })
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False)
    if use_wandb:
        wandb.config.update({"resolved": resolved_config["resolved"]}, allow_val_change=True)

    dev_datasets = []
    for idx, dev_bin_path in enumerate(dev_bin_paths):
        dataset = TokenBinDataset(
            dev_bin_path,
            context_size=context_size,
            start_token=0,
            end_token=None,
        )

        if dev_bin_names is not None:
            dev_name = dev_bin_names[idx]
        else:
            base_name = os.path.basename(dev_bin_path)
            dev_name = base_name if base_name else f"dev_{idx + 1}"

            if any(existing_name == dev_name for existing_name, _ in dev_datasets):
                dev_name = f"{dev_name}_{idx + 1}"

        dev_datasets.append((dev_name, dataset))

    collate_fn = partial(token_bin_collate, context_size=context_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            shuffle=False,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )

    dev_dataloaders = None
    if dev_datasets:
        print(f"Created {len(dev_datasets)} separate dev dataset(s).")

        dev_dataloaders = {
            dev_name: DataLoader(
                dev_dataset,
                batch_size=dev_batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                shuffle=False,
                pin_memory=device.type == "cuda",
            )
            for dev_name, dev_dataset in dev_datasets
        }

    # -------------------------
    # Train
    # -------------------------

    model = train(
        model=model,
        train_dataloader=train_dataloader,
        device=device,
        model_path=model_path,
        learning_rate=learning_rate,
        adapter_type=adapter_type,
        adapter_config=adapter_config,
        variational_modeling=variational_modeling,
        num_epochs=num_epochs,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        val_dataloader=val_dataloader,
        progress_interval=progress_interval,
        val_interval=val_interval,
        dev_dataloaders=dev_dataloaders,
        kl_loss_weight=kl_loss_weight,
        kl_warmup_fraction=kl_warmup_fraction,
        kl_schedule=kl_schedule,
        shift_regularization=shift_regularization,
        layer_adapter_max_temperature=attention_temperature,
        layer_adapter_min_temperature=min_attention_temperature,
        aggregation_strategy=aggregation_strategy,
        history_path=history_path,
        resume_state=resume_state,
        seed=seed,
    )

    save_model(model, adapter_type, adapter_config, model_path)

    wandb.save(resolved_config_path)
    wandb.finish()

if __name__ == '__main__':
    main()
