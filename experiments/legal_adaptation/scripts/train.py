import json
import numpy as np
import torch
import os
import argparse
from functools import partial
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import math
import wandb
from datetime import datetime
from llm_adapter import setup_model, save_model, load_model
from peft import LoraConfig, TaskType
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

        if max_tokens is not None:
            self.data = np.memmap(bin_path, dtype=dtype, mode="r", shape=(max_tokens,))
        else:
            self.data = np.memmap(bin_path, dtype=dtype, mode="r")

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
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.step_count = 0

    def step(self, step_count:int=None):
        """Update the learning rate based on the current step."""
        self.step_count += step_count or 1

        if self.step_count < self.warmup_steps:
            # Linear warm-up phase
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine decay phase
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)  # Clamp to [0, 1]
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

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
    
def validate(model, val_dataloader, device, device_type, use_amp):
    """
    Perform validation on a dataset and compute average loss.

    Args:
        model: PyTorch model to validate
        val_dataloader: DataLoader for validation data
        device: torch.device to run on
        device_type: device type ('cuda', 'mps', 'cpu')
        use_amp: whether to use automatic mixed precision

    Returns:
        avg_loss: average loss across all validation batches
        num_batches: number of batches processed
    """
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for val_batch, _ in tqdm(val_dataloader, desc="Validating"):
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            #with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            outputs = model(**val_batch)
            loss = outputs.loss
            val_loss += loss.detach().float().item()
            num_batches += 1

    avg_loss = val_loss / max(num_batches, 1)
    return avg_loss, num_batches

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
    history_path=None
):
    raw_model = model.to(device)
    trainable_params = [p for p in raw_model.parameters() if p.requires_grad]

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

    # Compile the model for faster forward passes
    try:
        model = torch.compile(raw_model, backend="auto", mode="default")
        print(f"Model compiled successfully on {device_type}")
    except Exception as e:
        print(f"torch.compile() not available or failed: {e}. Training with standard model.")
        model = raw_model
    #model = raw_model

    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    if early_stopping_patience > 0:
        if val_dataloader is not None:
            print(f"Early stopping enabled (validation-based): patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        else:
            print(f"Early stopping enabled (training-based): patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    else:
        print("Early stopping disabled")

    step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        acc_loss = []
        acc_kl_loss = []
        acc_shift_loss = []
        progress = 0
        current_temperature = None

        # Use progress bar without total if length is unknown
        progress_bar = tqdm(total=dataloader_len, desc="Processing")
        for i, (batch, _) in enumerate(train_dataloader):
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
                
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
            
            if kl_scheduler is not None:
                kl_scheduler.step()

            batch_loss = loss.detach().float().item()
            batch_lm_loss = lm_loss.detach().float().item()
            total_loss += batch_loss
            num_batches += 1
            acc_loss.append(batch_loss)

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
                    + (f", LR: {scheduler.get_lr():.0e}" if scheduler else "")
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
                }

                if scheduler is not None:
                    log_dict["train/learning_rate"] = scheduler.get_lr()

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
            if (i + 1) % val_interval == 0: 
                # Early stopping check (training-based, only if no validation set)
                if val_dataloader is not None and early_stopping_patience > 0:
                    avg_val_loss, _ = validate(raw_model, val_dataloader, device, device_type, use_amp) # Note: the loss does not include the KL divergence component, since that is not computed during validation.  This means that when using variational modeling, the absolute value of the validation loss may not be directly comparable to the training loss, but it can still be used for early stopping based on relative improvements.
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
                            break

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
                        avg_dev_loss, _ = validate(raw_model, dev_loader, device, device_type, use_amp)
                        wandb.log({f"{dev_name}_loss": avg_dev_loss}, step=step)
                        print(f"{dev_name} Loss: {avg_dev_loss:.4f}")

                print(f"save parameters - progress {progress}")
                save_model(raw_model, adapter_type, adapter_config, model_path+'-trace')

        progress_bar.close()

        avg_loss = total_loss / max(num_batches, 1)
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss}, step=step)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

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
    parser.add_argument("--dev_bin_path", type=str, default=None, help="Path to dev token-bin file (overrides data config)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to save the trained model (overrides training config)")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment for logging purposes (overrides training config)")
    parser.add_argument("--history_path", type=str, default=None, help="Path to save training history as JSONL (overrides training config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (overrides training config)")

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

    if chkpt:
        print(f"Loading model from {chkpt}")
        model, tokenizer, adapter_config = load_model(chkpt)

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
            }

        elif adapter_type == "lora":
            adapter_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
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

    train_datasets = []
    val_datasets = []
    dev_datasets = []

    for train_bin_path in train_bin_paths:
        dataset = TokenBinDataset(
            train_bin_path,
            context_size=context_size,
            start_token=0,
            end_token=None,
        )

        if val_fraction > 0:
            total_size = len(dataset)
            val_size = int(total_size * val_fraction)
            train_size = total_size - val_size

            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed),
            )

            print(
                f"Split {train_bin_path} into train ({train_size} chunks) "
                f"and validation ({val_size} chunks)."
            )

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        else:
            train_datasets.append(dataset)

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

    train_dataset = (
        MixedDataset(train_datasets)
        if len(train_datasets) > 1
        else train_datasets[0]
    )

    val_dataset = (
        MixedDataset(val_datasets)
        if len(val_datasets) > 1
        else val_datasets[0]
        if val_datasets
        else None
    )

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
        history_path=history_path
    )

    save_model(model, adapter_type, adapter_config, model_path)

    wandb.save(resolved_config_path)
    wandb.finish()

if __name__ == '__main__':
    main()