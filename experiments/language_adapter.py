import json
import numpy as np
import torch
import os
import argparse
from functools import partial
from torch.utils.data import Dataset, random_split, ConcatDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import math
import wandb
from datetime import datetime
from llm_adapter import setup_model, save_model, load_model
from peft import LoraConfig, TaskType
import random


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
    """
    Read pre-tokenized token IDs from a binary file produced by
    data/prepare_dataset.py and serve fixed-length context windows.

    The matching .json metadata sidecar (same path, .json extension) is read
    automatically to determine the correct dtype.  Falls back to uint16 when
    the sidecar is absent.
    
    Supports indexed access so it can be used with ConcatDataset/random_split.
    """

    def __init__(self, bin_path: str, context_size: int = 1024, max_tokens: int = None, start_token: int = 0, end_token: int = None):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Token-bin file not found: {bin_path}")

        meta_path = os.path.splitext(bin_path)[0] + ".json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            dtype = np.dtype(meta["dtype"])
            print(f"TokenBinDataset: loaded metadata from {meta_path}")
            print(f"  tokenizer : {meta.get('tokenizer_name', 'unknown')}")
            print(f"  EOS token : {meta.get('eos_token', 'unknown')} (id={meta.get('eos_token_id', '?')})")
        else:
            dtype = np.uint16
            print(f"TokenBinDataset: no metadata sidecar found, defaulting to uint16")

        self.context_size = context_size
        self.start_token = start_token

        if max_tokens is not None:
            self.data = np.memmap(bin_path, dtype=dtype, mode='r', shape=(max_tokens,))
            print(f"TokenBinDataset: loading with max_tokens={max_tokens}")
        else:
            self.data = np.memmap(bin_path, dtype=dtype, mode='r')

        # Support slicing the dataset
        if end_token is None:
            end_token = len(self.data)
        self.end_token = min(end_token, len(self.data))

        # Compute expected number of chunks for logging
        self.num_tokens = self.end_token - self.start_token
        self.num_chunks = int(np.ceil(self.num_tokens / context_size))

        print(f"  total tokens : {self.num_tokens:,}")
        print(f"  context size : {context_size}")
        print(f"  chunks       : {self.num_chunks:,}")
        if start_token > 0 or end_token < len(np.memmap(bin_path, dtype=dtype, mode='r')):
            print(f"  token range  : {start_token:,} - {self.end_token:,}")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = self.start_token + idx * self.context_size
        end = min(start + self.context_size, self.end_token)
        chunk = self.data[start:end].astype(np.int64)
        input_ids = torch.from_numpy(np.array(chunk))  # copy out of memmap
        return {"input_ids": input_ids, "progress": 1}

def token_bin_collate(batch, context_size):
    input_ids = torch.stack([x['input_ids'] for x in batch if len(x['input_ids']) == context_size])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    progress = None
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}, progress

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
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

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

def train(model, 
          train_dataloader, 
          device, model_path, 
          num_epochs=1, 
          adam_beta1=0.9, 
          adam_beta2=0.999, 
          weight_decay=0.0, 
          early_stopping_patience=3, 
          early_stopping_min_delta=1e-4, 
          val_dataloader=None,
          progress_interval=100,
          val_interval=1000,
          ):
    raw_model = model.to(device)
    trainable_params = [p for p in raw_model.parameters() if p.requires_grad]

    optimizer = Adam(
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
    else:
        scheduler = None
        print("LR scheduler disabled (unknown dataloader length)")

    device_type = device.type
    use_amp = False # device_type == "cuda" # AMP can cause instability for some models, so we disable it for now.  Can be re-enabled if desired.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Compile the model for faster forward passes
    #try:
    #    model = torch.compile(raw_model, backend="auto", mode="default")
    #    print(f"Model compiled successfully on {device_type}")
    #except Exception as e:
    #    print(f"torch.compile() not available or failed: {e}. Training with standard model.")
    #    model = raw_model
    model = raw_model

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
    kl_loss_weight = 1e-2  # Weight for KL divergence loss when using variational modeling in layer_adapter
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        acc_loss = []
        acc_kl_loss = []

        # Use progress bar without total if length is unknown
        progress_bar = tqdm(total=dataloader_len, desc="Processing")
        for i, (batch, progress) in enumerate(train_dataloader):
            batch = {k:v.to(device) for k,v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            #with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            outputs = model(**batch)
            loss = outputs.loss

            # If using layer_adapter with variational modeling, add KL divergence loss
            if adapter_type == 'layer_adapter' and variational_modeling:
                kl_loss = raw_model.transformer.encoder.get_kl_loss()
                loss += kl_loss_weight * kl_loss
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

            batch_loss = loss.detach().float().item()
            total_loss += batch_loss
            num_batches += 1
            acc_loss.append(batch_loss)

            step += 1
            progress_bar.update(progress if progress else 1)
    
            if (i + 1) % progress_interval == 0:
                running_loss = total_loss / num_batches
                avg_acc_loss = sum(acc_loss) / len(acc_loss) if acc_loss else 0.0
                avg_acc_kl_loss = sum(acc_kl_loss) / len(acc_kl_loss) if acc_kl_loss else 0.0

                description = (
                    f"running loss: {running_loss:.2f}, batch loss: {avg_acc_loss:.2f}"
                    + (f", LR: {scheduler.get_lr():.0e}" if scheduler else "")
                    + (f", best val loss: {best_val_loss:.2f}, patience: {patience_counter}/{early_stopping_patience}")
                    + (f", KL loss: {avg_acc_kl_loss:.4f}" if acc_kl_loss else "")
                )
                progress_bar.set_description(description)

                log_dict = {
                    "batch_loss": avg_acc_loss,
                    "running_loss": running_loss,
                }

                if scheduler is not None:
                    log_dict["learning_rate"] = scheduler.get_lr()

                if adapter_type == 'layer_adapter':
                    #log_dict["residual_scaler"] = raw_model.transformer.encoder.adapter_scale.item()

                    layer_token_attentions = raw_model.transformer.encoder.layer_attention_metrics
                    ent_layer_attention = layer_token_attentions["entropy_of_layer_attention"].cpu().item()
                    log_dict["layer_attn/entropy"] = ent_layer_attention

                    if variational_modeling:
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

                wandb.log(log_dict, step=step)

                if adapter_type == 'layer_adapter':
                    layer_token_attentions = raw_model.transformer.encoder.layer_attention_metrics
                    avg_layer_attention = layer_token_attentions["avg_attention_to_each_layer"].detach().cpu().numpy()

                    n_layer = raw_model.config.n_layer
                    nl = len(avg_layer_attention)

                    avg_layer_attention_data = [
                        [f"layer_{layer_idx}", avg_layer_attention[layer_idx - n_layer + nl - 1]]
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
                        )
                    }, step=step)

                acc_loss = []
                acc_kl_loss = []

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
                        if patience_counter >= early_stopping_patience:
                            print("Early stopping triggered. Ending training.")
                            break

                    wandb.log({"best_val_loss": best_val_loss, "patience_counter": patience_counter}, step=step)

                    if adapter_type == 'layer_adapter' and variational_modeling:
                        mu, std = raw_model.transformer.encoder.get_variational_stats()
                        mu_rms = mu.pow(2).mean().sqrt().item()
                        std_rms = std.pow(2).mean().sqrt().item()
                        print(f"Variational stats at validation - mu_rms: {mu_rms:.4f}, std_rms: {std_rms:.4f}")

                print(f"save parameters - progress {progress}")
                save_model(raw_model, adapter_type, adapter_config, model_path+'-trace')

        progress_bar.close()

        avg_loss = total_loss / max(num_batches, 1)
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss}, step=step)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered. Ending training.")
            break

    return raw_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a language model with adapters')
    parser.add_argument('--model_name', type=str, required=True, help='Name or path of the base model (e.g., gpt2)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Path to save/load the tokenizer (optional)')
    parser.add_argument('--num_tailor_layers', type=int, required=True, help='Number of tailor layers to add')
    parser.add_argument('--adapter_type', type=str, required=True, choices=['none', 'layer_adapter', 'lora'],
                        help='Type of adapter to use')
    parser.add_argument('--freeze_lm_heads', action='store_true', help='Whether to freeze the language model heads')
    parser.add_argument('--chkpt', type=str, default='',
                        help='Path to checkpoint to resume from (optional)')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--proj_name', type=str, required=True, help='Weights & Biases project name')
    parser.add_argument('--experiment_description', type=str, required=True, help='Experiment description for W&B')

    # Data source options: only pre-tokenized binary inputs are supported.
    parser.add_argument('--token_bin', type=str, nargs='+', required=True,
                        help='Path(s) to pre-tokenized .bin file(s) produced by data/prepare_dataset.py')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')

    # Optional parameters
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--context_size', type=int, default=1024, help='Context size (default: 1024)')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading (default: 0)')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1 (default: 0.9)')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2 (default: 0.999)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 penalty) (default: 0.0)')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Early stopping patience (epochs without improvement) (default: 3, 0 to disable)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4, help='Minimum loss improvement for early stopping (default: 1e-4)')
    parser.add_argument('--val_fraction', type=float, default=0.0, help='Fraction of training data to use for validation (default: 0.0 = no validation, early stopping disabled)')
    parser.add_argument('--val_interval', type=int, default=1000, help='Number of training steps between validations (default: 1000)')
    parser.add_argument('--progress_interval', type=int, default=100, help='Number of training steps between progress updates (default: 100)')

    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=4, help='LoRA rank (r parameter) (default: 4)')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha (scaling factor) (default: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout (default: 0.1)')
    parser.add_argument('--lora_target_modules', type=str, default='attn.c_proj', help='Comma-separated target modules for LoRA (default: attn.c_proj)')

    # layer_adapter parameters
    parser.add_argument('--num_aggregation_layers', type=int, default=None, help='Number of layers to aggregate in layer_adapter (default: None = all layers)')
    parser.add_argument('--prefix_length', type=int, default=0, help='Length of prefix embeddings for layer_adapter (default: 0 = no prefix)')
    parser.add_argument('--adjust_pre_mlps', action='store_true', help='Whether to adjust pre-MLP activations in layer_adapter (default: False)')
    parser.add_argument('--qk_dim', type=int, default=32, help='Dimension of Q and K in layer_adapter (default: 32)')
    parser.add_argument('--v_dim', type=int, default=None, help='Dimension of V in layer_adapter (default: None = same as Q/K)')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='Number of attention heads in layer_adapter (default: 4)')
    parser.add_argument('--attention_temperature', type=float, default=2.0, help='Temperature for attention in layer_adapter (default: 2.0)')
    parser.add_argument('--v_rank', type=int, default=0, help='Rank of V projection in layer_adapter (default: 0 = no low-rank projection)')
    parser.add_argument('--agg_representation_type', type=str, default='mid_mlp', choices=['pre_mlp', 'mid_mlp', 'post_mlp'], help='Type of representation to aggregate in layer_adapter (default: mid_mlp)')
    parser.add_argument('--agg_query_source', type=str, default='final_hidden', choices=["final_hidden", "top_repr"], help='Source of query for aggregation in layer_adapter (default: final_hidden)')
    parser.add_argument('--variational_modeling', action='store_true', help='Whether to use variational modeling in layer_adapter (default: False)')


    args = parser.parse_args()

    token_bin_paths = args.token_bin

    # Set random seed for reproducibility
    set_seed(args.seed)

    model_name = args.model_name
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    num_tailor_layers = args.num_tailor_layers
    adapter_type = args.adapter_type
    chkpt = args.chkpt
    num_epochs = args.num_epochs
    project_name = args.proj_name
    experiment_description = args.experiment_description
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    context_size = args.context_size
    num_workers = args.num_workers
    adam_beta1 = args.adam_beta1
    adam_beta2 = args.adam_beta2
    weight_decay = args.weight_decay
    early_stopping_patience = args.early_stopping_patience
    early_stopping_min_delta = args.early_stopping_min_delta
    val_fraction = args.val_fraction
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
    num_aggregation_layers = args.num_aggregation_layers
    prefix_length = args.prefix_length
    progress_interval = args.progress_interval
    val_interval = args.val_interval
    freeze_lm_heads = args.freeze_lm_heads
    pre_mlp_adjustment = args.adjust_pre_mlps
    qk_dim = args.qk_dim
    v_dim = args.v_dim
    num_attention_heads = args.num_attention_heads
    attention_temperature = args.attention_temperature
    v_rank = args.v_rank
    agg_representation_type = args.agg_representation_type
    agg_query_source = args.agg_query_source
    variational_modeling = args.variational_modeling

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    wandb.init(
    project=project_name,
    name=f"{experiment_description}_{current_date}",
    config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "context_size": context_size,
        "num_workers": num_workers,
        "model_name": model_name,
        "num_tailor_layers": num_tailor_layers,
        "adapter_type": adapter_type,
        "ckeck_point": chkpt,
        "num_epochs": num_epochs,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "weight_decay": weight_decay,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
        "num_aggregation_layers": num_aggregation_layers,
        "progress_interval": progress_interval,
        "val_interval": val_interval,
        "freeze_lm_heads": freeze_lm_heads,
        "adjust_pre_mlps": pre_mlp_adjustment,
        "qk_dim": qk_dim,
        "v_dim": v_dim,
        "num_attention_heads": num_attention_heads,
        "attention_temperature": attention_temperature,
        "v_rank": v_rank,
        "agg_representation_type": agg_representation_type,
        "agg_query_source": agg_query_source,
        "variational_modeling": variational_modeling,
    })

    device = set_device()
    print(f"The active device is {device}")

    if chkpt: # and os.path.exists(chkpt):
        print(f"loading model from {chkpt}")
        model, tokenizer, adapter_config = load_model(chkpt)
    else:
        if adapter_type == 'none':
            adapter_config = None
        elif adapter_type == 'layer_adapter':
            adapter_config = {
                'need_weights': True,
                'dropout': 0.1,
                'num_aggregation_layers': num_aggregation_layers,
                'prefix_length': prefix_length,
                'qk_dim': qk_dim,
                'num_attention_heads': num_attention_heads,
                'attention_temperature': attention_temperature,
                'representation_type': agg_representation_type,
                'query_source': agg_query_source,
                'variational_modeling': variational_modeling,
            }
        elif adapter_type == 'lora':
            adapter_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
            )

        print(f"Using pre-tokenized binary file(s): {token_bin_paths}")
        wechsel_config = None
        model, tokenizer = setup_model(model_name, adapter_type, adapter_config, num_tailor_layers, wechsel_config, tokenizer_path, freeze_lm_heads)

    # Create training dataset from pre-tokenized binary file(s)
    print(f"Creating training dataset from pre-tokenized binary file(s): {token_bin_paths}")
    if prefix_length > 0 and adapter_type == 'layer_adapter':
        if context_size + prefix_length > model.config.n_positions:
            print(f"Context size + prefix length ({context_size + prefix_length}) exceeds model's maximum position embeddings ({model.config.n_positions})")
            print(f"Adjusting context size for prefix embeddings: original {context_size}, prefix {prefix_length}")
            context_size -= prefix_length
            print(f"New context size: {context_size}")

    token_bin_datasets = [
        TokenBinDataset(
            bin_path=bin_path,
            context_size=context_size,
        )
        for bin_path in token_bin_paths
    ]
    full_dataset = ConcatDataset(token_bin_datasets)

    # Split into train and validation if requested
    if val_fraction > 0:
        total_size = len(full_dataset)
        val_size = int(total_size * val_fraction)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"Split into train ({train_size} chunks) and validation ({val_size} chunks)")
        val_collate_fn = partial(token_bin_collate, context_size=context_size)
    else:
        train_dataset = full_dataset
        val_dataset = None
        val_collate_fn = None

    collate_fn = partial(token_bin_collate, context_size=context_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        shuffle=True, # shuffling is fine since each item is a chunk of tokens and does not have an inherent order relative to other items
        pin_memory=True if device.type in ['cuda', 'mps'] else False
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size,
            collate_fn=val_collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            shuffle=False, # no need to shuffle validation data
            pin_memory=True if device.type in ['cuda', 'mps'] else False
        )

    model = train(model, 
                  train_dataloader, 
                  device, 
                  model_path, 
                  num_epochs, 
                  adam_beta1, 
                  adam_beta2, 
                  weight_decay, 
                  early_stopping_patience, 
                  early_stopping_min_delta, 
                  val_dataloader, 
                  progress_interval,
                  val_interval)

    save_model(model, adapter_type, adapter_config, model_path)

    wandb.save(f"{model_path}.wandb")
    wandb.finish()
