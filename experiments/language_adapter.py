import pdb

import json
import numpy as np
import torch
import os
import sys
import argparse
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from tqdm import tqdm
import math
import wandb
from datetime import datetime
from llm_adapter import setup_model, save_model
from peft import LoraConfig, TaskType

def set_device(device_name=None):
    if device_name:
        return torch.device(device_name)
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return torch.device(device)

class ChunkIterableDataset(IterableDataset):
    def __init__(self, corpus_path=None, tokenizer=None, chunk_size:int = 1024, context_size:int = 1024, dataset=None, text_column:str = "text"):
        """
        Initialize ChunkIterableDataset.

        Args:
            corpus_path: Path to text file (optional if dataset is provided)
            tokenizer: Tokenizer to use
            chunk_size: Size of chunks to read at once
            context_size: Size of context window for model
            dataset: HuggingFace dataset object (optional if corpus_path is provided)
            text_column: Column name in dataset containing text (default: "text")
        """
        if corpus_path is None and dataset is None:
            raise ValueError("Either corpus_path or dataset must be provided")

        if corpus_path is not None and dataset is not None:
            raise ValueError("Only one of corpus_path or dataset should be provided, not both")

        self.corpus_path = corpus_path
        self.dataset = dataset
        self.text_column = text_column
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.tokenizer = tokenizer
        self.tell = 0

        # Validate and get corpus size
        if corpus_path is not None:
            if not os.path.exists(corpus_path):
                raise FileNotFoundError(f"File {corpus_path} not found!")
            self.corpus_size = os.path.getsize(self.corpus_path)
            print(f"Corpus size: {self.corpus_size} bytes")
        else:
            # For datasets, try to get the size if available
            try:
                self.corpus_size = len(dataset)
                print(f"Dataset size: {self.corpus_size} samples")
            except (TypeError, AttributeError):
                # IterableDataset doesn't have len(), estimate will be done during iteration
                self.corpus_size = None
                print(f"Dataset size: Unknown (IterableDataset)")

    def __iter__(self):
        if self.corpus_path is not None:
            return self._chunk_generator_file()
        else:
            return self._chunk_generator_dataset()

    def _chunk_generator_file(self):
        """Generator that reads from a file."""
        self.tell = 0
        buffer = []
        with open(self.corpus_path, 'r', encoding='utf-8') as file:
            while True:
                chunk = file.read(self.chunk_size)
                if not chunk:
                    break

                self.tell += len(chunk.encode("utf-8"))
                buffer += self.tokenizer(chunk, truncation=False, add_special_tokens=False).input_ids

                while len(buffer) > self.context_size:
                    yield {
                        'input_ids': torch.tensor(buffer[:self.context_size]),
                        'progress': len(buffer[:self.context_size])  # Progress in bytes for file-based dataset,
                    }
                    buffer = buffer[self.context_size:]
                    self.tell = 0

        if buffer:
            yield {
                'input_ids': torch.tensor(buffer),
                'progress': len(buffer)  # Final progress for remaining buffer
            }

    def _chunk_generator_dataset(self):
        """Generator that reads from a HuggingFace dataset."""
        buffer = []
        for idx, sample in enumerate(self.dataset):
            text = sample[self.text_column]
            buffer += self.tokenizer(text, truncation=False, add_special_tokens=False).input_ids

            while len(buffer) > self.context_size:
                yield {
                    'input_ids': torch.tensor(buffer[:self.context_size]),
                    'progress': idx,
                }
                buffer = buffer[self.context_size:]

        if buffer:
            yield {
                'input_ids': torch.tensor(buffer),
                'progress': idx if idx else 0  # Use idx if available, else 0
            }

    def chunk_generator(self):
        """Alias for __iter__ for backward compatibility."""
        return self.__iter__()

    def __len__(self):
        """Return corpus size, or raise error if unknown (for IterableDataset)."""
        if self.corpus_size is None:
            raise TypeError("object of type 'IterableDataset' has no len()")
        return self.corpus_size


class TokenBinDataset(Dataset):
    """
    Read pre-tokenized token IDs from a binary file produced by
    data/prepare_dataset.py and serve fixed-length context windows.

    The matching .json metadata sidecar (same path, .json extension) is read
    automatically to determine the correct dtype.  Falls back to uint16 when
    the sidecar is absent.
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

        self.num_tokens = self.end_token - self.start_token
        self.num_chunks = int(np.ceil(self.num_tokens / context_size))

        print(f"  total tokens : {self.num_tokens:,}")
        print(f"  context size : {context_size}")
        print(f"  chunks       : {self.num_chunks:,}")
        if start_token > 0 or end_token < len(self.data):
            print(f"  token range  : {start_token:,} - {self.end_token:,}")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = self.start_token + idx * self.context_size
        end = min(start + self.context_size, self.end_token)
        chunk = self.data[start : end].astype(np.int64)
        input_ids = torch.from_numpy(np.array(chunk))  # copy out of memmap
        return {"input_ids": input_ids, "progress": 1}

    def collate(self, batch):
        input_ids = torch.stack([x['input_ids'] for x in batch if len(x['input_ids']) == self.context_size]) # Only include full context windows
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        progress = None
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':input_ids}, progress

def collate(batch):
    input_ids = [x['input_ids'] for x in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PADDING_VALUE)
    attention_mask = (input_ids != PADDING_VALUE)
    progress = None
    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':input_ids}, progress

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
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                outputs = model(**val_batch)
                loss = outputs.loss
            val_loss += loss.detach().float().item()
            num_batches += 1

    avg_loss = val_loss / max(num_batches, 1)
    return avg_loss, num_batches

def train(model, train_dataloader, device, model_path, num_epochs=1, adam_beta1=0.9, adam_beta2=0.999, weight_decay=0.0, early_stopping_patience=3, early_stopping_min_delta=1e-4, val_dataloader=None):
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
        print(f"LR scheduler set up with {warmup_steps} warmup steps and {total_steps} total steps")
    else:
        scheduler = None
        print("LR scheduler disabled (unknown dataloader length)")

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

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        acc_loss = []

        # Use progress bar without total if length is unknown
        progress_bar = tqdm(total=dataloader_len, desc="Processing")
        for i, (batch, progress) in enumerate(train_dataloader):
            batch = {k:v.to(device) for k,v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss

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

            progress_bar.update(progress if progress else 1)
            if i % 10 == 0:
                running_loss = total_loss / num_batches
                avg_acc_loss = sum(acc_loss) / len(acc_loss) if acc_loss else 0.0
                progress_bar.set_description((f"running loss: {running_loss:.2f}, batch loss: {avg_acc_loss:.2f}") +
                                            (f", LR: {scheduler.get_lr():.0e}" if scheduler else "") +
                                            (f", best val loss: {best_val_loss:.2f}, patience: {patience_counter}/{early_stopping_patience}"))

                wandb.log({"batch_loss": avg_acc_loss, "running_loss": running_loss})
                if scheduler is not None:
                    wandb.log({"learning_rate": scheduler.get_lr()})
                
                if adapter_type == 'layer_adapter':
                        residual_scaler = model.transformer.encoder.adapter_scale.item()
                        wandb.log({"residual_scaler": residual_scaler})
                
                acc_loss = []

            if i > 0 and i % 1000 == 0: 
                # Early stopping check (training-based, only if no validation set)
                if val_dataloader is not None and early_stopping_patience > 0:
                    avg_val_loss, _ = validate(model, val_dataloader, device, device_type, use_amp)
                    wandb.log({"val_loss": avg_val_loss})
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

                    wandb.log({"best_val_loss": best_val_loss, "patience_counter": patience_counter})
                
                print(f"save parameters - progress {progress}")
                save_model(raw_model, adapter_type, adapter_config, model_path+'-trace')

        progress_bar.close()

        avg_loss = total_loss / max(num_batches, 1)
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})
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
    parser.add_argument('--chkpt', type=str, default='',
                        help='Path to checkpoint to resume from (optional)')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--proj_name', type=str, required=True, help='Weights & Biases project name')
    parser.add_argument('--experiment_description', type=str, required=True, help='Experiment description for W&B')

    # WECHSEL parameters – only required when tokenizer training is needed
    # (i.e. not using --token_bin and not resuming from --chkpt)
    parser.add_argument('--src_language', type=str, default=None, help='Source language code for WECHSEL')
    parser.add_argument('--tgt_language', type=str, default=None, help='Target language code for WECHSEL')
    parser.add_argument('--wechsel_dictionary', type=str, default=None, help='Bilingual dictionary name or path for WECHSEL')

    # Data source options
    parser.add_argument('--train_data', type=str, default=None,
                        help='Path to training data file (use this OR --dataset_name, not both)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='HuggingFace dataset name (use this OR --train_data, not both)')
    parser.add_argument('--dataset_split', type=str, default='train',
                        help='Dataset split to use (default: train)')
    parser.add_argument('--dataset_config', type=str, default=None,
                        help='Dataset configuration (e.g., language code for multilingual datasets)')
    parser.add_argument('--text_column', type=str, default='text',
                        help='Column name containing text in dataset (default: text)')
    parser.add_argument('--language_filter', type=str, default=None,
                        help='Language code to filter samples (e.g., "da" for Danish; requires language column)')
    parser.add_argument('--token_bin', type=str, default=None,
                        help='Path to a pre-tokenized .bin file produced by data/prepare_dataset.py '
                             '(use this OR --train_data / --dataset_name, not in combination)')

    # Optional parameters
    parser.add_argument('--chunk_size', type=int, default=4*1024, help='Chunk size for data loading (default: 4096)')
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

    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=4, help='LoRA rank (r parameter) (default: 4)')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha (scaling factor) (default: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout (default: 0.1)')
    parser.add_argument('--lora_target_modules', type=str, default='attn.c_proj', help='Comma-separated target modules for LoRA (default: attn.c_proj)')


    args = parser.parse_args()

    # Validate data source
    sources_provided = sum(x is not None for x in [args.train_data, args.dataset_name, args.token_bin])
    if sources_provided == 0:
        parser.error("One of --train_data, --dataset_name, or --token_bin must be provided")
    if sources_provided > 1:
        parser.error("Only one of --train_data, --dataset_name, or --token_bin should be provided, not multiple")

    # WECHSEL args are only needed when the tokenizer must be trained
    # (no pre-tokenized binary and not resuming from a checkpoint)
    wechsel_needed = not args.chkpt and args.token_bin is None
    if wechsel_needed:
        missing = [flag for flag, val in [
            ('--src_language',      args.src_language),
            ('--tgt_language',      args.tgt_language),
            ('--wechsel_dictionary', args.wechsel_dictionary),
        ] if not val]
        if missing:
            parser.error(f"WECHSEL tokenizer training requires: {', '.join(missing)}")

    model_name = args.model_name
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    num_tailor_layers = args.num_tailor_layers
    adapter_type = args.adapter_type
    chkpt = args.chkpt
    num_epochs = args.num_epochs
    project_name = args.proj_name
    experiment_description = args.experiment_description
    src_language = args.src_language
    tgt_language = args.tgt_language
    wechsel_dictionary = args.wechsel_dictionary
    chunk_size = args.chunk_size
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
    })

    device = set_device()
    print(f"The active device is {device}")

    if chkpt: # and os.path.exists(chkpt):
        print(f"loading model from {chkpt}")
        model, tokenizer, adapter_config = load_learnable_params(chkpt)
        hf_dataset = None  # No dataset needed when loading from checkpoint
    else:
        if adapter_type == 'none':
            adapter_config = None
        elif adapter_type == 'layer_adapter':
            adapter_config = {
                'need_weights': False,
                'dropout': 0.1,
            }
        elif adapter_type == 'lora':
            adapter_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout
            )

        # Load data source first (needed for tokenizer training)
        if args.train_data is not None:
            print(f"Preparing data from file: {args.train_data}")
            hf_dataset = None
        elif args.token_bin is not None:
            print(f"Using pre-tokenized binary: {args.token_bin}")
            hf_dataset = None
        else:
            print(f"Loading dataset: {args.dataset_name}")
            if args.dataset_config:
                print(f"  Config: {args.dataset_config}")
            print(f"  Split: {args.dataset_split}")
            if args.language_filter:
                print(f"  Language filter: {args.language_filter}")

            from datasets import load_dataset
            if args.dataset_config:
                hf_dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split, streaming=True)
            else:
                hf_dataset = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True)

            # Apply language filter if specified
            if args.language_filter:
                print(f"Filtering dataset for language: {args.language_filter}")
                hf_dataset = hf_dataset.filter(
                    lambda x: x.get('lang', x.get('language', '')) == args.language_filter,
                    desc=f"Filtering for {args.language_filter}"
                )
                print(f"Filtered dataset size: {len(hf_dataset)} samples")
            
            #hf_dataset = hf_dataset.take(100_000)

        # Prepare wechsel_config with dataset if available.
        # Pre-tokenized binary data already has the tokenizer baked in, so
        # WECHSEL language adaptation is skipped in that case.
        if args.token_bin is not None:
            wechsel_config = None
        else:
            wechsel_config = {
                'train_corpus_path': args.train_data,
                'source_language': src_language,
                'target_language': tgt_language,
                'dictionary': wechsel_dictionary
            }
            if hf_dataset is not None:
                wechsel_config['dataset'] = hf_dataset
                wechsel_config['text_column'] = args.text_column

        model, tokenizer = setup_model(model_name, adapter_type, adapter_config, num_tailor_layers, wechsel_config, tokenizer_path)

    PADDING_VALUE = tokenizer.pad_token_id

    # Create training dataset
    if args.token_bin is not None:
        print(f"Creating training dataset from pre-tokenized binary: {args.token_bin}")
        full_dataset = TokenBinDataset(
            bin_path=args.token_bin,
            context_size=context_size,
        )

        # Split into train and validation if requested
        if val_fraction > 0:
            val_size = int(full_dataset.num_chunks * val_fraction)
            train_size = full_dataset.num_chunks - val_size

            # Calculate token boundaries for slicing
            val_start_token = train_size * context_size

            train_dataset = TokenBinDataset(
                bin_path=args.token_bin,
                context_size=context_size,
                end_token=val_start_token,
            )
            val_dataset = TokenBinDataset(
                bin_path=args.token_bin,
                context_size=context_size,
                start_token=val_start_token,
            )
            print(f"Split into train ({train_size} chunks) and validation ({val_size} chunks)")
            val_collate_fn = val_dataset.collate
        else:
            train_dataset = full_dataset
            val_dataset = None
            val_collate_fn = None

        collate_fn = train_dataset.collate
    elif args.train_data is not None:
        print(f"Creating training dataset from file: {args.train_data}")
        train_dataset = ChunkIterableDataset(
            corpus_path=args.train_data,
            tokenizer=tokenizer,
            context_size=context_size,
            chunk_size=chunk_size
        )
        collate_fn = collate
        val_dataset = None
        val_collate_fn = None
    else:
        print(f"Creating training dataset from HuggingFace dataset")
        # Dataset was already loaded above, reuse it
        train_dataset = ChunkIterableDataset(
            dataset=hf_dataset,
            tokenizer=tokenizer,
            context_size=context_size,
            chunk_size=chunk_size,
            text_column=args.text_column
        )
        collate_fn = collate
        val_dataset = None
        val_collate_fn = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
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
            pin_memory=True if device.type in ['cuda', 'mps'] else False
        )

    model = train(model, train_dataloader, device, model_path, num_epochs, adam_beta1, adam_beta2, weight_decay, early_stopping_patience, early_stopping_min_delta, val_dataloader)

    save_model(model, adapter_type, adapter_config, model_path)

    wandb.save(f"{model_path}.wandb")
    wandb.finish()
