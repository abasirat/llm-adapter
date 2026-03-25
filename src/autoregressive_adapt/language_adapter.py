import pdb

import torch
import os
import sys
import argparse
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from tqdm import tqdm
import math
import wandb
from datetime import datetime
from .model_setup import setup_model, load_learnable_params, save_learnable_params, train_tokenizer
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
    def __init__(self, corpus_path, tokenizer, chunk_size:int = 4*1024, context_size:int = 1024):
        if not os.path.exists(corpus_path):
            print(f"File {corpus_path} not found!")
            return
        
        self.corpus_path = corpus_path
        self.chunk_size = chunk_size

        self.context_size = context_size
        self.tokenizer = tokenizer

        self.corpus_size = os.path.getsize(self.corpus_path)
        print(f"Corpus size: {self.corpus_size} bytes")

        self.tell = 0

    def __iter__(self):
        return self.chunk_generator()
    
    #def __next__(self):
    #    pass
    def chunk_generator(self):
        """
        Generator that reads a file token by token.
        """
        self.tell = 0
        buffer = []
        with open(self.corpus_path, 'r', encoding='utf-8') as file:
            while True:
                chunk = file.read(self.chunk_size)
                if not chunk: break  # end of file reached

                self.tell += len(chunk.encode("utf-8"))

                buffer += self.tokenizer(chunk, truncation=False, add_special_tokens=False).input_ids
                
                while len(buffer) > self.context_size:
                    yield {
                        'input_ids':torch.tensor(buffer[:self.context_size]),
                        'progress':self.tell,
                    }
                    buffer = buffer[self.context_size:]
                    self.tell = 0
        
        if buffer:
            yield {
                'input_ids': torch.tensor(buffer),
                'progress': self.tell
            }
            
    def __len__(self):
        return self.corpus_size
    
    def __getitem__(self, idx):
        raise NotImplemented

def collate(batch):
    input_ids = [x['input_ids'] for x in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PADDING_VALUE)
    attention_mask = (input_ids != PADDING_VALUE)
    progress = sum(b['progress'] for b in batch)
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

def train(model, train_dataloader, device, model_path, num_epochs=1):
    optimizer = Adam(model.parameters(), lr=learning_rate)

    total_steps = num_epochs*len(train_dataloader)
    warmup_steps = int(.1*total_steps)
    base_lr = learning_rate
    scheduler = WarmUpCosineDecayScheduler(optimizer, warmup_steps, total_steps, base_lr)
    print(f"LR scheduler set up with {warmup_steps} warmup steps and {total_steps} total steps")

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(total=len(train_dataloader.dataset), desc="Processing")
        for i, (batch, progress) in enumerate(train_dataloader):
            batch = {k:v.to(device) for k,v in batch.items()}

            outputs = model(**batch)
            
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step(progress)

            progress_bar.update(progress)
            progress_bar.set_description(f"running loss: {total_loss/(i+1):.4f}, batch loss: {loss.item():.4f}, LR: {scheduler.get_lr():.6f}")

            wandb.log({"batch_loss": loss.item()})
            wandb.log({"avg_loss": total_loss / (i+1)})
            wandb.log({"learning_rate": scheduler.get_lr()})

            if i%1000 == 0:
                print(f"save parameters - progress {progress}")
                save_learnable_params(model, adapter_type, adapter_config, model_path+'-trace')

        progress_bar.close()

        avg_loss = total_loss / len(train_dataloader)
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a language model with adapters')
    parser.add_argument('model_name', type=str, help='Name or path of the base model (e.g., gpt2)')
    parser.add_argument('train_data', type=str, help='Path to training data file')
    parser.add_argument('model_path', type=str, help='Path to save the model checkpoint')
    parser.add_argument('num_tailor_layers', type=int, help='Number of tailor layers to add')
    parser.add_argument('adapter_type', type=str, choices=['none', 'layer_adapter', 'lora'],
                        help='Type of adapter to use')
    parser.add_argument('chkpt', type=str, nargs='?', default='',
                        help='Path to checkpoint to resume from (optional)')
    parser.add_argument('num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('project_name', type=str, help='Weights & Biases project name')
    parser.add_argument('experiment_description', type=str, help='Experiment description for W&B')
    parser.add_argument('src_language', type=str, help='Source language code')
    parser.add_argument('tgt_language', type=str, help='Target language code')
    parser.add_argument('wechsel_dictionary', type=str, help='Path to WECHSEL dictionary')
    parser.add_argument('--chunk_size', type=int, default=4*1024, help='Chunk size for data loading (default: 4096)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--context_size', type=int, default=1024, help='Context size (default: 1024)')

    args = parser.parse_args()

    model_name = args.model_name
    train_data = args.train_data
    model_path = args.model_path
    num_tailor_layers = args.num_tailor_layers
    adapter_type = args.adapter_type
    chkpt = args.chkpt
    num_epochs = args.num_epochs
    project_name = args.project_name
    experiment_description = args.experiment_description
    src_language = args.src_language
    tgt_language = args.tgt_language
    wechsel_dictionary = args.wechsel_dictionary
    chunk_size = args.chunk_size
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    context_size = args.context_size

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    wandb.init(
    project=project_name, 
    name=f"{experiment_description}_{current_date}", 
    config={                       
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "context_size": context_size,
        "model_name": model_name,
        "num_tailor_layers": num_tailor_layers,
        "adapter_type": adapter_type,
        "ckeck_point": chkpt,
        "num_epochs": num_epochs,
    })

    device = set_device()
    print(f"The active device is {device}")

    if chkpt: # and os.path.exists(chkpt):
        print(f"loading model from {chkpt}")
        model, tokenizer, adapter_config = load_learnable_params(chkpt)
    else:
        if adapter_type == 'none':
            adapter_config = None
        elif adapter_type == 'layer_adapter':
            adapter_config = {
                'hidden_size': 32,
                'num_heads': 8,
            }
        elif adapter_type == 'lora':
            adapter_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,  
                r=4,                          
                lora_alpha=16,             
                target_modules=["attn.c_proj"],  
                lora_dropout=0.1          
            )
        
        wechsel_config = {
            'train_corpus_path': train_data,
            'source_language': src_language,
            'target_language': tgt_language,
            'dictionary': wechsel_dictionary
        }

        model, tokenizer = setup_model(model_name, adapter_type, adapter_config, num_tailor_layers, wechsel_config)

    PADDING_VALUE = tokenizer.pad_token_id

    train_dataset = ChunkIterableDataset(train_data, tokenizer, context_size=context_size, chunk_size=chunk_size)
    train_dataloader = DataLoader(train_dataset, batch_size, collate_fn=collate)

    model = train(model, train_dataloader, device, model_path, num_epochs)

    save_learnable_params(model, adapter_type, adapter_config, model_path)

    wandb.save(f"{model_path}.wandb")
    wandb.finish()
