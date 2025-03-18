import torch
import os
import sys
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from tqdm import tqdm
import math
import wandb
from datetime import datetime
from model_setup import setup_model, load_learnable_params, save_learnable_params
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

    #model_name = 'gpt2'

    #train_data = 'fawiki-20181001-pages-articles-multistream-1-100000.txt'
    #val_data = 'fawiki-20181001-pages-articles-multistream100001-290169.txt'
    #train_data = "dr_articles.txt"
    #val_data = "politiken_articles.txt"
    #num_tailor_layers = 1

    model_name = sys.argv[1]
    train_data = sys.argv[2]
    model_path = sys.argv[3]
    num_tailor_layers = int(sys.argv[4])
    adapter_type = sys.argv[5]
    chkpt = sys.argv[6]
    num_epochs = int(sys.argv[7])
    project_name = sys.argv[8]
    experiment_description = sys.argv[9]
    
    learning_rate = 5e-5
    batch_size = 1
    context_size = 1024

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
        
        model, tokenizer = setup_model(model_name, adapter_type, adapter_config, num_tailor_layers)
    PADDING_VALUE = tokenizer.pad_token_id

    train_dataset = ChunkIterableDataset(train_data, tokenizer, context_size=context_size)
    train_dataloader = DataLoader(train_dataset, batch_size, collate_fn=collate)

    model = train(model, train_dataloader, device, model_path, num_epochs)

    save_learnable_params(model, adapter_type, adapter_config, model_path)

    wandb.save(f"{model_path}.wandb")
    wandb.finish()
