import sys
import torch
from llm_adapter import load_learnable_params

model_path = "models/gpt2_dr_articles_layer_adapter.pt"

model, tokenizer, adapter_config = load_learnable_params(model_path)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = "På overraskende kort tid har oprørsgrupper taget kontrol"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))