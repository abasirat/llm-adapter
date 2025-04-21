import torch
from layer_adapter import LayerAdapter
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, LlamaForCausalLM
from language_tailor import LanguageAdapter
from peft import get_peft_model
import os
import pdb

def setup_model(model_name='gpt2', adapter_type='none', adapter_config=None, num_tailor_layers=0):
    
    if model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')
        if tokenizer.pad_token is None:  # If the tokenizer doesn't have a pad_token
            tokenizer.pad_token = tokenizer.eos_token

        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            device_map='auto',  
        )
        
        for param in model.parameters(): 
            param.requires_grad = False
        print_trainable_parameters(model, "Base Model")

        if adapter_type == 'layer_adapter':
            model.transformer = LayerAdapter(model.transformer, **adapter_config)
            print_trainable_parameters(model, "Layer Adapted")
        elif adapter_type == 'lora':
            model = get_peft_model(model, adapter_config)
            print_trainable_parameters(model, "Lora")
        
        # the task specific tailor module
        model.transformer = LanguageAdapter(model.transformer, num_tailor_layers)

    if 'Llama' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None: 
            tokenizer.pad_token = tokenizer.eos_token
            
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map='auto',  
        )

        for param in model.parameters(): 
            param.requires_grad = False
        print_trainable_parameters(model, "Base Model")

        if adapter_type == 'layer_adapter':
            model.model = LayerAdapter(model.model, **adapter_config)
            print_trainable_parameters(model, "Layer Adapted")
        elif adapter_type == 'lora':
            model = get_peft_model(model, adapter_config)
            print_trainable_parameters(model, "Lora")
        
        # the task specific tailor module
        model.model = LanguageAdapter(model.model, num_tailor_layers)

    # Make lm_head trainable
    for param in model.lm_head.parameters():
        param.requires_grad = True
    print_trainable_parameters(model, "Tailored")

    return model, tokenizer

def print_trainable_parameters(model, message):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"{message} trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")

def save_learnable_params(model, adapter_type, adapter_config, save_path="learnable_params.pth"):
    learnable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    torch.save({
        'learnable_params':learnable_params, 
        'encoder_config': model.config,
        'adapter_type': adapter_type,
        'adapter_config': adapter_config,
        'num_tailor_layers': model.model.num_tailor_layers
    },
    save_path)
    print(f"Parameters saved to {save_path}")

def load_learnable_params(save_path):
    saved_params = torch.load(save_path, map_location="cpu")

    model_name = saved_params['encoder_config'].name_or_path
    num_tailor_layers = saved_params['num_tailor_layers']
    adapter_type = saved_params['adapter_type']
    adapter_config = saved_params['adapter_config']
    model, tokenizer = setup_model(model_name, adapter_type, adapter_config, num_tailor_layers)
    
    model_state_dict = model.state_dict()
    model_state_dict.update(saved_params['learnable_params'])
    model.load_state_dict(model_state_dict)
    print(f"Parameters loaded from {save_path}")
    return model, tokenizer, adapter_config
