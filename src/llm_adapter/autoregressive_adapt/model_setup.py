import torch
from .layer_adapter import LayerAdapter
from transformers import AutoTokenizer, GPT2LMHeadModel
from .language_tailor import LanguageAdapter
from peft import get_peft_model
from wechsel import WECHSEL, load_embeddings
import os
import pdb

def setup_model(model_name='gpt2', adapter_type='none', adapter_config=None, num_tailor_layers=0, wechsel_config=None, path_to_tokenizer=None):
    
    if path_to_tokenizer is not None and os.path.exists(path_to_tokenizer):
        tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer, padding_side='left')
        print(f"Loaded tokenizer from {path_to_tokenizer}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token is None:  # If the tokenizer doesn't have a pad_token
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        device_map='auto',
    )

    # If wechsel_config is provided and tokenizer needs to be trained, do so before setting up adapters
    if wechsel_config is not None and (path_to_tokenizer is None or not os.path.exists(path_to_tokenizer)):
        # Determine if using file or dataset
        train_corpus_path = wechsel_config.get('train_corpus_path')
        dataset = wechsel_config.get('dataset')
        text_column = wechsel_config.get('text_column', 'text')
        max_train_size = wechsel_config.get('max_train_size')

        model, tokenizer = train_tokenizer(
            train_corpus_path=train_corpus_path,
            source_tokenizer=tokenizer,
            model=model,
            source_language=wechsel_config['source_language'],
            target_language=wechsel_config['target_language'],
            dictionary=wechsel_config['dictionary'],
            dataset=dataset,
            text_column=text_column,
            max_train_size=max_train_size,
        )

        if path_to_tokenizer is not None:
            tokenizer.save_pretrained(path_to_tokenizer)
            print(f"Tokenizer saved to {path_to_tokenizer}")

    # Freeze all base model parameters
    for param in model.parameters(): 
        param.requires_grad = False
    print_trainable_parameters(model, "Base Model Frozen")

    if adapter_type == 'layer_adapter':
        model.transformer = LayerAdapter(model.transformer, **adapter_config)
        print_trainable_parameters(model, "Layer Adapted")
    elif adapter_type == 'lora':
        model = get_peft_model(model, adapter_config)
        print_trainable_parameters(model, "Lora")
    
    # the task specific tailor module
    model.transformer = LanguageAdapter(model.transformer, num_tailor_layers, dropout=adapter_config.get('dropout', 0.1))
    print_trainable_parameters(model, "Tailored")

    # Make input and output embeddings trainable by default
    # Note that in GPT-2, input and output embeddings are tied, so we only need to make the input embeddings trainable. If using a model with separate output embeddings, we would also need to make those trainable.
    #model.transformer.encoder.encoder.get_input_embeddings().weight.requires_grad = True
    #if not model.config.tie_word_embeddings:
    #    model.transformer.encoder.encoder.get_output_embeddings().weight.requires_grad = True
    #print_trainable_parameters(model, "Embeddings")

    # Make lm_head trainable
    # In GPT-2, the lm_head is tied to the input embeddings, so if the input embeddings are trainable, the lm_head will be trainable as well. If using a model with separate lm_head, we would need to make that trainable explicitly.
    for param in model.lm_head.parameters():
        param.requires_grad = True
    print_trainable_parameters(model, "Headed")

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

def save_model(
    model,
    adapter_type,
    adapter_config,
    save_path="learnable_params.pth",
    wechsel_config=None,
    tokenizer=None,
    tokenizer_path=None,
):
    learnable_params = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    payload = {
        "learnable_params": learnable_params,
        "model_name": model.config.name_or_path,
        "adapter_type": adapter_type,
        "adapter_config": adapter_config,
        "num_tailor_layers": model.transformer.num_tailor_layers,
        "wechsel_config": wechsel_config,
    }

    torch.save(payload, save_path)
    print(f"Parameters saved to {save_path}")

    if tokenizer is not None and tokenizer_path is not None:
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")

def load_model(param_path, tokenizer_path=None):
    saved_params = torch.load(param_path, map_location="cpu", weights_only=False)

    model_name = saved_params["model_name"]
    num_tailor_layers = saved_params["num_tailor_layers"]
    adapter_type = saved_params["adapter_type"]
    adapter_config = saved_params["adapter_config"]
    wechsel_config = saved_params.get("wechsel_config", None)

    model, tokenizer = setup_model(
        model_name=model_name,
        adapter_type=adapter_type,
        adapter_config=adapter_config,
        num_tailor_layers=num_tailor_layers,
        wechsel_config=wechsel_config,
        path_to_tokenizer=tokenizer_path,
    )

    # Restore trainable params
    missing, unexpected = model.load_state_dict(saved_params["learnable_params"], strict=False)

    print(f"Parameters loaded from {param_path}")
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    return model, tokenizer, adapter_config

def train_tokenizer(train_corpus_path=None, source_tokenizer=None, model=None, source_language=None, target_language=None, dictionary=None, chunk_size=4*1024, dataset=None, text_column="text", max_train_size=None):
    """
    Train a new tokenizer using WECHSEL for language adaptation.

    Args:
        train_corpus_path: Path to training corpus file (mutually exclusive with dataset)
        source_tokenizer: Tokenizer to train from
        model: Model to update embeddings for
        source_language: Source language code
        target_language: Target language code
        dictionary: Path to bilingual dictionary
        chunk_size: Chunk size for reading files
        dataset: HuggingFace dataset object (mutually exclusive with train_corpus_path)
        text_column: Column name in dataset containing text (default: "text")
        max_train_size: Maximum number of UTF-8 bytes to consume when training
            the tokenizer. If None, use the full available dataset.

    Returns:
        model, target_tokenizer: Updated model and trained tokenizer
    """

    if train_corpus_path is None and dataset is None:
        raise ValueError("Either train_corpus_path or dataset must be provided")

    if train_corpus_path is not None and dataset is not None:
        raise ValueError("Only one of train_corpus_path or dataset should be provided, not both")

    if max_train_size is not None and max_train_size <= 0:
        raise ValueError("max_train_size must be greater than 0 when provided")

    # Create batch iterator based on data source
    if train_corpus_path is not None:
        def batch_iterator():
            bytes_used = 0
            with open(train_corpus_path, 'r', encoding='utf-8') as f:
                for chunk in iter(lambda: f.read(chunk_size), ''):
                    if max_train_size is not None:
                        remaining_bytes = max_train_size - bytes_used
                        if remaining_bytes <= 0:
                            break

                    chunk_bytes = len(chunk.encode('utf-8'))
                    bytes_used += chunk_bytes
                    yield chunk

                    if max_train_size is not None and bytes_used >= max_train_size:
                        break
    else:
        def batch_iterator():
            bytes_used = 0
            for sample in dataset:
                text = sample[text_column]
                if max_train_size is not None:
                    remaining_bytes = max_train_size - bytes_used
                    if remaining_bytes <= 0:
                        break

                text_bytes = len(text.encode('utf-8'))
                bytes_used += text_bytes
                yield text

                if max_train_size is not None and bytes_used >= max_train_size:
                    break

    target_tokenizer = source_tokenizer.train_new_from_iterator(
        batch_iterator(),
        vocab_size=len(source_tokenizer)
    )

    wechsel = WECHSEL(
        load_embeddings(source_language),
        load_embeddings(target_language),
        bilingual_dictionary=dictionary
    )

    target_embeddings, info = wechsel.apply(
        source_tokenizer,
        target_tokenizer,
        model.get_input_embeddings().weight.detach().cpu().numpy(),
    )

    model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
    model.config.vocab_size = len(target_embeddings)

    # if the model has separate output embeddings, also copy those
    if not model.config.tie_word_embeddings:
        target_out_embeddings, info = wechsel.apply(
            source_tokenizer,
            target_tokenizer,
            model.get_output_embeddings().weight.detach().cpu().numpy(),
        )
        model.get_output_embeddings().weight.data = torch.from_numpy(target_out_embeddings)

    return model, target_tokenizer

