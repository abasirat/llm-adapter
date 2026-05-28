#!/usr/bin/env python3
import argparse
from html import parser
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_adapter import load_model as load_adapter_model


def load_prompts(path):
    prompts = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if path.endswith(".jsonl"):
                obj = json.loads(line)
                prompts.append(obj["prompt"])
            else:
                prompts.append(line)

    return prompts


def load_model(model_name, device):
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        model, tokenizer, _ = load_adapter_model(model_name)
    model.to(device)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def generate_for_model(model_name, prompts, args):
    model, tokenizer = load_model(model_name, args.device)

    outputs = []

    for prompt in tqdm(prompts, desc=f"Generating: {model_name}"):
        prompt_for_generation = prompt.strip() + "\n"
        inputs = tokenizer(prompt_for_generation, return_tensors="pt").to(args.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            top_k=args.top_k if args.do_sample else None,
            num_return_sequences=args.num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

        for seq_id, output_ids in enumerate(generated_ids):
            full_text = tokenizer.decode(output_ids, skip_special_tokens=True)

            generated_only_ids = output_ids[inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(
                generated_only_ids,
                skip_special_tokens=True,
            )

            outputs.append(
                {
                    "model": model_name,
                    "prompt": prompt,
                    "generation_id": seq_id,
                    "text": generated_text.strip(),
                    "full_text": full_text.strip(),
                }
            )

    del model
    torch.cuda.empty_cache()

    return outputs


def safe_model_name(model_name):
    base_name = model_name.split("/")[-1]
    return (
        base_name.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more HuggingFace model names or local model paths",
    )

    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="TXT file with one prompt per line, or JSONL with {'prompt': ...}",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_outputs",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--do_sample",
        action="store_true",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
    )

    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)

    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prompts = load_prompts(args.prompts_file)

    print(f"Loaded {len(prompts)} prompts.")

    for model_name in args.models:
        outputs = generate_for_model(model_name, prompts, args)

        out_name = safe_model_name(model_name) + "_outputs.jsonl"
        out_path = os.path.join(args.output_dir, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            for obj in outputs:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"Saved {len(outputs)} outputs to {out_path}")


if __name__ == "__main__":
    main()
