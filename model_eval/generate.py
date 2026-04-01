import sys
import torch
import argparse
import json
from pathlib import Path
from llm_adapter import load_learnable_params
from transformers import AutoTokenizer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained language model')

    # Model and tokenizer paths
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model parameters')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to tokenizer')

    # Input/Output
    parser.add_argument('--prompts_file', type=str, required=True,
                        help='Path to input file with prompts (one per line)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output file for generation results (JSON)')

    # Generation parameters
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length of generated text (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling parameter (default: 0.9)')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                        help='Repetition penalty (default: 1.2)')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=2,
                        help='No repeat n-gram size (default: 2)')
    parser.add_argument('--do_sample', type=bool, default=True,
                        help='Use sampling instead of greedy decoding (default: True)')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: cuda, mps, or cpu (default: auto-detect)')

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model, tokenizer, adapter_config = load_learnable_params(args.model_path, tokenizer_path=args.tokenizer_path)
    model.to(device)
    model.eval()

    # Read prompts from file
    print(f"Reading prompts from {args.prompts_file}")
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(prompts)} prompts")

    # Generate text for each prompt
    results = []
    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_length=args.max_length,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            results.append({
                "prompt": prompt,
                "generated": generated_text,
                "index": i
            })

    # Save results to output file
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(results)} outputs successfully")


if __name__ == "__main__":
    main()