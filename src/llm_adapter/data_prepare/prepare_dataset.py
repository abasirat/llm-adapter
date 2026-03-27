"""
Tokenize a dataset (e.g. Wikipedia) and store the result as a flat binary file
of token IDs, suitable for efficient memory-mapped language model training.

Each document is tokenized and terminated with the tokenizer's EOS token before
being appended to the output stream.  Two files are produced:

  <output_prefix>_<split>.bin   – flat binary array of token IDs (uint16 or int32)
  <output_prefix>_<split>.json  – metadata  (tokenizer, vocab size, total tokens …)

The .bin file can be loaded during training with numpy.memmap for zero-copy access:

    import numpy as np
    data = np.memmap("wikipedia_train.bin", dtype="uint16", mode="r")

Usage examples
--------------
# Wikipedia (English, 20220301.en config)
python data/prepare_dataset.py \
    --dataset_name wikipedia \
    --dataset_config 20220301.en \
    --split train \
    --tokenizer_name gpt2 \
    --output_prefix data/wikipedia

# Local text file (one document per line)
python data/prepare_dataset.py \
    --input_file data/dr_articles.txt \
    --tokenizer_name gpt2 \
    --output_prefix data/dr_articles

# WECHSEL – adapt source tokenizer to a target language, then tokenize
# Trains a new target-language tokenizer from the corpus, applies cross-lingual
# WECHSEL embedding transfer, saves the adapted tokenizer, and writes the .bin.
python data/prepare_dataset.py \
    --input_file data/dr_articles.txt \
    --tokenizer_name gpt2 \
    --output_prefix data/dr_articles \
    --model_name gpt2 \
    --wechsel_src_lang en \
    --wechsel_tgt_lang da \
    --wechsel_dictionary danish \
    --save_tokenizer models/gpt2_danish_tokenizer
"""

import argparse
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dtype_for_vocab(vocab_size: int):
    """Choose the most compact integer dtype that can represent all token IDs."""
    if vocab_size <= 2**16:
        return np.uint16
    return np.int32


def _flush_buffer(buf: list, out_file, dtype):
    """Write a list of ints to the output binary file."""
    arr = np.array(buf, dtype=dtype)
    arr.tofile(out_file)


def tokenize_hf_dataset(
    dataset,
    tokenizer,
    text_column: str,
    output_path: str,
    dtype,
    chunk_size: int = 4096,
    num_proc: int = 1,
):
    """
    Iterate over a HuggingFace dataset, tokenize each document, append EOS,
    and write token IDs to *output_path* as a flat binary array.

    Returns the total number of tokens written.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            "The tokenizer does not have an EOS token. "
            "Set tokenizer.eos_token before calling this script."
        )

    total_tokens = 0
    write_buffer = []

    with open(output_path, "wb") as out_file:
        for sample in tqdm(dataset, desc="Tokenizing", unit="doc"):
            text = sample[text_column]
            if not text or not text.strip():
                continue

            ids = tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
            ).input_ids
            ids.append(eos_id)  # document boundary marker

            write_buffer.extend(ids)
            total_tokens += len(ids)

            # Flush periodically to keep memory bounded
            if len(write_buffer) >= chunk_size * 256:
                _flush_buffer(write_buffer, out_file, dtype)
                write_buffer = []

        if write_buffer:
            _flush_buffer(write_buffer, out_file, dtype)

    return total_tokens


def tokenize_text_file(
    input_path: str,
    tokenizer,
    output_path: str,
    dtype,
    chunk_size: int = 4096,
):
    """
    Read a plain-text file where each non-empty line is treated as a separate
    document.  Tokenize, append EOS, and write to *output_path*.

    Returns the total number of tokens written.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            "The tokenizer does not have an EOS token. "
            "Set tokenizer.eos_token before calling this script."
        )

    total_tokens = 0
    write_buffer = []

    with open(input_path, "r", encoding="utf-8") as in_file, \
         open(output_path, "wb") as out_file:

        for line in tqdm(in_file, desc="Tokenizing", unit="line"):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            ids = tokenizer(
                line,
                truncation=False,
                add_special_tokens=False,
            ).input_ids
            ids.append(eos_id)

            write_buffer.extend(ids)
            total_tokens += len(ids)

            if len(write_buffer) >= chunk_size * 256:
                _flush_buffer(write_buffer, out_file, dtype)
                write_buffer = []

        if write_buffer:
            _flush_buffer(write_buffer, out_file, dtype)

    return total_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a dataset and save as a binary token-ID file."
    )

    # ── Data source (mutually exclusive) ───────────────────────────────────
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset_name",
        type=str,
        help="HuggingFace dataset name, e.g. 'wikipedia'.",
    )
    source.add_argument(
        "--input_file",
        type=str,
        help="Path to a local text file (one document per line).",
    )

    # ── HuggingFace dataset options ─────────────────────────────────────────
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset configuration, e.g. '20220301.en' for Wikipedia.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column that contains the document text (default: 'text').",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Load the HuggingFace dataset in streaming mode (low RAM usage).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap the number of documents processed (useful for testing).",
    )

    # ── Tokenizer ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="gpt2",
        help="HuggingFace tokenizer name or local path (default: gpt2).",
    )

    # ── Output ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help=(
            "Output file prefix, e.g. 'data/wikipedia'. "
            "The split name is appended automatically: "
            "'data/wikipedia_train.bin' and 'data/wikipedia_train.json'."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
        help="Tokenization chunk size (default: 4096).",
    )

    # ── WECHSEL tokenizer adaptation (optional) ─────────────────────────────
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name/path needed for WECHSEL embedding transfer (e.g. 'gpt2'). "
             "Required when --wechsel_src_lang is set.",
    )
    parser.add_argument(
        "--wechsel_src_lang",
        type=str,
        default=None,
        help="Source language code for WECHSEL (e.g. 'en'). "
             "Enables full WECHSEL tokenizer adaptation.",
    )
    parser.add_argument(
        "--wechsel_tgt_lang",
        type=str,
        default=None,
        help="Target language code for WECHSEL (e.g. 'da'). Required with --wechsel_src_lang.",
    )
    parser.add_argument(
        "--wechsel_dictionary",
        type=str,
        default=None,
        help="Bilingual dictionary name or path (e.g. 'danish'). Required with --wechsel_src_lang.",
    )
    parser.add_argument(
        "--save_tokenizer",
        type=str,
        default=None,
        help="Directory path to save the (WECHSEL-adapted) tokenizer.",
    )

    args = parser.parse_args()

    # ── Validate WECHSEL args ───────────────────────────────────────────────
    if args.wechsel_src_lang:
        missing = [f for f, v in [("--wechsel_tgt_lang", args.wechsel_tgt_lang),
                                   ("--wechsel_dictionary", args.wechsel_dictionary),
                                   ("--model_name", args.model_name)] if not v]
        if missing:
            parser.error(f"--wechsel_src_lang requires: {', '.join(missing)}")

    # ── Load tokenizer ──────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.eos_token is None:
        # Fall back to a common sentinel if EOS is not defined
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        print("Warning: tokenizer had no EOS token; added '<|endoftext|>'.")
    print(f"  vocab size : {tokenizer.vocab_size}")
    print(f"  EOS token  : '{tokenizer.eos_token}'  (id={tokenizer.eos_token_id})")

    # ── Load HF dataset early (reused for WECHSEL and tokenization) ─────────
    dataset = None
    if args.dataset_name:
        from datasets import load_dataset

        print(f"\nLoading dataset '{args.dataset_name}'"
              + (f" ({args.dataset_config})" if args.dataset_config else "")
              + f" – split '{args.split}'"
              + (" [streaming]" if args.streaming else ""))

        load_kwargs = dict(split=args.split, streaming=args.streaming)
        if args.dataset_config:
            dataset = load_dataset(args.dataset_name, args.dataset_config, **load_kwargs)
        else:
            dataset = load_dataset(args.dataset_name, **load_kwargs)

        if args.max_samples is not None:
            dataset = dataset.take(args.max_samples) if args.streaming \
                      else dataset.select(range(min(args.max_samples, len(dataset))))
            print(f"  Capped at {args.max_samples} samples.")

    # ── Optional WECHSEL tokenizer adaptation ──────────────────────────────
    # Trains a new target-language tokenizer from the corpus via
    # train_new_from_iterator, then applies cross-lingual embedding transfer
    # using WECHSEL.  The adapted tokenizer (not the model) is what is used
    # for the subsequent tokenization of the dataset.
    wechsel_applied = False
    if args.wechsel_src_lang:
        from transformers import AutoModelForCausalLM
        from llm_adapter import train_tokenizer

        print(f"\nApplying WECHSEL tokenizer adaptation")
        print(f"  source language : {args.wechsel_src_lang}")
        print(f"  target language : {args.wechsel_tgt_lang}")
        print(f"  dictionary      : {args.wechsel_dictionary}")
        print(f"  model           : {args.model_name}")

        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        _, tokenizer = train_tokenizer(
            train_corpus_path=args.input_file,
            source_tokenizer=tokenizer,
            model=model,
            source_language=args.wechsel_src_lang,
            target_language=args.wechsel_tgt_lang,
            dictionary=args.wechsel_dictionary,
            dataset=dataset,
            text_column=args.text_column,
        )
        del model  # embedding-transferred model not needed here
        print(f"  WECHSEL done. Adapted vocab size: {len(tokenizer)}")
        wechsel_applied = True

    # ── Save tokenizer (if requested) ───────────────────────────────────────
    if args.save_tokenizer:
        os.makedirs(args.save_tokenizer, exist_ok=True)
        tokenizer.save_pretrained(args.save_tokenizer)
        print(f"  Tokenizer saved to: {args.save_tokenizer}")

    # ── dtype (computed after potential vocab change from WECHSEL) ──────────
    dtype = _dtype_for_vocab(tokenizer.vocab_size)
    print(f"  storage dtype: {dtype.__name__}")

    # ── Determine output paths ──────────────────────────────────────────────
    if args.dataset_name:
        split_tag = args.split
    else:
        base = os.path.splitext(os.path.basename(args.input_file))[0]
        split_tag = base

    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)
    bin_path  = f"{args.output_prefix}_{split_tag}.bin"
    meta_path = f"{args.output_prefix}_{split_tag}.json"

    # ── Tokenize ────────────────────────────────────────────────────────────
    # For streaming HF datasets, iterating again after WECHSEL is fine –
    # each __iter__() call on an IterableDataset creates a fresh stream.
    if args.dataset_name:
        print(f"\nWriting tokenized data to: {bin_path}")
        total_tokens = tokenize_hf_dataset(
            dataset,
            tokenizer,
            text_column=args.text_column,
            output_path=bin_path,
            dtype=dtype,
            chunk_size=args.chunk_size,
        )

    else:
        if not os.path.exists(args.input_file):
            sys.exit(f"Error: input file not found: {args.input_file}")

        print(f"\nReading from local file: {args.input_file}")
        print(f"Writing tokenized data to: {bin_path}")
        total_tokens = tokenize_text_file(
            args.input_file,
            tokenizer,
            output_path=bin_path,
            dtype=dtype,
            chunk_size=args.chunk_size,
        )

    # ── Write metadata ──────────────────────────────────────────────────────
    file_size_mb = os.path.getsize(bin_path) / (1024 ** 2)
    metadata = {
        "tokenizer_name": args.tokenizer_name,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "vocab_size": tokenizer.vocab_size,
        "dtype": dtype.__name__,
        "total_tokens": total_tokens,
        "file_size_mb": round(file_size_mb, 2),
        "source": args.dataset_name or args.input_file,
        "split": split_tag,
        "wechsel": {
            "applied": wechsel_applied,
            "src_lang": args.wechsel_src_lang,
            "tgt_lang": args.wechsel_tgt_lang,
            "dictionary": args.wechsel_dictionary,
            "model_name": args.model_name,
            "tokenizer_saved_to": args.save_tokenizer,
        } if wechsel_applied else {"applied": False},
        "format": (
            "Flat binary array of token IDs. "
            "Load with: np.memmap('<file>.bin', dtype='<dtype>', mode='r')"
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone.")
    print(f"  Total tokens : {total_tokens:,}")
    print(f"  File size    : {file_size_mb:.1f} MB")
    print(f"  Binary file  : {bin_path}")
    print(f"  Metadata     : {meta_path}")
    if args.save_tokenizer:
        print(f"  Tokenizer    : {args.save_tokenizer}")
    print(
        f"\nTo load during training:\n"
        f"  import numpy as np\n"
        f"  data = np.memmap('{bin_path}', dtype='{dtype.__name__}', mode='r')"
    )


if __name__ == "__main__":
    main()
