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
    --max_bin_size_gb 5 \
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
import logging
import os
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def _dtype_for_vocab(vocab_size: int):
    """Choose the most compact integer dtype that can represent all token IDs."""
    if vocab_size <= 2**16:
        return np.uint16
    return np.int32


def _flush_buffer(buf: list, out_file, dtype):
    """Write a list of ints to the output binary file."""
    arr = np.array(buf, dtype=dtype)
    arr.tofile(out_file)


def _fit_document_to_budget(ids, eos_id, tokens_written, max_tokens):
    """
    Fit a tokenized document into the remaining token budget.

    Returns:
        fitted_ids, limit_reached, was_truncated
    """
    if max_tokens is None:
        return ids, False, False

    remaining_tokens = max_tokens - tokens_written
    if remaining_tokens <= 0:
        return [], True, False

    if len(ids) <= remaining_tokens:
        return ids, False, False

    if tokens_written > 0:
        return [], True, False

    if remaining_tokens == 1:
        return [eos_id], True, True

    trimmed_ids = ids[: remaining_tokens - 1] + [eos_id]
    return trimmed_ids, True, True


def tokenize_hf_dataset(
    dataset,
    tokenizer,
    text_column: str,
    output_path: str,
    dtype,
    chunk_size: int = 4096,
    max_tokens: int = None,
):
    """
    Iterate over a HuggingFace dataset, tokenize each document, append EOS,
    and write token IDs to *output_path* as a flat binary array.

    Returns the total number of tokens written and whether the output was capped.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            "The tokenizer does not have an EOS token. "
            "Set tokenizer.eos_token before calling this script."
        )

    total_tokens = 0
    write_buffer = []
    limit_reached = False
    was_truncated = False

    logger.info("Stage: tokenizing HuggingFace dataset into %s", output_path)

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

            ids, limit_reached, doc_truncated = _fit_document_to_budget(
                ids,
                eos_id,
                total_tokens,
                max_tokens,
            )
            was_truncated = was_truncated or doc_truncated
            if limit_reached and not ids:
                break

            write_buffer.extend(ids)
            total_tokens += len(ids)

            # Flush periodically to keep memory bounded
            if len(write_buffer) >= chunk_size * 256:
                _flush_buffer(write_buffer, out_file, dtype)
                write_buffer = []

            if limit_reached:
                break

        if write_buffer:
            _flush_buffer(write_buffer, out_file, dtype)

    if limit_reached or was_truncated:
        logger.info("Stage: output size cap reached while tokenizing dataset")

    return total_tokens, limit_reached or was_truncated


def tokenize_text_file(
    input_path: str,
    tokenizer,
    output_path: str,
    dtype,
    chunk_size: int = 4096,
    max_tokens: int = None,
):
    """
    Read a plain-text file where each non-empty line is treated as a separate
    document.  Tokenize, append EOS, and write to *output_path*.

    Returns the total number of tokens written and whether the output was capped.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            "The tokenizer does not have an EOS token. "
            "Set tokenizer.eos_token before calling this script."
        )

    total_tokens = 0
    write_buffer = []
    limit_reached = False
    was_truncated = False

    logger.info("Stage: tokenizing local text file %s into %s", input_path, output_path)

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

            ids, limit_reached, doc_truncated = _fit_document_to_budget(
                ids,
                eos_id,
                total_tokens,
                max_tokens,
            )
            was_truncated = was_truncated or doc_truncated
            if limit_reached and not ids:
                break

            write_buffer.extend(ids)
            total_tokens += len(ids)

            if len(write_buffer) >= chunk_size * 256:
                _flush_buffer(write_buffer, out_file, dtype)
                write_buffer = []

            if limit_reached:
                break

        if write_buffer:
            _flush_buffer(write_buffer, out_file, dtype)

    if limit_reached or was_truncated:
        logger.info("Stage: output size cap reached while tokenizing local file")

    return total_tokens, limit_reached or was_truncated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_logging()

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
        default=None,
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
    parser.add_argument(
        "--max_bin_size_gb",
        type=float,
        default=5.0,
        help="Maximum size of the output .bin file in GB (default: 5).",
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
    if args.max_bin_size_gb <= 0:
        parser.error("--max_bin_size_gb must be greater than 0")

    # ── Load tokenizer ──────────────────────────────────────────────────────
    logger.info("Stage: loading tokenizer %s", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.eos_token is None:
        # Fall back to a common sentinel if EOS is not defined
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        logger.warning("Tokenizer had no EOS token; added '<|endoftext|>'.")
    logger.info("Tokenizer ready | vocab size=%s | eos=%r | eos_id=%s", tokenizer.vocab_size, tokenizer.eos_token, tokenizer.eos_token_id)

    # ── Load HF dataset early (reused for WECHSEL and tokenization) ─────────
    dataset = None
    if args.dataset_name:
        from datasets import load_dataset

        logger.info(
            "Stage: loading dataset %s%s | split=%s%s",
            args.dataset_name,
            f" ({args.dataset_config})" if args.dataset_config else "",
            args.split,
            " | streaming" if args.streaming else "",
        )

        
        load_kwargs = dict(split=args.split, streaming=args.streaming, trust_remote_code=True)
        if args.dataset_config:
            dataset = load_dataset(args.dataset_name, args.dataset_config, **load_kwargs)
        else:
            dataset = load_dataset(args.dataset_name, **load_kwargs)

        if args.max_samples is not None:
            dataset = dataset.take(args.max_samples) if args.streaming \
                      else dataset.select(range(min(args.max_samples, len(dataset))))
            logger.info("Dataset sample cap applied | max_samples=%s", args.max_samples)

    # ── Optional WECHSEL tokenizer adaptation ──────────────────────────────
    # Trains a new target-language tokenizer from the corpus via
    # train_new_from_iterator, then applies cross-lingual embedding transfer
    # using WECHSEL.  The adapted tokenizer (not the model) is what is used
    # for the subsequent tokenization of the dataset.
    wechsel_applied = False
    if args.wechsel_src_lang:
        from transformers import AutoModelForCausalLM
        from llm_adapter import train_tokenizer

        logger.info(
            "Stage: applying WECHSEL | src=%s | tgt=%s | dictionary=%s | model=%s",
            args.wechsel_src_lang,
            args.wechsel_tgt_lang,
            args.wechsel_dictionary,
            args.model_name,
        )

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
            max_train_size=args.max_bin_size_gb * (1024 ** 3) // 2,  # Use up to half of the output cap for tokenizer training
        )
        del model  # embedding-transferred model not needed here
        logger.info("WECHSEL complete | adapted vocab size=%s", len(tokenizer))
        wechsel_applied = True

    # ── Save tokenizer (if requested) ───────────────────────────────────────
    if args.save_tokenizer:
        os.makedirs(args.save_tokenizer, exist_ok=True)
        tokenizer.save_pretrained(args.save_tokenizer)
        logger.info("Tokenizer saved to %s", args.save_tokenizer)

    # ── dtype (computed after potential vocab change from WECHSEL) ──────────
    dtype = _dtype_for_vocab(tokenizer.vocab_size)
    logger.info("Storage dtype selected | dtype=%s", dtype.__name__)
    dtype_bytes = np.dtype(dtype).itemsize
    max_bin_size_bytes = int(args.max_bin_size_gb * (1024 ** 3))
    max_tokens = max_bin_size_bytes // dtype_bytes
    logger.info("Output size cap configured | size_gb=%.2f | size_bytes=%s", args.max_bin_size_gb, f"{max_bin_size_bytes:,}")

    if max_tokens <= 0:
        parser.error("--max_bin_size_gb is too small for the selected token dtype")

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
        total_tokens, output_capped = tokenize_hf_dataset(
            dataset,
            tokenizer,
            text_column=args.text_column,
            output_path=bin_path,
            dtype=dtype,
            chunk_size=args.chunk_size,
            max_tokens=max_tokens,
        )

    else:
        if not os.path.exists(args.input_file):
            sys.exit(f"Error: input file not found: {args.input_file}")

        total_tokens, output_capped = tokenize_text_file(
            args.input_file,
            tokenizer,
            output_path=bin_path,
            dtype=dtype,
            chunk_size=args.chunk_size,
            max_tokens=max_tokens,
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
        "max_bin_size_gb": args.max_bin_size_gb,
        "max_bin_size_bytes": max_bin_size_bytes,
        "output_capped": output_capped,
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

    logger.info("Stage: completed dataset preparation")
    logger.info("Run summary | total_tokens=%s | file_size_mb=%.1f", f"{total_tokens:,}", file_size_mb)
    logger.info("Artifacts | binary=%s | metadata=%s", bin_path, meta_path)
    if output_capped:
        logger.info("Output cap reached configured maximum binary size")
    if args.save_tokenizer:
        logger.info("Artifacts | tokenizer=%s", args.save_tokenizer)
    logger.info(
        "Load hint | np.memmap('%s', dtype='%s', mode='r')",
        bin_path,
        dtype.__name__,
    )


if __name__ == "__main__":
    main()
