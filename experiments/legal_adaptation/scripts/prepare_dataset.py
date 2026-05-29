"""
Prepare a causal-LM dataset as a flat binary token-ID file.

Output:
  <output_prefix>_<split>.bin
  <output_prefix>_<split>.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
import transformers


logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_yaml(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dtype_for_vocab_size(vocab_size: int):
    if vocab_size <= 2**16:
        return np.uint16
    return np.int32


def flush_buffer(buffer, out_file, dtype):
    np.asarray(buffer, dtype=dtype).tofile(out_file)


def fit_document_to_budget(ids, eos_id, tokens_written, max_tokens):
    if max_tokens is None:
        return ids, False, False

    remaining = max_tokens - tokens_written

    if remaining <= 0:
        return [], True, False

    if len(ids) <= remaining:
        return ids, False, False

    if tokens_written > 0:
        return [], True, False

    if remaining == 1:
        return [eos_id], True, True

    return ids[: remaining - 1] + [eos_id], True, True


def preprocess_text(
    text,
    strip_whitespace=True,
    remove_empty_lines=True,
    lowercase=False,
    min_characters=0,
    max_characters=None,
):
    if text is None:
        return None

    if strip_whitespace:
        text = text.strip()

    if remove_empty_lines:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

    if lowercase:
        text = text.lower()

    if len(text) < min_characters:
        return None

    if max_characters is not None and len(text) > max_characters:
        text = text[:max_characters]

    return text


def tokenize_iterator(
    iterator,
    tokenizer,
    output_path,
    dtype,
    text_getter,
    chunk_size=4096,
    max_tokens=None,
    preprocessing=None,
):
    eos_id = tokenizer.eos_token_id

    if eos_id is None:
        raise ValueError("Tokenizer must have an EOS token.")

    preprocessing = preprocessing or {}

    total_tokens = 0
    docs_seen = 0
    docs_written = 0
    write_buffer = []
    output_capped = False
    any_truncated = False

    progress_bar = tqdm(desc="Tokenizing", unit="doc", total=max_tokens)
    with open(output_path, "wb") as out_file:
        for sample in iterator:
            docs_seen += 1

            text = text_getter(sample)
            text = preprocess_text(text, **preprocessing)

            if not text:
                continue

            ids = tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
            ).input_ids

            ids.append(eos_id)

            ids, limit_reached, doc_truncated = fit_document_to_budget(
                ids=ids,
                eos_id=eos_id,
                tokens_written=total_tokens,
                max_tokens=max_tokens,
            )

            any_truncated = any_truncated or doc_truncated

            if limit_reached and not ids:
                output_capped = True
                break

            write_buffer.extend(ids)
            total_tokens += len(ids)
            docs_written += 1

            progress_bar.update(len(ids))
            if len(write_buffer) >= chunk_size * 256:
                flush_buffer(write_buffer, out_file, dtype)
                write_buffer = []

            if limit_reached:
                output_capped = True
                break

        if write_buffer:
            flush_buffer(write_buffer, out_file, dtype)

    return {
        "total_tokens": total_tokens,
        "documents_seen": docs_seen,
        "documents_written": docs_written,
        "output_capped": output_capped,
        "document_truncated": any_truncated,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)

    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--dataset_name", type=str)
    source.add_argument("--input_file", type=str)

    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--streaming", action="store_true", default=None)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, required=False)

    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--max_bin_size_gb", type=float, default=None)

    return parser.parse_args()


def merge_config_and_args(args, cfg):
    data = cfg.copy()

    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            data[key] = value

    if not data.get("dataset_name") and not data.get("input_file"):
        raise ValueError("Provide either dataset_name or input_file.")

    if not data.get("output_prefix"):
        raise ValueError("output_prefix is required.")

    return data


def main():
    setup_logging()
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = merge_config_and_args(args, cfg)

    if cfg["max_bin_size_gb"] <= 0:
        raise ValueError("max_bin_size_gb must be greater than 0.")

    logger.info("Loading tokenizer: %s", cfg["tokenizer_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        logger.warning("Tokenizer had no EOS token; added '<|endoftext|>'.")

    tokenizer_size = len(tokenizer)
    dtype = dtype_for_vocab_size(tokenizer_size)
    dtype_bytes = np.dtype(dtype).itemsize

    
    max_bin_size_bytes = int(cfg["max_bin_size_gb"] * (1024**3))
    max_tokens = max_bin_size_bytes // dtype_bytes

    if max_tokens <= 0:
        raise ValueError("max_bin_size_gb is too small for selected dtype.")

    split = cfg.get("split", "train")
    output_prefix = cfg["output_prefix"]

    bin_path = str(Path.joinpath(Path(output_prefix), "data.bin"))
    meta_path = str(Path.joinpath(Path(output_prefix), "data.json"))

    Path(output_prefix).mkdir(parents=True, exist_ok=True)

    logger.info("Output binary: %s", bin_path)
    logger.info("Output metadata: %s", meta_path)
    logger.info("Storage dtype: %s", dtype.__name__)
    logger.info("Maximum tokens: %s", f"{max_tokens:,}")

    preprocessing = cfg.get("preprocessing", {})

    if cfg.get("dataset_name"):
        from datasets import load_dataset
        import datasets

        logger.info(
            "Loading HF dataset: %s | config=%s | split=%s",
            cfg["dataset_name"],
            cfg.get("dataset_config"),
            split,
        )

        load_kwargs = {
            "split": split,
            "streaming": cfg.get("streaming", False),
            "trust_remote_code": True,
        }

        if cfg.get("dataset_config"):
            dataset = load_dataset(
                cfg["dataset_name"],
                cfg["dataset_config"],
                **load_kwargs,
            )
        else:
            dataset = load_dataset(
                cfg["dataset_name"],
                **load_kwargs,
            )

        if cfg.get("max_samples") is not None:
            max_samples = cfg["max_samples"]
            if cfg.get("streaming", False):
                dataset = dataset.take(max_samples)
            else:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

        stats = tokenize_iterator(
            iterator=dataset,
            tokenizer=tokenizer,
            output_path=bin_path,
            dtype=dtype,
            text_getter=lambda x: x[cfg.get("text_column", "text")],
            chunk_size=cfg.get("chunk_size", 4096),
            max_tokens=max_tokens,
            preprocessing=preprocessing,
        )

        datasets_version = datasets.__version__

    else:
        input_file = cfg["input_file"]

        if not os.path.exists(input_file):
            sys.exit(f"Input file not found: {input_file}")

        with open(input_file, "r", encoding="utf-8") as f:
            stats = tokenize_iterator(
                iterator=f,
                tokenizer=tokenizer,
                output_path=bin_path,
                dtype=dtype,
                text_getter=lambda line: line.rstrip("\n"),
                chunk_size=cfg.get("chunk_size", 4096),
                max_tokens=max_tokens,
                preprocessing=preprocessing,
            )

        datasets_version = None

    file_size_mb = os.path.getsize(bin_path) / (1024**2)

    metadata = {
        "source": {
            "dataset_name": cfg.get("dataset_name"),
            "dataset_config": cfg.get("dataset_config"),
            "input_file": cfg.get("input_file"),
            "split": split,
            "text_column": cfg.get("text_column"),
            "streaming": cfg.get("streaming", False),
            "max_samples": cfg.get("max_samples"),
        },
        "tokenizer": {
            "tokenizer_name": cfg["tokenizer_name"],
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_length": len(tokenizer),
        },
        "preprocessing": preprocessing,
        "storage": {
            "bin_path": bin_path,
            "metadata_path": meta_path,
            "dtype": dtype.__name__,
            "file_size_mb": round(file_size_mb, 2),
            "max_bin_size_gb": cfg["max_bin_size_gb"],
            "max_bin_size_bytes": max_bin_size_bytes,
        },
        "stats": stats,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets_version,
        },
        "command": " ".join(sys.argv),
        "format": "Flat binary array of token IDs loadable with numpy.memmap.",
        "load_hint": f"np.memmap('{bin_path}', dtype='{dtype.__name__}', mode='r')",
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Completed dataset preparation.")
    logger.info("Tokens written: %s", f"{stats['total_tokens']:,}")
    logger.info("Documents written: %s", f"{stats['documents_written']:,}")
    logger.info("Binary size: %.2f MB", file_size_mb)


if __name__ == "__main__":
    main()