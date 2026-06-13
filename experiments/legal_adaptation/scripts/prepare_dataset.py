"""
Prepare a causal-LM dataset as a flat binary token-ID file.

The dataset is tokenised and packed into a single binary file containing a flat array of token IDs.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
import transformers

from evaluations.ledgar.evaluator import _build_prompt as _ledgar_build_prompt
from evaluations.casehold.evaluator import _build_prompt as _casehold_build_prompt

from huggingface_hub.errors import HfHubHTTPError  # to catch potential HTTP errors when streaming datasets from Hugging Face Hub

import datetime # used for filtering operations defined in the data config file



logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
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

def clean_text(text, cleaning_rules):
    if text is None:
        return None
    regex_substitutions = cleaning_rules.get("regex_substitutions", [])
    for rule in regex_substitutions:
        logger.debug("Applying cleaning rule: %s", rule)
        pattern = rule.get("pattern")
        replacement = rule.get("replacement", "")
        logger.debug("Pattern: %s | Replacement: %s", pattern, replacement)
        logger.debug("Original text sample: %s", text[:200])
        if pattern is not None:
            text = re.sub(pattern, replacement, text)
        logger.debug("Cleaned text sample: %s", text[:200])
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
    cleaning=None,
    text_path=None,
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

    text_out_file = open(text_path, "w", encoding="utf-8") if text_path else None

    progress_bar = tqdm(desc="Tokenizing", unit="doc", total=max_tokens)
    with open(output_path, "wb") as out_file:
        try:
            for sample in iterator:
                docs_seen += 1

                text = text_getter(sample)
                text = preprocess_text(text, **preprocessing)

                if cleaning:
                    text = clean_text(text, cleaning)

                if not text:
                    continue

                if text_out_file:
                    text_out_file.write(text)

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

                if len(write_buffer) >= chunk_size * 256:
                    flush_buffer(write_buffer, out_file, dtype)
                    progress_bar.update(len(write_buffer))
                    write_buffer = []

                if limit_reached:
                    output_capped = True
                    break

        except TimeoutError:
            logger.warning("Tokenization timed out. Ignore the current sample and continue.")
        except HfHubHTTPError as e:
            logger.warning("Hugging Face Hub HTTP error during tokenization: %s. Ignore the current sample and continue.", str(e))
        except RuntimeError as e:
            logger.warning("Runtime error during tokenization (possibly due to a very long document): %s. Ignore the current sample and continue.", str(e))



        if write_buffer:
            flush_buffer(write_buffer, out_file, dtype)
            progress_bar.update(len(write_buffer))

    return {
        "total_tokens": total_tokens,
        "documents_seen": docs_seen,
        "documents_written": docs_written,
        "output_capped": output_capped,
        "document_truncated": any_truncated,
    }


# ---------------------------------------------------------------------------
# Task-specific text formatters
# ---------------------------------------------------------------------------

def _apply_task_formatter(dataset, task: str, tokenizer):
    """Return a text_getter callable for a named evaluation task.

    The returned function accepts a single dataset example and returns the
    formatted string that will be tokenised and packed into the binary file.
    """
    if task == "casehold":
        def fmt(ex):
            choice = ex["endings"][int(ex["label"])]
            return _casehold_build_prompt(ex["context"], choice)
        return fmt

    if task == "ledgar":
        label_names = dataset.features["label"].names
        def fmt(ex, _ln=label_names):
            choice = _ln[ex["label"]].replace("_", " ").replace("-", " ")
            return _ledgar_build_prompt(ex["text"], choice)
        return fmt

    if task == "unfair_tos":
        return lambda ex: f"Terms of service clause: {ex['text'].strip()[:1200]}"

    raise ValueError(
        f"Unknown task formatter: {task!r}. Supported tasks: casehold, ledgar, unfair_tos"
    )


def _build_text_getter(cfg: dict, dataset, tokenizer):
    """Return the text_getter function based on task / text_template / text_column."""
    task = cfg.get("task")
    if task:
        return _apply_task_formatter(dataset, task, tokenizer)
    text_template = cfg.get("text_template")
    if text_template:
        return lambda x, _t=text_template: _t.format(**x)
    col = cfg.get("text_column", "text")
    return lambda x: x[col]


# ---------------------------------------------------------------------------
# Core dataset preparation (importable)
# ---------------------------------------------------------------------------

def run_prepare_dataset(cfg: dict) -> None:
    """Tokenise and binarise a dataset according to *cfg*.

    Reads ``cfg`` keys identical to the CLI / YAML config format.
    This function is importable so that other scripts (e.g. post_train_task.py)
    can call it directly without spawning a subprocess.
    """
    if not cfg.get("max_bin_size_gb"):
        raise ValueError("max_bin_size_gb must be set and greater than 0.")
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
    output_dir = cfg["output_dir"]

    bin_path = str(Path(output_dir) / "data.bin")
    meta_path = str(Path(output_dir) / "data.json")
    text_path = str(Path(output_dir) / "data.txt") if cfg.get("save_text", False) else None

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Output binary: %s", bin_path)
    logger.info("Output metadata: %s", meta_path)
    logger.info("Storage dtype: %s", dtype.__name__)
    logger.info("Maximum tokens: %s", f"{max_tokens:,}")

    preprocessing = cfg.get("preprocessing", {})
    cleaning = cfg.get("cleaning", {})

    if cfg.get("dataset_name"):
        from datasets import load_dataset
        import datasets as _datasets_mod

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
            dataset = load_dataset(cfg["dataset_name"], **load_kwargs)

        if cfg.get("max_samples") is not None:
            max_samples = cfg["max_samples"]
            if cfg.get("streaming", False):
                dataset = dataset.take(max_samples)
            else:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        if cfg.get("filters"):
            for filter_cfg in cfg["filters"]:
                filter_name = filter_cfg.get("name", "unnamed_filter")
                operation = filter_cfg["operation"]
                logger.info("Applying filter: %s | operation: %s", filter_name, operation)
                dataset = dataset.filter(eval(operation))

        text_getter = _build_text_getter(cfg, dataset, tokenizer=tokenizer)

        stats = tokenize_iterator(
            iterator=dataset,
            tokenizer=tokenizer,
            output_path=bin_path,
            dtype=dtype,
            text_getter=text_getter,
            chunk_size=cfg.get("chunk_size", 4096),
            max_tokens=max_tokens,
            preprocessing=preprocessing,
            cleaning=cleaning,
            text_path=text_path,
        )

        datasets_version = _datasets_mod.__version__

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
                cleaning=cleaning,
                text_path=text_path,
            )

        datasets_version = None

    file_size_mb = os.path.getsize(bin_path) / (1024**2)

    metadata = {
        "source": {
            "dataset_name": cfg.get("dataset_name"),
            "dataset_config": cfg.get("dataset_config"),
            "input_file": cfg.get("input_file"),
            "split": split,
            "task": cfg.get("task"),
            "text_column": cfg.get("text_column"),
            "text_template": cfg.get("text_template"),
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
        "cleaning": cleaning,
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
    parser.add_argument("--output_dir", type=str, required=False)

    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--max_bin_size_gb", type=float, default=None)

    parser.add_argument("--progress_interval", type=int, default=10, help="Log progress every N documents.")

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

    if not data.get("output_dir"):
        raise ValueError("output_dir is required.")

    return data


def main():
    setup_logging(logging.INFO)
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = merge_config_and_args(args, cfg)
    run_prepare_dataset(cfg)


if __name__ == "__main__":
    main()