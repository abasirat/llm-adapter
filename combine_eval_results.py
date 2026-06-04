#!/usr/bin/env python3
"""
combine_eval_results.py – Compare evaluation results across multiple models.

Reads the JSON files produced by run_all_evaluations.py and prints a single
aligned table with metrics as rows and models as columns.

Usage:
    python combine_eval_results.py results/modelA_eval.json results/modelB_eval.json ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _flatten_summary(data: dict) -> Dict[str, Optional[float]]:
    """Return {evaluator.metric: value} from a JSON output file."""
    flat: Dict[str, Optional[float]] = {}

    # Build alias→subset lookup from embedded config (perplexity only)
    ppl_subset: Dict[str, str] = {}
    for ds_spec in (
        data.get("config", {})
            .get("evaluations", {})
            .get("perplexity", {})
            .get("datasets", []) or []
    ):
        alias = ds_spec.get("name", "")
        subset = ds_spec.get("subset")
        if alias and subset:
            ppl_subset[alias] = subset

    for evaluator, block in data.get("results", {}).items():
        if block.get("status") == "failed":
            flat[f"{evaluator}.__error__"] = None
            continue
        for metric, value in block.get("summary", {}).items():
            if evaluator == "perplexity" and ppl_subset:
                for alias, subset in ppl_subset.items():
                    if metric.startswith(alias + "_"):
                        metric = f"{alias}[{subset}]{metric[len(alias):]}"
                        break
            flat[f"{evaluator}.{metric}"] = value
    return flat


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _short_label(model_path: str) -> str:
    """Return the bare filename without extension as a display label."""
    p = Path(model_path)
    name = p.name
    # Strip common checkpoint suffixes
    for suffix in ("-best", "-trace", ".pt", ".bin", ".safetensors"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _format_value(v) -> str:
    if v is None:
        return "ERROR"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_table(
    metric_keys: List[str],
    model_labels: List[str],
    rows: Dict[str, List],
    col_width: int = 10,
) -> None:
    metric_col = max((len(k) for k in metric_keys), default=30) + 2

    # Header
    header = f"{'Metric':<{metric_col}}" + "".join(
        f"{label[:col_width - 1]:>{col_width}}" for label in model_labels
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for key in metric_keys:
        cells = rows[key]
        line = f"{key:<{metric_col}}" + "".join(
            f"{_format_value(v):>{col_width}}" for v in cells
        )
        print(line)

    print(separator)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine and compare evaluation JSON results across models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        metavar="FILE",
        help="One or more JSON files produced by run_all_evaluations.py.",
    )
    parser.add_argument(
        "--col_width",
        type=int,
        default=12,
        help="Width of each value column.",
    )
    args = parser.parse_args()

    # Load all files
    records: List[dict] = []
    model_paths: List[str] = []
    for path in args.result_files:
        try:
            data = _load(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
            continue
        records.append(data)
        model_paths.append(data.get("model", path))

    if not records:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    # Collect union of all metric keys (preserve insertion order)
    all_keys: List[str] = []
    seen = set()
    for rec in records:
        for k in _flatten_summary(rec):
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Build per-row value lists
    rows: Dict[str, list] = {k: [] for k in all_keys}
    for rec in records:
        flat = _flatten_summary(rec)
        for k in all_keys:
            rows[k].append(flat.get(k))

    # Print legend (full model names)
    labels = [f"[{i+1}]" for i in range(len(model_paths))]
    print("\nModel legend:")
    for label, path in zip(labels, model_paths):
        print(f"  {label}  {path}")

    print("\nEvaluation Summary")
    _print_table(all_keys, labels, rows, col_width=args.col_width)


if __name__ == "__main__":
    main()
