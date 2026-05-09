#!/usr/bin/env python3
"""
run_all_evaluations.py – Central evaluation runner.

Imports each evaluator module, calls its evaluate() function, aggregates the
results, and prints a clean summary table.  If one evaluator fails the error
is recorded but execution continues for the remaining evaluators.

Usage:
    python run_all_evaluations.py \\
        --model_name_or_path gpt2 \\
        --device cpu \\
        --max_examples 100 \\
        --output_file results.json

Extending the framework
-----------------------
To add a new benchmark:
  1. Create  evaluations/<name>/evaluator.py  with an  evaluate()  function.
  2. Register it in EVALUATORS below.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Evaluator registry
# Each entry is (display_name, import_path).  The import_path must point to a
# module that exposes  evaluate(model_name_or_path, device, **kwargs) -> dict.
# ---------------------------------------------------------------------------

EVALUATORS: List[Tuple[str, str]] = [
    ("truthfulqa", "evaluations.truthfulqa.evaluator"),
    ("lambada",    "evaluations.lambada.evaluator"),
    ("style",      "evaluations.style.evaluator"),
    ("perplexity", "evaluations.perplexity.evaluator"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_evaluator(import_path: str):
    """Dynamically import an evaluator module and return it."""
    import importlib
    return importlib.import_module(import_path)


def _run_evaluator(
    name: str,
    import_path: str,
    model_name_or_path: str,
    device: str,
    kwargs: dict,
) -> Dict[str, Any]:
    """
    Run a single evaluator safely.

    Returns a dict whose keys are either metric names or "__error__" if the
    evaluator raised an exception.
    """
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    try:
        module = _import_evaluator(import_path)
        result = module.evaluate(
            model_name_or_path=model_name_or_path,
            device=device,
            **kwargs,
        )
        return result
    except Exception:
        error_msg = traceback.format_exc()
        print(f"[ERROR] Evaluator '{name}' failed:\n{error_msg}", file=sys.stderr)
        return {"__error__": error_msg}


def _flatten_for_display(name: str, metrics: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """
    Return a list of (key, value) pairs suitable for the summary table.

    Keys prefixed with "_" (e.g. "_raw") are internal and excluded from display.
    """
    if "__error__" in metrics:
        return [(f"{name}.__error__", "FAILED (see stderr)")]

    rows = []
    for k, v in metrics.items():
        if k.startswith("_"):
            continue  # Internal keys
        display_key = f"{name}.{k}"
        if isinstance(v, float):
            rows.append((display_key, f"{v:.4f}"))
        elif v is None:
            rows.append((display_key, "N/A"))
        else:
            rows.append((display_key, str(v)))

    return rows


def _print_summary(model_name_or_path: str, all_metrics: Dict[str, Dict]) -> None:
    """Print the final summary table to stdout."""
    rows: List[Tuple[str, Any]] = []
    for name, metrics in all_metrics.items():
        rows.extend(_flatten_for_display(name, metrics))

    key_width = max((len(k) for k, _ in rows), default=30) + 2

    print(f"\nModel: {model_name_or_path}")
    print()
    print("Evaluation Summary")
    print("-" * (key_width + 15))
    for key, val in rows:
        print(f"  {key:<{key_width}}{val}")
    print("-" * (key_width + 15))


def _prepare_json_output(
    model_name_or_path: str,
    all_metrics: Dict[str, Dict],
) -> dict:
    """
    Build the JSON-serialisable output dict.

    Per-example results (_raw) are included only in the JSON output, not in
    the terminal summary.
    """
    output = {
        "model": model_name_or_path,
        "results": {},
    }

    for name, metrics in all_metrics.items():
        if "__error__" in metrics:
            output["results"][name] = {"status": "failed", "error": metrics["__error__"]}
        else:
            # Scalars for top-level summary.
            scalar = {k: v for k, v in metrics.items() if not k.startswith("_")}
            # Full per-example results, if the evaluator returned them.
            raw = metrics.get("_raw", {})
            output["results"][name] = {"summary": scalar, "raw": raw}

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all causal LM evaluations and print a summary table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model name (e.g. gpt2) or local adapter checkpoint path.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Torch device: cuda, mps, or cpu.  "
            "Defaults to auto-detection (cuda > mps > cpu)."
        ),
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on examples per benchmark (useful for quick tests).",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="If given, save the full JSON results to this path.",
    )

    # Allow users to skip individual evaluators.
    parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=[],
        metavar="NAME",
        help="Names of evaluators to skip (e.g. --skip legal_style perplexity).",
    )

    parser.add_argument(
        "--legal_prompts_path",
        type=str,
        default=None,
        help="Path to a .txt/.jsonl file or directory of prompt files for the legal style condition.",
    )

    parser.add_argument(
        "--general_prompts_path",
        type=str,
        default=None,
        help="Path to a .txt/.jsonl file or directory of prompt files for the general style condition.",
    )

    parser.add_argument(
        "--style_min_new_tokens",
        type=int,
        default=None,
        help="Minimum tokens to generate per prompt in style evaluation (default: 20).",
    )

    parser.add_argument(
        "--style_max_new_tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate per prompt in style evaluation (default: 80).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve device (auto-detect if not specified).
    from evaluations.utils import select_device, safe_save_json
    device = select_device(args.device)
    print(f"Device: {device}")

    # Kwargs forwarded to every evaluator.
    shared_kwargs = {}
    if args.max_examples is not None:
        shared_kwargs["max_examples"] = args.max_examples
    if args.legal_prompts_path is not None:
        shared_kwargs["legal_prompts_path"] = args.legal_prompts_path
    if args.general_prompts_path is not None:
        shared_kwargs["general_prompts_path"] = args.general_prompts_path
    if args.style_min_new_tokens is not None:
        shared_kwargs["min_new_tokens"] = args.style_min_new_tokens
    if args.style_max_new_tokens is not None:
        shared_kwargs["max_new_tokens"] = args.style_max_new_tokens

    all_metrics: Dict[str, Dict] = {}

    for name, import_path in EVALUATORS:
        if name in args.skip:
            print(f"\n[skip] {name}")
            continue

        all_metrics[name] = _run_evaluator(
            name=name,
            import_path=import_path,
            model_name_or_path=args.model_name_or_path,
            device=device,
            kwargs=shared_kwargs,
        )

    # Terminal summary.
    _print_summary(args.model_name_or_path, all_metrics)

    # Optional JSON output.
    if args.output_file:
        output = _prepare_json_output(args.model_name_or_path, all_metrics)
        safe_save_json(output, args.output_file)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
