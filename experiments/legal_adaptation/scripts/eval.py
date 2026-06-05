#!/usr/bin/env python3
"""
eval.py – Config-driven central evaluation runner.

Runs a configurable suite of causal-LM evaluations. Each evaluation has its own
section under `evaluations:` and can be enabled/disabled independently.

Example:
    python scripts/eval.py \
        --config configs/evaluation/legal_eval.yaml

CLI values override YAML values for common options and evaluation enable flags.
There is intentionally no --skip option; use `enabled: true/false` in YAML or
`--enable-<name>` / `--disable-<name>` from the CLI.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required. Install it with `pip install pyyaml`.") from exc


# ---------------------------------------------------------------------------
# Evaluator registry
# Each evaluator module must expose:
#   evaluate(model_name_or_path: str, device: str, **kwargs) -> dict
# ---------------------------------------------------------------------------

EVALUATORS: List[Tuple[str, str]] = [
    ("truthfulqa", "evaluations.truthfulqa.evaluator"),
    ("lambada", "evaluations.lambada.evaluator"),
    ("style", "evaluations.style.evaluator"),
    ("perplexity", "evaluations.perplexity.evaluator"),
    ("casehold", "evaluations.casehold.evaluator"),
    ("ledgar", "evaluations.ledgar.evaluator"),
    ("unfair_tos", "evaluations.unfair_tos.evaluator"),
]

EVALUATOR_NAMES = [name for name, _ in EVALUATORS]


# ---------------------------------------------------------------------------
# YAML/config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively update mapping `base` with non-None values from `update`."""
    for key, value in update.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def compact_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def normalise_style_keys(eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Allow YAML-friendly aliases for the style evaluator."""
    eval_cfg = dict(eval_cfg)

    # User-facing config used `classifier_path`; evaluator expects `clf_path`.
    if "classifier_path" in eval_cfg and "clf_path" not in eval_cfg:
        eval_cfg["clf_path"] = eval_cfg.pop("classifier_path")

    return eval_cfg


def post_train_for_eval(
    model_name_or_path: str,
    device: str,
    post_training_cfg: Dict[str, Any],
) -> str:
    """Fine-tune adapter parameters for a task and return the checkpoint path.

    Checks for an existing checkpoint first (respects ``force_retrain``).
    Lazily imports ``post_train_task`` so the heavy training dependencies are
    only loaded when post-training is actually requested.
    """
    output_dir = post_training_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("post_training.output_dir must be specified.")

    best_checkpoint = os.path.join(output_dir, "checkpoint-best")
    force_retrain   = post_training_cfg.get("force_retrain", False)

    if not force_retrain and os.path.exists(best_checkpoint):
        print(f"\n[post_training] Reusing existing checkpoint: {best_checkpoint}")
        return best_checkpoint

    training_config_path = post_training_cfg.get("training_config")
    data_config_path     = post_training_cfg.get("data_config")

    if not training_config_path:
        raise ValueError("post_training.training_config must be specified.")
    if not data_config_path:
        raise ValueError("post_training.data_config must be specified.")

    training_cfg = load_yaml(training_config_path)
    data_cfg     = load_yaml(data_config_path)

    # Make post_train_task importable (lives in the same scripts/ directory).
    _scripts_dir = str(Path(__file__).parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)

    from post_train_task import run_post_training  # noqa: PLC0415

    return run_post_training(
        model_path=model_name_or_path,
        training_cfg=training_cfg,
        data_cfg=data_cfg,
        output_dir=output_dir,
        device=device,
    )


def build_evaluator_kwargs(
    eval_name: str,
    eval_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Build kwargs passed to a specific evaluator."""
    kwargs = copy.deepcopy(eval_cfg)
    kwargs.pop("enabled", None)
    kwargs.pop("import_path", None)
    kwargs.pop("post_training", None)  # consumed by eval.py; not forwarded to evaluators

    # Global max_examples is inherited unless the evaluation has its own value.
    if "max_examples" not in kwargs and global_cfg.get("max_examples") is not None:
        kwargs["max_examples"] = global_cfg["max_examples"]

    # Optional global generation defaults are useful for generation-based evals.
    generation_cfg = global_cfg.get("generation", {}) or {}
    if eval_name == "style":
        kwargs = normalise_style_keys(kwargs)
        for key in ("temperature", "top_p", "do_sample", "max_new_tokens"):
            if key not in kwargs and key in generation_cfg:
                kwargs[key] = generation_cfg[key]

    return kwargs


# ---------------------------------------------------------------------------
# Evaluator execution
# ---------------------------------------------------------------------------

def import_evaluator(import_path: str):
    return importlib.import_module(import_path)


def run_evaluator(
    name: str,
    import_path: str,
    model_name_or_path: str,
    device: str,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    print(f"\n{'=' * 70}")
    print(f"Running evaluation: {name}")
    print(f"Module: {import_path}")
    print(f"Kwargs: {json.dumps(kwargs, indent=2, default=str)}")
    print(f"{'=' * 70}")

    try:
        module = import_evaluator(import_path)
        result = module.evaluate(
            model_name_or_path=model_name_or_path,
            device=device,
            **kwargs,
        )
        if not isinstance(result, dict):
            raise TypeError(f"Evaluator '{name}' returned {type(result)}, expected dict.")
        return result
    except Exception:
        error_msg = traceback.format_exc()
        print(f"[ERROR] Evaluation '{name}' failed:\n{error_msg}", file=sys.stderr)
        return {"__error__": error_msg}


def flatten_for_display(name: str, metrics: Dict[str, Any]) -> List[Tuple[str, Any]]:
    if "__error__" in metrics:
        return [(f"{name}.__error__", "FAILED")]

    rows: List[Tuple[str, Any]] = []
    for k, v in metrics.items():
        if k.startswith("_"):
            continue
        key = f"{name}.{k}"
        if isinstance(v, float):
            rows.append((key, f"{v:.4f}"))
        elif isinstance(v, (int, str, bool)) or v is None:
            rows.append((key, v))
        else:
            # Avoid dumping huge nested values into the terminal table.
            rows.append((key, f"<{type(v).__name__}>"))
    return rows


def print_summary(model_name_or_path: str, all_metrics: Dict[str, Dict[str, Any]]) -> None:
    rows: List[Tuple[str, Any]] = []
    for name, metrics in all_metrics.items():
        rows.extend(flatten_for_display(name, metrics))

    key_width = max((len(k) for k, _ in rows), default=30) + 2

    print(f"\nModel: {model_name_or_path}\n")
    print("Evaluation Summary")
    print("-" * (key_width + 20))
    for key, value in rows:
        print(f"  {key:<{key_width}}{value}")
    print("-" * (key_width + 20))


def prepare_json_output(
    model_name_or_path: str,
    device: str,
    config: Dict[str, Any],
    all_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "device": device,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "results": {},
    }

    for name, metrics in all_metrics.items():
        if "__error__" in metrics:
            output["results"][name] = {
                "status": "failed",
                "error": metrics["__error__"],
            }
        else:
            summary = {k: v for k, v in metrics.items() if not k.startswith("_")}
            raw = metrics.get("_raw", {})
            output["results"][name] = {
                "status": "ok",
                "summary": summary,
                "raw": raw,
            }

    return output


def write_summary_csv(path: str, all_metrics: Dict[str, Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["evaluation", "metric", "value"])
        writer.writeheader()
        for eval_name, metrics in all_metrics.items():
            if "__error__" in metrics:
                writer.writerow({"evaluation": eval_name, "metric": "status", "value": "failed"})
                continue
            for metric, value in metrics.items():
                if metric.startswith("_"):
                    continue
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False)
                writer.writerow({"evaluation": eval_name, "metric": metric, "value": value})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a config-driven evaluation suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None, help="Path to YAML evaluation config.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="HF model name or local checkpoint path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save the results (model, binarised data, and json outputs).")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, cuda, mps, or cpu.")
    parser.add_argument("--max_examples", type=int, default=None, help="Global example cap inherited by enabled evaluations.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional global batch size stored in config metadata.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed stored in config metadata.")

    # Per-evaluation enable/disable flags. These override YAML.
    for name in EVALUATOR_NAMES:
        parser.add_argument(
            f"--enable-{name}",
            dest=f"enable_{name}",
            action="store_true",
            default=None,
            help=f"Enable the {name} evaluation.",
        )
        parser.add_argument(
            f"--disable-{name}",
            dest=f"enable_{name}",
            action="store_false",
            help=f"Disable the {name} evaluation.",
        )

    # Common per-evaluator CLI overrides.
    parser.add_argument("--truthfulqa_split", type=str, default=None)
    parser.add_argument("--truthfulqa_normalize", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--lambada_split", type=str, default=None)

    parser.add_argument("--casehold_split", type=str, default=None)

    parser.add_argument("--ledgar_split", type=str, default=None)
    parser.add_argument("--ledgar_label_batch_size", type=int, default=None)
    parser.add_argument("--ledgar_max_length", type=int, default=None)

    parser.add_argument("--unfair_tos_split", type=str, default=None)

    parser.add_argument("--perplexity_languages", type=str, nargs="*", default=None)
    parser.add_argument("--perplexity_max_samples", type=int, default=None)

    parser.add_argument("--style_legal_prompts_path", type=str, default=None)
    parser.add_argument("--style_general_prompts_path", type=str, default=None)
    parser.add_argument("--style_clf_path", type=str, default=None)
    parser.add_argument("--style_classifier_path", type=str, default=None)
    parser.add_argument("--style_min_new_tokens", type=int, default=None)
    parser.add_argument("--style_max_new_tokens", type=int, default=None)
    parser.add_argument("--style_temperature", type=float, default=None)
    parser.add_argument("--style_top_p", type=float, default=None)
    parser.add_argument("--style_train_classifier", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--summary_csv", type=str, default=None, help="Optional CSV summary output path.")

    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    top_level_overrides = compact_none(
        {
            "model_name_or_path": args.model_name_or_path,
            "device": args.device,
            "max_examples": args.max_examples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    deep_update(cfg, top_level_overrides)

    cfg.setdefault("evaluations", {})

    # Boolean enable overrides.
    for name in EVALUATOR_NAMES:
        flag_value = getattr(args, f"enable_{name}")
        if flag_value is not None:
            cfg["evaluations"].setdefault(name, {})["enabled"] = flag_value

    # Per-evaluator option overrides.
    overrides: Dict[str, Dict[str, Any]] = {
        "truthfulqa": compact_none({"split": args.truthfulqa_split, "normalize": args.truthfulqa_normalize}),
        "lambada": compact_none({"split": args.lambada_split}),
        "casehold": compact_none({"split": args.casehold_split}),
        "ledgar": compact_none(
            {
                "split": args.ledgar_split,
                "label_batch_size": args.ledgar_label_batch_size,
                "max_length": args.ledgar_max_length,
            }
        ),
        "unfair_tos": compact_none({"split": args.unfair_tos_split}),
        "perplexity": compact_none(
            {
                "languages": args.perplexity_languages,
                "max_samples": args.perplexity_max_samples,
            }
        ),
        "style": compact_none(
            {
                "legal_prompts_path": args.style_legal_prompts_path,
                "general_prompts_path": args.style_general_prompts_path,
                "clf_path": args.style_clf_path or args.style_classifier_path,
                "min_new_tokens": args.style_min_new_tokens,
                "max_new_tokens": args.style_max_new_tokens,
                "temperature": args.style_temperature,
                "top_p": args.style_top_p,
                "train_classifier": args.style_train_classifier,
            }
        ),
    }

    for name, values in overrides.items():
        if values:
            cfg["evaluations"].setdefault(name, {})
            deep_update(cfg["evaluations"][name], values)

    if args.summary_csv is not None:
        cfg.setdefault("output", {})["summary_csv_path"] = args.summary_csv
        cfg.setdefault("output", {})["save_summary_csv"] = True

    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    if not cfg.get("model_name_or_path"):
        raise ValueError("model_name_or_path must be provided in YAML or via --model_name_or_path.")

    enabled = [
        name for name in EVALUATOR_NAMES
        if cfg.get("evaluations", {}).get(name, {}).get("enabled", False)
    ]
    if not enabled:
        raise ValueError(
            "No evaluations are enabled. Set evaluations.<name>.enabled: true "
            "or use --enable-<name>."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config) if args.config else {}
    cfg = apply_cli_overrides(cfg, args)
    validate_config(cfg)

    # Resolve device.
    from evaluations.utils import safe_save_json, select_device

    model_name_or_path = cfg["model_name_or_path"]
    output_file = os.path.join(args.output_dir, "results.json")
    device = select_device(cfg.get("device", "auto"))

    print(f"Device: {device}")
    print(f"Model:  {model_name_or_path}")
    print(f"Output file: {output_file}\n")

    all_metrics: Dict[str, Dict[str, Any]] = {}
    eval_cfgs = cfg.get("evaluations", {}) or {}

    for name, default_import_path in EVALUATORS:
        this_eval_cfg = copy.deepcopy(eval_cfgs.get(name, {}) or {})
        enabled = bool(this_eval_cfg.get("enabled", False))

        if not enabled:
            print(f"\n[disabled] {name}")
            continue

        import_path = this_eval_cfg.get("import_path", default_import_path)
        kwargs = build_evaluator_kwargs(name, this_eval_cfg, cfg)

        # Optional task-specific post-training before evaluation.
        eval_model_path = model_name_or_path
        post_cfg = (this_eval_cfg.get("post_training") or {})
        post_cfg["output_dir"] = post_cfg.get("output_dir") or os.path.join(args.output_dir, f"post_training_{name}")
        if post_cfg.get("enabled", False):
            print(f"\n[post_training] Starting post-training for evaluation: {name}")
            eval_model_path = post_train_for_eval(
                model_name_or_path=model_name_or_path,
                device=device,
                post_training_cfg=post_cfg,
            )

        all_metrics[name] = run_evaluator(
            name=name,
            import_path=import_path,
            model_name_or_path=eval_model_path,
            device=device,
            kwargs=kwargs,
        )

    print_summary(model_name_or_path, all_metrics)

    output_cfg = cfg.get("output", {}) or {}

    if output_file:
        ensure_parent_dir(output_file)
        output = prepare_json_output(
            model_name_or_path=model_name_or_path,
            device=device,
            config=cfg,
            all_metrics=all_metrics,
        )
        safe_save_json(output, output_file)
        print(f"\nResults saved to: {output_file}")

    if output_cfg.get("save_summary_csv", False):
        summary_csv_path = output_cfg.get("summary_csv_path")
        if summary_csv_path is None and output_file:
            summary_csv_path = str(Path(output_file).with_suffix(".summary.csv"))
        if summary_csv_path is None:
            raise ValueError("output.save_summary_csv=true requires output.summary_csv_path or top-level output_file.")
        write_summary_csv(summary_csv_path, all_metrics)
        print(f"Summary CSV saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()
