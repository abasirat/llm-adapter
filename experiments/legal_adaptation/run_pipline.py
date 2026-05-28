# run_pipeline.py

import argparse
import itertools
import subprocess
import yaml
from pathlib import Path
from datetime import datetime


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_cmd(cmd, log_file):
    print("\nRunning:")
    print(" ".join(cmd))

    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(f"Command failed. See log: {log_file}")


def stage_done(path):
    return Path(path).exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_name = cfg["experiment_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    models = cfg["models"]
    adapters = cfg["adapters"]
    seeds = cfg.get("seeds", [42])

    for model_cfg, adapter_cfg, seed in itertools.product(models, adapters, seeds):
        model_name = Path(model_cfg).stem
        adapter_name = Path(adapter_cfg).stem

        run_name = f"{experiment_name}_{model_name}_{adapter_name}_seed{seed}"
        run_dir = output_root / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        log_dir = run_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        resolved_config_path = run_dir / "resolved_config.yaml"

        resolved_cfg = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "seed": seed,
            "model_config": model_cfg,
            "adapter_config": adapter_cfg,
            "data_config": cfg["data_config"],
            "training_config": cfg["training_config"],
            "eval_config": cfg["eval_config"],
            "run_dir": str(run_dir),
            "created_at": timestamp,
        }

        with open(resolved_config_path, "w") as f:
            yaml.safe_dump(resolved_cfg, f, sort_keys=False)

        print(f"\n=== Run: {run_name} ===")

        # 1. Prepare data
        data_done = run_dir / "data.done"
        if args.force or not stage_done(data_done):
            cmd = [
                "python", "scripts/prepare_data.py",
                "--config", cfg["data_config"],
                "--output_dir", str(run_dir / "data"),
            ]

            if args.dry_run:
                print(" ".join(cmd))
            else:
                run_cmd(cmd, log_dir / "prepare_data.log")
                data_done.touch()

        # 2. Train
        train_done = run_dir / "train.done"
        checkpoint_dir = run_dir / "checkpoint"

        if args.force or not stage_done(train_done):
            cmd = [
                "python", "scripts/train.py",
                "--model_config", model_cfg,
                "--adapter_config", adapter_cfg,
                "--training_config", cfg["training_config"],
                "--data_dir", str(run_dir / "data"),
                "--output_dir", str(checkpoint_dir),
                "--seed", str(seed),
            ]

            if args.dry_run:
                print(" ".join(cmd))
            else:
                run_cmd(cmd, log_dir / "train.log")
                train_done.touch()

        # 3. Evaluate
        eval_done = run_dir / "eval.done"

        if args.force or not stage_done(eval_done):
            cmd = [
                "python", "scripts/evaluate.py",
                "--eval_config", cfg["eval_config"],
                "--checkpoint_dir", str(checkpoint_dir),
                "--output_dir", str(run_dir / "eval"),
                "--seed", str(seed),
            ]

            if args.dry_run:
                print(" ".join(cmd))
            else:
                run_cmd(cmd, log_dir / "eval.log")
                eval_done.touch()

    # 4. Collect results
    collect_cmd = [
        "python", "scripts/collect_results.py",
        "--runs_dir", str(output_root / "runs"),
        "--output_dir", str(output_root / "results"),
    ]

    # 5. Generate plots
    plot_cmd = [
        "python", "scripts/plot_results.py",
        "--results_dir", str(output_root / "results"),
        "--output_dir", str(output_root / "figures"),
    ]

    if args.dry_run:
        print(" ".join(collect_cmd))
        print(" ".join(plot_cmd))
    else:
        run_cmd(collect_cmd, output_root / "collect_results.log")
        run_cmd(plot_cmd, output_root / "plot_results.log")


if __name__ == "__main__":
    main()