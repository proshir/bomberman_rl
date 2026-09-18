"""Evaluate selected 400-step history-agent checkpoints on held-out boards."""

import hashlib
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from run_benchmark import SOURCE_DIR, save_json

BOARD_SEEDS = list(range(22016, 22032))
TRAINING_SEEDS = [0, 1, 2]
CHECKPOINT_ROUNDS = [300, 400, 500]
STEP_BUDGETS = [100, 400]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(pilot, seed, episode):
    candidates = [
        pilot / f"seed_{seed}/seed_{seed}/checkpoints/episode_{episode:04d}.pkl",
        pilot / f"seed_{seed}/checkpoints/episode_{episode:04d}.pkl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing checkpoint for seed {seed}, round {episode}")


def run_policy(checkpoint, max_steps, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(SOURCE_DIR / "run_benchmark.py"),
        "--agents", "tree_fqi_history_agent", "--model-path", str(checkpoint),
        "--seeds", *map(str, BOARD_SEEDS), "--max-steps", str(max_steps),
        "--agent-seeds", "0", "--seats", "0", "1", "2", "3",
        "--batch-size", str(4 * len(BOARD_SEEDS)), "--output", str(output),
    ]
    with open(output.with_suffix(".log"), "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(output / "summary.json") as file:
        return json.load(file)["agents"]["tree_fqi_history_agent"]


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output directory.")
    args.output.mkdir(parents=True)

    rows = []
    for episode in CHECKPOINT_ROUNDS:
        for seed in TRAINING_SEEDS:
            checkpoint = checkpoint_path(args.pilot, seed, episode)
            checkpoint_hash = file_hash(checkpoint)
            for budget in STEP_BUDGETS:
                directory = args.output / f"steps_{budget:04d}" / f"round_{episode:04d}" / f"seed_{seed}"
                summary = run_policy(checkpoint, budget, directory)
                if file_hash(checkpoint) != checkpoint_hash:
                    raise RuntimeError(f"Checkpoint changed during evaluation: {checkpoint}")
                rows.append({
                    "training_seed": seed,
                    "checkpoint_round": episode,
                    "max_steps": budget,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_hash": checkpoint_hash,
                    **summary,
                })
                print(f"round {episode}, seed {seed}, {budget} steps: "
                      f"{summary['mean']:.2f} coins", flush=True)

    report = {"status": "held-out frozen checkpoint comparison", "rows": rows, "aggregates": {}}
    for episode in CHECKPOINT_ROUNDS:
        for budget in STEP_BUDGETS:
            selected = [row for row in rows if row["checkpoint_round"] == episode and
                        row["max_steps"] == budget]
            report["aggregates"][f"round_{episode:04d}/steps_{budget:04d}"] = {
                "mean_coins": float(np.mean([row["mean"] for row in selected])),
                "run_means": [row["mean"] for row in selected],
                "completion_rate": float(np.mean([row["completion_rate"] for row in selected])),
                "mean_repeated_states": float(np.mean([row["mean_repeated_states"] for row in selected])),
                "mean_invalid_actions": float(np.mean([row["mean_invalid_actions"] for row in selected])),
                "mean_self_deaths": float(np.mean([row["mean_suicides"] for row in selected])),
            }
    save_json(args.output / "summary.json", report)
    print(json.dumps(report["aggregates"], indent=2))


if __name__ == "__main__":
    main()
