"""Plot frozen evaluations and per-round diagnostics from a combat pilot."""

# Sahand was here.

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path) as file:
        return json.load(file)


def load_jsonl(path):
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def rolling(values, window=25):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return np.arange(len(values)), values
    means = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.arange(window - 1, len(values)), means


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.experiment / "plots" / "training_overview.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    seeds = []
    curves = []
    rounds_by_seed = {}
    for directory in sorted(args.experiment.glob("seed_*")):
        if not directory.is_dir():
            continue
        seed = int(directory.name.split("_")[-1])
        curve = load_json(directory / "learning_curve.json")
        rounds = load_jsonl(directory / "rounds.jsonl")
        seeds.append(seed)
        curves.append(curve)
        rounds_by_seed[seed] = rounds

    if not curves:
        parser.error(f"No seed results found under {args.experiment}")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Frozen-policy evaluation curves.
    evaluation_axes = axes[0, 0]
    evaluation_by_seed = {}
    for index, (seed, curve) in enumerate(zip(seeds, curves)):
        episodes = [row["episode"] for row in curve]
        means = [row["mean"] for row in curve]
        evaluation_by_seed[seed] = means
        evaluation_axes.plot(episodes, means, marker="o", linewidth=2,
                             label=f"seed {seed}", color=colors[index])
    all_values = np.asarray(list(evaluation_by_seed.values()), dtype=float)
    evaluation_axes.plot(episodes, all_values.mean(axis=0), color="black",
                         marker="o", linewidth=2.5, label="mean")
    baseline = all_values[:, 0].mean()
    evaluation_axes.axhline(baseline, color="black", linestyle="--",
                            linewidth=1, label=f"round-0 baseline ({baseline:.2f})")
    evaluation_axes.set_title("Frozen-policy evaluation")
    evaluation_axes.set_xlabel("training round")
    evaluation_axes.set_ylabel("mean coins")
    evaluation_axes.legend(fontsize=8)

    # Per-round training coins.
    training_axes = axes[0, 1]
    for index, seed in enumerate(seeds):
        rounds = rounds_by_seed[seed]
        x, y = rolling([row["coins"] for row in rounds])
        training_axes.plot(x + 1, y, color=colors[index], label=f"seed {seed}")
    training_axes.set_title("Training coins (25-round rolling mean)")
    training_axes.set_xlabel("training round")
    training_axes.set_ylabel("coins")
    training_axes.legend(fontsize=8)

    # Crate destruction and bomb use.
    activity_axes = axes[1, 0]
    for index, seed in enumerate(seeds):
        rounds = rounds_by_seed[seed]
        x, crates = rolling([row["crates"] for row in rounds])
        _, bombs = rolling([row["bombs"] for row in rounds])
        activity_axes.plot(x + 1, crates, color=colors[index],
                           label=f"seed {seed} crates")
        activity_axes.plot(x + 1, bombs, color=colors[index], linestyle=":",
                           alpha=0.8, label=f"seed {seed} bombs")
    activity_axes.set_title("Bombing activity (25-round rolling mean)")
    activity_axes.set_xlabel("training round")
    activity_axes.set_ylabel("events per round")
    activity_axes.legend(fontsize=7, ncol=2)

    # Shaped reward and safety diagnostics.
    reward_axes = axes[1, 1]
    for index, seed in enumerate(seeds):
        rounds = rounds_by_seed[seed]
        x, rewards = rolling([row["reward"] for row in rounds])
        reward_axes.plot(x + 1, rewards, color=colors[index],
                         label=f"seed {seed} reward")
    reward_axes.axhline(0, color="black", linewidth=1)
    reward_axes.set_title("Shaped training reward (25-round rolling mean)")
    reward_axes.set_xlabel("training round")
    reward_axes.set_ylabel("reward")
    reward_axes.legend(fontsize=8)

    figure.suptitle("Combat FQI crate pilot", fontsize=15)
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
