"""Prepare and evaluate the corrected 101-vs-113 feature experiment.

Training is launched separately through jobctl. Generated manifests retain
the fixed Classic boards/seats and add matched solo regression measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from prepare_offensive_checkpoint import expand_checkpoint

SOURCE = Path(__file__).resolve().parent
EXPERIMENT = SOURCE.parent / "bomberman_rl_exp"
INITIAL = Path("/export/scratch/salitanl/bomberman_feature_variants_20260919/agent029_adversarial_window_staged_600_cpu_jobctl")
CONTROL = "Agent_030_combat_ddqn_escape_replay_agent"
CANDIDATE = "Agent_031_combat_ddqn_offensive_escape_agent"


def write_json(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    originals = [INITIAL / f"seed_{seed}/checkpoints/episode_0600.pkl" for seed in range(3)]
    for seed, original in enumerate(originals):
        migrated = expand_checkpoint(torch.load(original, map_location="cpu", weights_only=False))
        migrated["initial_checkpoint_source"] = str(original)
        migrated["initial_checkpoint_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
        destination = root / f"initial_113/seed_{seed}/episode_0600.pkl"
        destination.parent.mkdir(parents=True)
        with destination.open("xb") as output:
            torch.save(migrated, output)
    lineups = [
        {"name": "peaceful", "scenario": "classic", "opponents": ["peaceful_agent"]},
        {"name": "coin_collector", "scenario": "classic", "opponents": ["coin_collector_agent"]},
        {"name": "rule_based", "scenario": "classic", "opponents": ["rule_based_agent"]},
        {"name": "three_rule_based", "scenario": "classic", "opponents": ["rule_based_agent"] * 3},
        {"name": "mixed_strong", "scenario": "classic",
         "opponents": ["rule_based_agent", "coin_collector_agent", "peaceful_agent"]},
        {"name": "solo", "scenario": "coin-heaven", "opponents": []},
        {"name": "solo", "scenario": "loot-crate", "opponents": []},
    ]
    candidates = [{"name": "baseline029_600", "agent": "Agent_029_combat_ddqn_adversarial_window_agent",
                   "checkpoints": list(map(str, originals))}]
    for name, agent in (("control101", CONTROL), ("candidate113", CANDIDATE)):
        candidates.append({"name": name, "agent": agent, "checkpoints": [
            str(root / name / f"seed_{seed}/checkpoints/episode_0900.pkl") for seed in range(3)
        ]})
    write_json(root / "evaluation_manifest.json", {
        "board_seeds": list(range(32000, 32008)), "agent_seeds": [0],
        "seats": [0, 1, 2, 3], "max_steps": 400, "diagnostics": True,
        "opponent_lineups": lineups, "candidates": candidates,
    })
    for name, agent, initial in (
        ("control101", CONTROL, originals),
        ("candidate113", CANDIDATE, [root / f"initial_113/seed_{seed}/episode_0600.pkl" for seed in range(3)]),
    ):
        command = [sys.executable, str(SOURCE / "run_combat_training.py"),
                   "--agent", agent, "--curriculum", "tournament-combat-retained-solo",
                   "--rounds", "900", "--max-steps", "400", "--eval-every", "50",
                   "--seeds", "0", "1", "2", "--eval-seeds", "33000", "33001", "33002",
                   "--eval-seats", "0", "--diagnostics", "--parallel-seeds",
                   "--eval-workers", "2", "--eval-scenario-workers", "1",
                   "--initial-checkpoints", *map(str, initial), "--output", str(root / name)]
        write_json(root / f"{name}_command.json", command)
    print(f"Prepared original and migrated checkpoints plus evaluation manifest in {root}")


def train(root, name):
    command = json.loads((root / f"{name}_command.json").read_text())
    subprocess.run(command, check=True, cwd=EXPERIMENT)


def evaluate(root):
    subprocess.run([
        sys.executable, str(EXPERIMENT / "eval_suite/run_suite.py"),
        "--manifest", str(root / "evaluation_manifest.json"),
        "--output", str(root / "frozen_eval"), "--parallel", "3", "--game-workers", "4",
    ], check=True, cwd=EXPERIMENT)
    manifest = json.loads((root / "evaluation_manifest.json").read_text())
    comparisons = []
    for candidate in manifest["candidates"]:
        for lineup in manifest["opponent_lineups"]:
            games = []
            for seed in range(3):
                name = f"{candidate['name']}_{lineup['scenario']}_{lineup['name']}_{seed}"
                for path in sorted((root / "frozen_eval" / name / "batches").glob("*_results.json")):
                    games.extend(json.loads(path.read_text()))
            if len(games) != 96:
                raise ValueError(f"Incomplete evaluation for {candidate['name']}/{lineup}: {len(games)}")
            rows = [game["agents"][0] for game in games]
            record = {"candidate": candidate["name"], "scenario": lineup["scenario"],
                      "lineup": lineup["name"], "games": len(games)}
            for key in ("score", "coins", "kills", "suicides", "invalid_actions"):
                record["mean_" + key] = sum(row[key] for row in rows) / len(rows)
            record["survival_rate"] = sum(not row["dead"] for row in rows) / len(rows)
            if lineup["opponents"]:
                scores = [[agent["score"] for agent in game["agents"]] for game in games]
                record["outright_win_rate"] = sum(s[0] > max(s[1:]) for s in scores) / len(scores)
                record["joint_first_rate"] = sum(s[0] == max(s) for s in scores) / len(scores)
                record["mean_rank"] = sum(1 + sum(v > s[0] for v in s[1:]) +
                                          .5 * sum(v == s[0] for v in s[1:])
                                          for s in scores) / len(scores)
            comparisons.append(record)
            print(json.dumps(record), flush=True)
    write_json(root / "comparison.json", comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "control101", "candidate113", "evaluate"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.action == "prepare":
        prepare(root)
    elif args.action == "evaluate":
        evaluate(root)
    else:
        train(root, args.action)
