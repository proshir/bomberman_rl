"""Train and evaluate the bomb-aware fitted-Q agent on crates or opponents."""

# Sahand was here.

import hashlib
import importlib
import json
import logging
import pickle
import random
import subprocess
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

import settings as s
from run_benchmark import BenchmarkWorld, SOURCE_DIR, save_json


DEFAULT_AGENT = "combat_fqi_agent"


def evaluate(config, learner, episode, interactions):
    """Freeze the current trees and evaluate them in a fresh process."""
    directory = Path(config["output"])
    checkpoint = directory / "checkpoints" / f"episode_{episode:04d}.pkl"
    with open(checkpoint, "wb") as file:
        pickle.dump(learner.trees, file)

    output = directory / "evaluation" / f"episode_{episode:04d}"
    command = [
        sys.executable,
        str(SOURCE_DIR / "run_benchmark.py"),
        "--agents", config["agent"],
        "--scenario", config["scenario"],
        "--model-path", str(checkpoint),
        "--max-steps", str(config["max_steps"]),
        "--seeds", *map(str, config["eval_seeds"]),
        "--agent-seeds", "0",
        "--seats", *map(str, config["eval_seats"]),
        "--batch-size", str(len(config["eval_seeds"]) * len(config["eval_seats"])),
        "--metric", "score" if config["opponents"] else "coins",
        "--output", str(output),
    ]
    if config["opponents"]:
        command.extend(["--opponents", *config["opponents"]])
    output.parent.mkdir(exist_ok=True)
    with open(output.with_suffix(".log"), "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(output / "summary.json") as file:
        result = json.load(file)["agents"][config["agent"]]
    return {
        "episode": episode,
        "interactions": interactions,
        "checkpoint": str(checkpoint),
        **result,
    }


def train(config):
    """Run one independent training seed and preserve every round."""
    directory = Path(config["output"])
    callbacks = importlib.import_module(
        f"agent_code.{config['agent']}.callbacks")
    callbacks.MODEL_PATH = directory / "training.pkl"
    s.MAX_STEPS = config["max_steps"]
    s.LOG_GAME = s.LOG_AGENT_WRAPPER = s.LOG_AGENT_CODE = logging.WARNING
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    args = SimpleNamespace(
        seed=config["seed"],
        agent_seed=config["seed"],
        seat=0,
        scenario=config["scenario"],
        no_gui=True,
        save_replay=False,
        save_stats=False,
        match_name=None,
        continue_without_training=False,
        silence_errors=False,
        log_dir=str(directory / "logs"),
        agent_log_dir=str(directory / "logs" / "agents"),
    )
    (directory / "checkpoints").mkdir()
    (directory / "logs").mkdir()
    lineup = [(config["agent"], True)] + [
        (name, False) for name in config["opponents"]
    ]
    world = BenchmarkWorld(args, lineup)
    learner = world.agents[0].backend.runner.fake_self

    curve = [evaluate(config, learner, 0, 0)]
    save_json(directory / "learning_curve.json", curve)
    interactions = 0
    training_seconds = 0.0
    with open(directory / "rounds.jsonl", "w") as file:
        for episode in tqdm(range(1, config["rounds"] + 1),
                            desc=f"{config['agent']} seed {config['seed']}"):
            board_seed = config["board_start"] + episode - 1
            world.rng = np.random.default_rng(board_seed)
            args.seat = (episode - 1) % 4
            epsilon = learner.epsilon
            started = perf_counter()
            world.new_round()
            while world.running:
                world.do_step()
            training_seconds += perf_counter() - started
            interactions += world.step

            agent = world.agents[0]
            stats = agent.statistics
            record = {
                "episode": episode,
                "board_seed": board_seed,
                "agent_seed": config["seed"],
                "seat": args.seat,
                "steps": world.step,
                "interactions": interactions,
                "score": agent.score,
                "coins": stats["coins"],
                "crates": stats["crates"],
                "kills": stats["kills"],
                "suicides": stats["suicides"],
                "bombs": stats["bombs"],
                "invalid_actions": stats["invalid"],
                "survived": not agent.dead,
                "reward": learner.last_round_reward,
                "epsilon": epsilon,
                "buffer_size": len(learner.transitions),
                "tree_nodes": sum(tree.tree_.node_count for tree in learner.trees
                                  if tree is not None),
                "training_seconds": training_seconds,
            }
            file.write(json.dumps(record) + "\n")
            file.flush()
            if episode % config["eval_every"] == 0 or episode == config["rounds"]:
                curve.append(evaluate(config, learner, episode, interactions))
                save_json(directory / "learning_curve.json", curve)
    world.end()


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--scenario",
                        choices=["coin-heaven", "loot-crate", "classic"],
                        default="loot-crate")
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, nargs="+",
                        default=list(range(30000, 30008)))
    parser.add_argument("--eval-seats", type=int, nargs="+", choices=range(4),
                        default=[0, 1, 2, 3])
    parser.add_argument(
        "--parallel-seeds", action="store_true",
        help="Train independent seeds concurrently on multi-core CPU hosts.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return args
    if args.output is None or args.output.exists():
        parser.error("Choose a new --output directory.")
    if len(args.opponents) > 3:
        parser.error("At most three opponents are allowed.")
    if not (SOURCE_DIR / "agent_code" / args.agent / "callbacks.py").is_file():
        parser.error(f"Unknown training agent: {args.agent}")
    if min(args.rounds, args.max_steps, args.eval_every) < 1:
        parser.error("Round and step counts must be positive.")
    for values in (args.seeds, args.eval_seeds, args.eval_seats):
        if len(set(values)) != len(values):
            parser.error("Seed and seat lists may not contain duplicates.")
    for opponent in args.opponents:
        if not (SOURCE_DIR / "agent_code" / opponent / "callbacks.py").is_file():
            parser.error(f"Unknown opponent: {opponent}")
    training_boards = set(range(4000, 4000 + args.rounds * len(args.seeds)))
    if training_boards.intersection(args.eval_seeds):
        parser.error("Evaluation seeds overlap training board seeds.")
    return args


def run_training(args):
    train_module = importlib.import_module(f"agent_code.{args.agent}.train")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["hyperparameters"] = {
        name: value for name, value in vars(train_module).items()
        if name.isupper() and isinstance(value, (int, float))
    }
    config["code_version"] = subprocess.check_output(
        ["git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD"], text=True).strip()
    source_paths = [
        Path(__file__).resolve(), SOURCE_DIR / "run_benchmark.py",
        SOURCE_DIR / "environment.py", SOURCE_DIR / "agents.py",
        SOURCE_DIR / "settings.py", SOURCE_DIR / "events.py",
        SOURCE_DIR / "agent_code" / args.agent / "callbacks.py",
        SOURCE_DIR / "agent_code" / args.agent / "features.py",
        SOURCE_DIR / "agent_code" / args.agent / "safety.py",
        SOURCE_DIR / "agent_code" / args.agent / "train.py",
    ]
    if args.agent == "combat_fqi_history_antistag_agent":
        # The variant intentionally imports the already audited combat feature
        # and safety implementation; hash those dependencies for provenance.
        source_paths.extend([
            SOURCE_DIR / "agent_code" / "combat_fqi_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_agent" / "safety.py",
        ])
    config["source_hashes"] = {
        str(path.relative_to(SOURCE_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    config["python"] = sys.version
    config["numpy"] = np.__version__
    import sklearn
    config["sklearn"] = sklearn.__version__
    save_json(args.output / "config.json", config)

    jobs = []
    for index, seed in enumerate(args.seeds):
        directory = args.output / f"seed_{seed}"
        directory.mkdir()
        run_config = config.copy()
        run_config["seed"] = seed
        run_config["output"] = str(directory)
        run_config["board_start"] = 4000 + index * args.rounds
        config_path = directory / "config.json"
        save_json(config_path, run_config)
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--worker", str(config_path)]
        if args.parallel_seeds:
            jobs.append((seed, directory, subprocess.Popen(command)))
        else:
            subprocess.run(command, check=True)
            jobs.append((seed, directory, None))

    for seed, _, process in jobs:
        if process is not None and process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode,
                                                f"training seed {seed}")

    curve = []
    for seed, directory, _ in jobs:
        with open(directory / "learning_curve.json") as file:
            curve.extend({"seed": seed, **row} for row in json.load(file))
    save_json(args.output / "learning_curve.json", curve)
    for row in curve:
        metric = "score" if args.opponents else "coins"
        print(f"Seed {row['seed']}, episode {row['episode']}: "
              f"{metric} = {row['mean']:.2f}")


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        with open(args.worker) as file:
            train(json.load(file))
    else:
        run_training(args)


if __name__ == "__main__":
    main()
