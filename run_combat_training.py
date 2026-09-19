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
DEFAULT_CLASSIC_OPPONENTS = [
    "peaceful_agent", "coin_collector_agent", "rule_based_agent",
]


def _cycled_choice(options, offset):
    return options[int(offset) % len(options)]


def episode_plan(config, episode):
    """Return scenario, optional opponent, and replay weights for one episode."""
    if config.get("curriculum") == "mixed":
        scenario = "coin-heaven" if episode % 2 else "loot-crate"
        return scenario, None, {"coin-heaven": 0.5, "loot-crate": 0.5}
    if config.get("curriculum") != "staged-combat":
        return config["scenario"], None, {config["scenario"]: 1.0}

    if episode <= 100:
        # 70% navigation, 30% crate exposure.
        scenario = _cycled_choice(
            ("coin-heaven", "coin-heaven", "coin-heaven", "coin-heaven",
             "coin-heaven", "coin-heaven", "coin-heaven", "loot-crate",
             "loot-crate", "loot-crate"), episode - 1,
        )
        return scenario, None, {"coin-heaven": 0.7, "loot-crate": 0.3}
    if episode <= 300:
        # Reverse the emphasis while rehearsing navigation.
        scenario = _cycled_choice(
            ("coin-heaven", "loot-crate", "loot-crate", "loot-crate"),
            episode - 101,
        )
        return scenario, None, {"coin-heaven": 0.25, "loot-crate": 0.75}

    # Half classic experience, with the retained solo tasks sampled equally.
    scenario = _cycled_choice(
        ("classic", "classic", "coin-heaven", "loot-crate"), episode - 301,
    )
    opponent = None
    if scenario == "classic":
        classic_index = (episode - 301) // 4 * 2 + (episode - 301) % 2
        opponent = _cycled_choice(config["classic_opponents"], classic_index)
    return scenario, opponent, {
        "coin-heaven": 0.25, "loot-crate": 0.25, "classic": 0.5,
    }


def evaluate(config, learner, episode, interactions, scenario):
    """Freeze the current model and evaluate it in a fresh process."""
    directory = Path(config["output"])
    checkpoint = directory / "checkpoints" / f"episode_{episode:04d}.pkl"
    callbacks = importlib.import_module(
        f"agent_code.{config['agent']}.callbacks")
    if hasattr(callbacks, "save_checkpoint"):
        callbacks.save_checkpoint(learner, checkpoint)
    else:
        with open(checkpoint, "wb") as file:
            pickle.dump(learner.trees, file)

    evaluation_root = directory / "evaluation"
    if config.get("curriculum") in {"mixed", "staged-combat"}:
        evaluation_root = evaluation_root / scenario
    output = evaluation_root / f"episode_{episode:04d}"
    command = [
        sys.executable,
        str(SOURCE_DIR / "run_benchmark.py"),
        "--agents", config["agent"],
        "--scenario", scenario,
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
    if config.get("diagnostics"):
        command.append("--diagnostics")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_suffix(".log"), "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(output / "summary.json") as file:
        result = json.load(file)["agents"][config["agent"]]
    return {
        "episode": episode,
        "scenario": scenario,
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
    callbacks.RESUME_PATH = (
        Path(config["resume_checkpoint"]).resolve()
        if config.get("resume_checkpoint") else None
    )
    s.MAX_STEPS = config["max_steps"]
    s.LOG_GAME = s.LOG_AGENT_WRAPPER = s.LOG_AGENT_CODE = logging.WARNING
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    # DQN network initialization otherwise depends on process entropy, which
    # makes repeated runs with the same experiment seed incomparable.
    try:
        import torch
        torch.manual_seed(int(config["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config["seed"]))
    except ImportError:
        pass

    solo_args = SimpleNamespace(
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
    (directory / "checkpoints").mkdir(exist_ok=True)
    (directory / "logs").mkdir(exist_ok=True)
    world = BenchmarkWorld(
        solo_args,
        [(config["agent"], True)] + [(name, False) for name in config["opponents"]],
    )
    learner = world.agents[0].backend.runner.fake_self
    classic_worlds = {}
    if config.get("curriculum") == "staged-combat":
        # Build one reusable world per opponent, but rebind each candidate
        # wrapper to the solo learner. This keeps one model, optimizer, and
        # replay buffer while rotating the classic opponent every episode.
        for opponent in config["classic_opponents"]:
            classic_log_dir = directory / "logs" / opponent
            classic_agent_log_dir = classic_log_dir / "agents"
            classic_log_dir.mkdir(parents=True, exist_ok=True)
            classic_agent_log_dir.mkdir(parents=True, exist_ok=True)
            classic_args = SimpleNamespace(
                seed=config["seed"], agent_seed=config["seed"], seat=0,
                scenario="classic", no_gui=True, save_replay=False,
                save_stats=False, match_name=None,
                continue_without_training=False, silence_errors=False,
                log_dir=str(classic_log_dir),
                agent_log_dir=str(classic_agent_log_dir),
            )
            classic_world = BenchmarkWorld(
                classic_args, [(config["agent"], True), (opponent, False)]
            )
            classic_world.agents[0].backend.runner.fake_self = learner
            classic_worlds[opponent] = (classic_world, classic_args)

    start_episode = int(config.get("start_episode", 0))
    curve_path = directory / "learning_curve.json"
    if start_episode and curve_path.is_file():
        with open(curve_path) as file:
            curve = json.load(file)
    else:
        curve = [
            evaluate(config, learner, 0, 0, scenario)
            for scenario in config["evaluation_scenarios"]
        ]
        save_json(curve_path, curve)
    # A resumed checkpoint carries the interaction counter.  This preserves
    # epsilon/training progress in logs even though the current checkpoint
    # format does not serialize the replay buffer itself.
    interactions = int(getattr(learner, "env_steps", 0))
    training_seconds = 0.0
    rounds_path = directory / "rounds.jsonl"
    mode = "a" if start_episode and rounds_path.is_file() else "w"
    with open(rounds_path, mode) as file:
        for episode in tqdm(range(start_episode + 1, config["rounds"] + 1),
                            desc=f"{config['agent']} seed {config['seed']}"):
            board_seed = config["board_start"] + episode - 1
            scenario, opponent, replay_weights = episode_plan(config, episode)
            if opponent is None:
                active_world, active_args = world, solo_args
            else:
                active_world, active_args = classic_worlds[opponent]
            active_world.rng = np.random.default_rng(board_seed)
            # Round IDs are part of the action-time feature cache. Multiple
            # worlds therefore use the global training episode as their round.
            active_world.round = episode - 1
            active_args.seat = (episode - 1) % 4
            active_args.scenario = scenario
            if hasattr(learner.replay_buffer, "set_context"):
                learner.replay_buffer.set_context(scenario, replay_weights)
            epsilon = learner.epsilon
            started = perf_counter()
            active_world.new_round()
            while active_world.running:
                active_world.do_step()
            training_seconds += perf_counter() - started
            interactions += active_world.step

            agent = active_world.agents[0]
            stats = agent.statistics
            record = {
                "episode": episode,
                "board_seed": board_seed,
                "scenario": scenario,
                "opponent": opponent,
                "replay_weights": replay_weights,
                "agent_seed": config["seed"],
                "seat": active_args.seat,
                "steps": active_world.step,
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
                "buffer_size": len(getattr(learner, "transitions", [])),
                "replay_buffer_size": len(getattr(learner, "replay_buffer", [])),
                "optimizer_steps": int(getattr(learner, "optimizer_steps", 0)),
                "average_loss": getattr(learner, "last_round_average_loss", None),
                "tree_nodes": sum(tree.tree_.node_count for tree in
                                  getattr(learner, "trees", []) if tree is not None),
                "training_seconds": training_seconds,
            }
            file.write(json.dumps(record) + "\n")
            file.flush()
            if episode % config["eval_every"] == 0 or episode == config["rounds"]:
                curve.extend(
                    evaluate(config, learner, episode, interactions, scenario)
                    for scenario in config["evaluation_scenarios"]
                )
                save_json(directory / "learning_curve.json", curve)
    world.end()
    for classic_world, _ in classic_worlds.values():
        classic_world.end()


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--scenario",
                        choices=["coin-heaven", "loot-crate", "classic"],
                        default="loot-crate")
    parser.add_argument("--curriculum", choices=["none", "mixed", "staged-combat"],
                        default="none",
                        help=("Training schedule: none, alternating solo mixed, or "
                              "100 navigation / 200 crate / 300 retained-combat."))
    parser.add_argument("--diagnostics", action="store_true",
                        help="Record crate/coin progress diagnostics during evaluation.")
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument(
        "--classic-opponents", nargs="+", default=DEFAULT_CLASSIC_OPPONENTS,
        help="Single-opponent classic roster rotated during staged-combat training.",
    )
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
    parser.add_argument(
        "--resume", action="store_true",
        help=("Continue an existing output directory from its latest per-seed "
              "checkpoint; --rounds is the target total episode count."),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return args
    if args.output is None:
        parser.error("Choose an --output directory.")
    if args.resume:
        if not args.output.is_dir():
            parser.error("--resume requires an existing --output directory.")
    elif args.output.exists():
        parser.error("Choose a new --output directory, or pass --resume.")
    if len(args.opponents) > 3:
        parser.error("At most three opponents are allowed.")
    if args.curriculum in {"mixed", "staged-combat"} and args.opponents:
        parser.error("Curriculum schedules manage opponents internally; omit --opponents.")
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
    if args.curriculum == "staged-combat":
        if len(args.classic_opponents) > 3:
            parser.error("At most three staged classic opponents are supported.")
        for opponent in args.classic_opponents:
            if not (SOURCE_DIR / "agent_code" / opponent / "callbacks.py").is_file():
                parser.error(f"Unknown staged classic opponent: {opponent}")
    training_boards = set(range(4000, 4000 + args.rounds * len(args.seeds)))
    if training_boards.intersection(args.eval_seeds):
        parser.error("Evaluation seeds overlap training board seeds.")
    return args


def run_training(args):
    train_module = importlib.import_module(f"agent_code.{args.agent}.train")
    args.output = args.output.resolve()
    previous_config = {}
    if args.resume:
        config_path = args.output / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume {args.output}: missing config.json"
            )
        with open(config_path) as file:
            previous_config = json.load(file)
        for name in ("agent", "scenario", "curriculum", "opponents", "classic_opponents"):
            if name in previous_config and getattr(args, name) != previous_config[name]:
                raise ValueError(
                    f"--resume configuration mismatch for {name}: "
                    f"requested {getattr(args, name)!r}, "
                    f"existing {previous_config[name]!r}"
                )
        if "seeds" in previous_config and set(args.seeds) != set(previous_config["seeds"]):
            raise ValueError(
                "--resume requires the same training seed list as the existing run"
            )
    else:
        args.output.mkdir(parents=True)
    config = vars(args).copy()
    config["output"] = str(args.output)
    if args.resume:
        config["resumed_from_rounds"] = previous_config.get("rounds")
    config["evaluation_scenarios"] = (
        ["coin-heaven", "loot-crate"]
        if args.curriculum in {"mixed", "staged-combat"} else [args.scenario]
    )
    config["hyperparameters"] = {
        name: value for name, value in vars(train_module).items()
        if name.isupper() and isinstance(value, (int, float, str))
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
    if args.agent in {
        "combat_fqi_history_antistag_agent",
        "combat_fqi_history_antistag_topology_agent",
    }:
        # These variants intentionally import the already audited combat
        # feature and safety implementations; hash those dependencies for
        # provenance.
        source_paths.extend([
            SOURCE_DIR / "agent_code" / "combat_fqi_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_agent" / "safety.py",
        ])
    if args.agent == "combat_fqi_history_antistag_topology_agent":
        source_paths.extend([
            SOURCE_DIR / "agent_code" / "combat_fqi_history_antistag_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_history_antistag_agent" / "safety.py",
        ])
    if args.agent == "combat_dqn_agent":
        source_paths.extend([
            SOURCE_DIR / "agent_code" / "combat_fqi_history_antistag_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_history_antistag_agent" / "safety.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_history_antistag_agent" / "train.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "config.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "model.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "replay.py",
        ])
    if args.agent == "combat_dqn_topology_agent":
        source_paths.extend([
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "callbacks.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "config.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "model.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "replay.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "train.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_history_antistag_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_history_antistag_agent" / "safety.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_history_antistag_topology_agent" / "features.py",
        ])
    if args.agent in {
        "Agent_022_combat_ddqn_route_agent",
        "Agent_024_combat_ddqn_action_safety_agent",
        "Agent_025_combat_ddqn_short_cycle_agent",
        "Agent_026_combat_ddqn_target_coverage_agent",
        "Agent_027_combat_ddqn_short_cycle_staged_replay_agent",
        "Agent_028_combat_ddqn_dueling_256_agent",
    }:
        # This successor intentionally reuses the repaired DQN implementation
        # and history/safety code while replacing only its representation.
        source_paths.extend([
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "callbacks.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "config.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "model.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "replay.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_agent" / "train.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_agent" / "safety.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_history_antistag_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_history_antistag_agent" / "safety.py",
        ])
    if args.agent == "Agent_027_combat_ddqn_short_cycle_staged_replay_agent":
        source_paths.extend([
            SOURCE_DIR / "agent_code" / args.agent / "replay.py",
            SOURCE_DIR / "agent_code" / "Agent_025_combat_ddqn_short_cycle_agent" / "features.py",
        ])
    if args.agent == "Agent_028_combat_ddqn_dueling_256_agent":
        source_paths.extend([
            SOURCE_DIR / "agent_code" / args.agent / "model.py",
            SOURCE_DIR / "agent_code" / "Agent_027_combat_ddqn_short_cycle_staged_replay_agent" / "callbacks.py",
            SOURCE_DIR / "agent_code" / "Agent_027_combat_ddqn_short_cycle_staged_replay_agent" / "replay.py",
            SOURCE_DIR / "agent_code" / "Agent_025_combat_ddqn_short_cycle_agent" / "features.py",
        ])
    if args.agent == "Agent_023_spatial_hybrid_rainbow_agent":
        # Agent 023 is intentionally self-contained; include every runtime
        # module in provenance rather than relying on a shared DQN package.
        source_paths.extend([
            SOURCE_DIR / "agent_code" / args.agent / "__init__.py",
            SOURCE_DIR / "agent_code" / args.agent / "config.py",
            SOURCE_DIR / "agent_code" / args.agent / "model.py",
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
        if args.resume:
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Cannot resume seed {seed}: missing {directory}"
                )
            checkpoints = sorted(directory.glob("checkpoints/episode_*.pkl"))
            if not checkpoints:
                raise FileNotFoundError(
                    f"Cannot resume seed {seed}: no episode checkpoint found"
                )
            checkpoint = checkpoints[-1]
            try:
                checkpoint_episode = int(checkpoint.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"Invalid episode checkpoint filename: {checkpoint}"
                ) from error
            if checkpoint_episode >= args.rounds:
                raise ValueError(
                    f"Seed {seed} is already at episode {checkpoint_episode}; "
                    f"--rounds must be larger than the checkpoint episode."
                )
            with open(directory / "config.json") as file:
                previous_seed_config = json.load(file)
        else:
            directory.mkdir()
            checkpoint = None
            checkpoint_episode = 0
            previous_seed_config = {}
        run_config = config.copy()
        run_config["seed"] = seed
        run_config["output"] = str(directory)
        run_config["board_start"] = previous_seed_config.get(
            "board_start", 4000 + index * args.rounds
        )
        run_config["start_episode"] = checkpoint_episode
        if checkpoint is not None:
            run_config["resume_checkpoint"] = str(checkpoint.resolve())
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
