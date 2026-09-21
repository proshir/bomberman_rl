"""Train and evaluate the bomb-aware fitted-Q agent on crates or opponents."""

# Sahand was here.

import hashlib
import importlib
import json
import logging
import os
import pickle
import random
import subprocess
import sys
from argparse import ArgumentParser, SUPPRESS
from concurrent.futures import ThreadPoolExecutor
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
TOURNAMENT_CURRICULA = {"tournament-combat", "tournament-combat-retained-solo"}
LEAGUE_CURRICULA = {
    "league-combat", "agent038-population", "final-finetune",
    "agent043-staged-league", "agent043-staged-league-memory-retention",
}
COMPLETE_LINEUP_CURRICULA = {
    "final-finetune", "agent043-staged-league",
    "agent043-staged-league-memory-retention",
}
FINAL_FINETUNE_FORBIDDEN_OPPONENTS = {
    "coin_collector_agent", "peaceful_agent",
}


def _cycled_choice(options, offset):
    return options[int(offset) % len(options)]


def episode_plan(config, episode):
    """Return scenario, optional opponent, and replay weights for one episode."""
    if config.get("curriculum") in {
            "agent043-staged-league",
            "agent043-staged-league-memory-retention"}:
        # A strict three-stage progression requested for Agent 043:
        # navigation foundation, balanced solo transfer, then only complete
        # four-player Classic games from a focused opponent roster.
        if episode <= 100:
            return "coin-heaven", None, {"coin-heaven": 1.0}
        if episode <= 300:
            scenario = _cycled_choice(
                ("coin-heaven", "loot-crate"), episode - 101,
            )
            return scenario, None, {
                "coin-heaven": 0.50, "loot-crate": 0.50,
            }

        lineups = config["classic_lineups"]
        offset = episode - 301
        if config.get("curriculum") == "agent043-staged-league-memory-retention":
            # Each deterministically shuffled ten-round block contains eight
            # complete Classic games and one round of each solo task.  The
            # ordering is random per seed/block, while the exact 80/10/10
            # allocation and balanced lineup exposure remain reproducible.
            block, within = divmod(offset, 10)
            slots = [
                ("classic", index % len(lineups)) for index in range(8)
            ] + [("coin-heaven", None), ("loot-crate", None)]
            random.Random(
                (int(config["seed"]) + 1) * 1_000_003 + block
            ).shuffle(slots)
            scenario, lineup_index = slots[within]
            weights = {"coin-heaven": 0.10, "loot-crate": 0.10}
            for lineup in lineups:
                weights["classic|" + ",".join(lineup)] = 0.80 / len(lineups)
            if scenario != "classic":
                return scenario, None, weights
            return scenario, list(lineups[lineup_index]), weights

        cycle, within = divmod(offset, len(lineups))
        order = list(range(len(lineups)))
        random.Random(
            (int(config["seed"]) + 1) * 1_000_003 + cycle
        ).shuffle(order)
        weights = {
            "classic|" + ",".join(lineup): 1.0 / len(lineups)
            for lineup in lineups
        }
        return "classic", list(lineups[order[within]]), weights

    if config.get("curriculum") in LEAGUE_CURRICULA and episode > 300:
        # Agent 038 uses the design's 70/15/15 population/solo mixture.
        if config.get("curriculum") == "agent038-population":
            offset = episode - 301
            phase = offset % 20
            scenario = (
                "classic" if phase < 14
                else "coin-heaven" if phase < 17
                else "loot-crate"
            )
            lineups = config["classic_lineups"]
            weights = {"coin-heaven": 0.15, "loot-crate": 0.15}
            for lineup in lineups:
                tag = "classic|" + ",".join(lineup)
                weights[tag] = weights.get(tag, 0.0) + 0.70 / len(lineups)
            if scenario != "classic":
                return scenario, None, weights
            combat_index = (offset // 20) * 14 + phase
            cycle, within = divmod(combat_index, len(lineups))
            order = list(range(len(lineups)))
            random.Random(
                (int(config["seed"]) + 1) * 1_000_003 + cycle
            ).shuffle(order)
            return scenario, list(lineups[order[within]]), weights

        if config.get("curriculum") == "final-finetune":
            # Keep solo retention, but make every non-solo game a complete
            # four-player Classic lineup.  The launch script supplies the
            # frozen rule-based/imported opponent roster.
            offset = episode - 301
            scenario = ("classic", "classic", "classic",
                        "coin-heaven", "loot-crate")[offset % 5]
            lineups = config["classic_lineups"]
            weights = {"coin-heaven": 0.20, "loot-crate": 0.20}
            for lineup in lineups:
                tag = "classic|" + ",".join(lineup)
                weights[tag] = weights.get(tag, 0.0) + 0.60 / len(lineups)
            if scenario != "classic":
                return scenario, None, weights
            combat_index = (offset // 5) * 3 + offset % 5
            cycle, within = divmod(combat_index, len(lineups))
            order = list(range(len(lineups)))
            random.Random(
                (int(config["seed"]) + 1) * 1_000_003 + cycle
            ).shuffle(order)
            return scenario, list(lineups[order[within]]), weights

        # Three combat games and one game of each solo task per five rounds.
        # Shuffle the fixed lineup roster once per cycle for reproducible,
        # balanced exposure instead of relying on global RNG state.
        offset = episode - 301
        scenario = ("classic", "classic", "classic", "coin-heaven",
                    "loot-crate")[offset % 5]
        lineups = config["classic_lineups"]
        weights = {"coin-heaven": 0.20, "loot-crate": 0.20}
        for lineup in lineups:
            tag = "classic|" + ",".join(lineup)
            weights[tag] = weights.get(tag, 0.0) + 0.60 / len(lineups)
        if scenario != "classic":
            return scenario, None, weights
        combat_index = (offset // 5) * 3 + offset % 5
        cycle, within = divmod(combat_index, len(lineups))
        order = list(range(len(lineups)))
        random.Random((int(config["seed"]) + 1) * 1_000_003 + cycle).shuffle(order)
        return scenario, list(lineups[order[within]]), weights
    if config.get("curriculum") == "mixed":
        scenario = "coin-heaven" if episode % 2 else "loot-crate"
        return scenario, None, {"coin-heaven": 0.5, "loot-crate": 0.5}
    if config.get("curriculum") not in {"staged-combat", *TOURNAMENT_CURRICULA,
                                           *LEAGUE_CURRICULA}:
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

    # The retained-solo variant deliberately generates solo transitions after
    # episode 600. Replay weights alone cannot retain a task in a fresh buffer.
    if (config.get("curriculum") == "tournament-combat-retained-solo" and
            episode > 600 and config.get("classic_lineups")):
        scenario = _cycled_choice(
            ("classic", "classic", "classic", "coin-heaven", "loot-crate"),
            episode - 601,
        )
        lineup = list(config["classic_lineups"][-1])
        weights = {
            "coin-heaven": 0.20,
            "loot-crate": 0.20,
            "classic|" + ",".join(lineup): 0.60,
        }
        return scenario, lineup if scenario == "classic" else None, weights

    # Historical tournament-combat behavior remains unchanged.
    if (config.get("curriculum") == "tournament-combat" and episode > 600
            and config.get("classic_lineups")):
        lineup = list(config["classic_lineups"][-1])
        return "classic", lineup, {
            "coin-heaven": 0.125,
            "loot-crate": 0.125,
            "classic|" + ",".join(lineup): 0.75,
        }

    # Half classic experience, with the retained solo tasks sampled equally.
    scenario = _cycled_choice(
        ("classic", "classic", "coin-heaven", "loot-crate"), episode - 301,
    )
    opponent = None
    if scenario == "classic":
        lineups = config.get("classic_lineups")
        if config.get("curriculum") in TOURNAMENT_CURRICULA and lineups:
            lineup_index = (episode - 301) // 2
            opponent = list(_cycled_choice(lineups, lineup_index))
            weights = {
                "coin-heaven": 0.25,
                "loot-crate": 0.25,
            }
            for lineup in lineups:
                tag = "classic|" + ",".join(lineup)
                weights[tag] = 0.5 / len(lineups)
            return scenario, opponent, weights
        classic_index = (episode - 301) // 4 * 2 + (episode - 301) % 2
        opponent = _cycled_choice(config["classic_opponents"], classic_index)
    return scenario, opponent, {
        "coin-heaven": 0.25, "loot-crate": 0.25, "classic": 0.5,
    }


def save_evaluation_checkpoint(config, learner, episode):
    """Save one immutable checkpoint shared by all evaluations of an episode."""
    directory = Path(config["output"])
    checkpoint = directory / "checkpoints" / f"episode_{episode:04d}.pkl"
    callbacks = importlib.import_module(
        f"agent_code.{config['agent']}.callbacks")
    if hasattr(callbacks, "save_checkpoint"):
        callbacks.save_checkpoint(learner, checkpoint)
    else:
        with open(checkpoint, "wb") as file:
            pickle.dump(learner.trees, file)
    return checkpoint


def evaluate_checkpoint(config, checkpoint, episode, interactions, evaluation):
    """Evaluate one frozen checkpoint in a fresh CPU process."""
    scenario = evaluation["scenario"]
    opponents = list(evaluation.get("opponents", []))
    evaluation_name = evaluation["name"]
    directory = Path(config["output"])

    evaluation_root = directory / "evaluation"
    if config.get("curriculum") in {"mixed", "staged-combat",
                                     *TOURNAMENT_CURRICULA,
                                     *LEAGUE_CURRICULA}:
        evaluation_root = evaluation_root / evaluation_name
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
        # A one-game batch enables real CPU parallelism.  The previous single
        # large batch is retained for the serial compatibility path.
        "--batch-size", str(
            1 if int(config.get("eval_workers", 1)) > 1
            else len(config["eval_seeds"]) * len(config["eval_seats"])
        ),
        "--parallel", str(int(config.get("eval_workers", 1))),
        "--metric", "score" if opponents else "coins",
        "--output", str(output),
    ]
    if opponents:
        command.extend(["--opponents", *opponents])
    if config.get("diagnostics"):
        command.append("--diagnostics")
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluation_env = os.environ.copy()
    evaluation_env["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        evaluation_env[variable] = "1"
    with open(output.with_suffix(".log"), "w") as log:
        subprocess.run(
            command, check=True, stdout=log, stderr=subprocess.STDOUT,
            env=evaluation_env,
        )
    with open(output / "summary.json") as file:
        result = json.load(file)["agents"][config["agent"]]
    return {
        "episode": episode,
        "evaluation": evaluation_name,
        "scenario": scenario,
        "opponents": opponents,
        "interactions": interactions,
        "checkpoint": str(checkpoint),
        **result,
    }


def evaluate_all(config, learner, episode, interactions):
    """Evaluate all scenarios from one checkpoint, optionally concurrently.

    Each scenario has its own output directory and only reads the saved
    checkpoint.  Parallelism therefore affects wall-clock time only, not the
    training trajectory or the fixed evaluation task list.
    """
    checkpoint = save_evaluation_checkpoint(config, learner, episode)
    evaluations = [
        {"name": scenario, "scenario": scenario, "opponents": []}
        for scenario in config["evaluation_scenarios"]
    ]
    if config.get("curriculum") in (TOURNAMENT_CURRICULA | LEAGUE_CURRICULA):
        lineups = (config.get("league_eval_lineups", [])
                   if config.get("curriculum") in LEAGUE_CURRICULA
                   else config.get("classic_lineups", []))
        for lineup in lineups:
            opponents = list(lineup)
            evaluations.append({
                "name": "classic__" + "__".join(opponents),
                "scenario": "classic",
                "opponents": opponents,
            })
    workers = min(len(evaluations),
                  int(config.get("eval_scenario_workers", 1)))
    if workers == 1:
        return [
            evaluate_checkpoint(config, checkpoint, episode, interactions, evaluation)
            for evaluation in evaluations
        ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(
            lambda evaluation: evaluate_checkpoint(
                config, checkpoint, episode, interactions, evaluation),
            evaluations,
        ))


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
    if config.get("curriculum") in {"staged-combat", *TOURNAMENT_CURRICULA,
                                    *LEAGUE_CURRICULA}:
        # Build one reusable world per lineup, but rebind each candidate
        # wrapper to the solo learner. This keeps one model, optimizer, and
        # replay buffer while rotating the classic lineup every episode.
        lineups = config.get("classic_lineups") or [
            [opponent] for opponent in config["classic_opponents"]
        ]
        for lineup in lineups:
            lineup = tuple(lineup)
            if lineup in classic_worlds:
                continue
            lineup_tag = ",".join(lineup)
            classic_log_dir = directory / "logs" / lineup_tag.replace(",", "__")
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
                classic_args,
                [(config["agent"], True)] + [(name, False) for name in lineup],
            )
            classic_world.agents[0].backend.runner.fake_self = learner
            classic_worlds[lineup] = (classic_world, classic_args)

    start_episode = int(config.get("start_episode", 0))
    curve_path = directory / "learning_curve.json"
    if start_episode and curve_path.is_file():
        with open(curve_path) as file:
            curve = json.load(file)
    else:
        curve = evaluate_all(config, learner, start_episode,
                             int(getattr(learner, "env_steps", 0)))
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
            if config["agent"] in {"Agent_034_compact_fqi_agent",
                                   "Agent_035_compact_fqi_symmetry_agent",
                                   "Agent_036_compact_fqi_robust_agent"}:
                learner.scenario_tag = scenario
                learner.lineup_tag = (
                    tuple(opponent) if isinstance(opponent, (list, tuple))
                    else (() if opponent is None else (opponent,))
                )
                learner.board_seed = board_seed
                if hasattr(learner, "replay_weights"):
                    learner.replay_weights = replay_weights
            if opponent is None:
                active_world, active_args = world, solo_args
            else:
                lineup = (tuple(opponent) if isinstance(opponent, (list, tuple))
                          else (opponent,))
                active_world, active_args = classic_worlds[lineup]
            active_world.rng = np.random.default_rng(board_seed)
            # Round IDs are part of the action-time feature cache. Multiple
            # worlds therefore use the global training episode as their round.
            active_world.round = episode - 1
            active_args.seat = (episode - 1) % 4
            active_args.scenario = scenario
            if hasattr(getattr(learner, "replay_buffer", None), "set_context"):
                replay_tag = scenario
                if (opponent is not None and config.get("curriculum") in
                        (TOURNAMENT_CURRICULA | LEAGUE_CURRICULA)):
                    replay_tag += "|" + ",".join(lineup)
                learner.replay_buffer.set_context(replay_tag, replay_weights)
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
            if hasattr(learner, "combat_env_steps"):
                record["combat_env_steps"] = learner.combat_env_steps
                record["epsilon_after_round"] = learner.epsilon
                record["replay_tag_counts"] = {
                    tag: len(indices) for tag, indices in
                    learner.replay_buffer.indices_by_tag.items()
                }
                record["effective_replay_weights"] = learner.replay_buffer.weights
            if config["agent"] in {
                    "Agent_035_compact_fqi_symmetry_agent",
                    "Agent_036_compact_fqi_robust_agent"}:
                record.update({
                    "feature_seconds": learner.feature_seconds,
                    "safety_seconds": learner.safety_seconds,
                    "action_seconds": learner.action_seconds,
                    "max_action_seconds": learner.max_action_seconds,
                    "action_count": learner.action_count,
                    "fit_seconds": learner.fit_seconds,
                    "last_fit_records": learner.last_fit_records,
                    "last_fit_tag_counts": {
                        scenario + "|" + ",".join(lineup): count
                        for (scenario, lineup), count in
                        learner.last_fit_tag_counts.items()
                    },
                    "last_available_tag_counts": {
                        scenario + "|" + ",".join(lineup): count
                        for (scenario, lineup), count in
                        learner.last_available_tag_counts.items()
                    },
                    "last_effective_scenario_weights": (
                        learner.last_effective_scenario_weights
                    ),
                    "last_action_support": learner.last_action_support,
                })
            file.write(json.dumps(record) + "\n")
            file.flush()
            if episode % config["eval_every"] == 0 or episode == config["rounds"]:
                curve.extend(evaluate_all(config, learner, episode, interactions))
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
    parser.add_argument("--curriculum",
                        choices=["none", "mixed", "staged-combat",
                                 "tournament-combat",
                                 "tournament-combat-retained-solo",
                                 "league-combat", "agent038-population",
                                 "final-finetune", "Final_finetune",
                                 "agent043-staged-league",
                                 "agent043-staged-league-memory-retention"],
                        default="none",
                        help=("Training schedule: none, alternating solo mixed, or "
                              "100 navigation / 200 crate / 300 retained-combat. "
                              "tournament-combat uses four-player lineups; "
                              "the retained-solo variant interleaves solo rounds; "
                              "final-finetune retains solo rounds but uses only "
                              "complete four-player lineups outside solo training; "
                              "agent043-staged-league uses 100 Coin Heaven, 200 "
                              "balanced solo, then 600 complete Classic rounds; "
                              "its memory-retention variant keeps 20% solo rehearsal."))
    parser.add_argument("--diagnostics", action="store_true",
                        help="Record crate/coin progress diagnostics during evaluation.")
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument(
        "--classic-opponents", nargs="+", default=DEFAULT_CLASSIC_OPPONENTS,
        help="Single-opponent classic roster rotated during staged-combat training.",
    )
    parser.add_argument(
        "--classic-lineup", nargs="+", action="append", default=None,
        help=("One or more opponent names for a tournament-combat lineup; repeat "
              "for multiple lineups. Defaults to peaceful, collector, rule-based "
              "and 3x rule-based."),
    )
    parser.add_argument(
        "--league-eval-lineup", nargs="+", action="append", default=None,
        help="Fixed Classic opponent lineup for league checkpoint evaluation.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument(
        "--board-stride", type=int,
        help=("Distance between each seed's first training board. Defaults to "
              "--rounds; set this explicitly for a shortened matched run."),
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, nargs="+",
                        default=list(range(30000, 30008)))
    parser.add_argument("--eval-seats", type=int, nargs="+", choices=range(4),
                        default=[0, 1, 2, 3])
    parser.add_argument(
        "--eval-workers", type=int, default=8,
        help="CPU game workers within each frozen scenario evaluation.",
    )
    parser.add_argument(
        "--eval-scenario-workers", type=int, default=4,
        help="Frozen scenario evaluations to run concurrently per training seed.",
    )
    parser.add_argument(
        "--parallel-seeds", action="store_true",
        help="Train independent seeds concurrently on multi-core CPU hosts.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=("Continue an existing output directory from its latest per-seed "
              "checkpoint; --rounds is the target total episode count."),
    )
    parser.add_argument(
        "--initial-checkpoints", type=Path, nargs="+",
        help=("Initialize a new run from one checkpoint per training seed. "
              "Checkpoint filenames must be episode_NNNN.pkl."),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.curriculum == "Final_finetune":
        args.curriculum = "final-finetune"
    if args.worker:
        return args
    if args.output is None:
        parser.error("Choose an --output directory.")
    if args.resume:
        if not args.output.is_dir():
            parser.error("--resume requires an existing --output directory.")
    elif args.output.exists():
        parser.error("Choose a new --output directory, or pass --resume.")
    if args.resume and args.initial_checkpoints:
        parser.error("Choose --resume or --initial-checkpoints, not both.")
    if args.initial_checkpoints:
        if len(args.initial_checkpoints) != len(args.seeds):
            parser.error("--initial-checkpoints requires one path per training seed.")
        for checkpoint in args.initial_checkpoints:
            if not checkpoint.is_file():
                parser.error(f"Initial checkpoint does not exist: {checkpoint}")
            if not checkpoint.stem.startswith("episode_"):
                parser.error("Initial checkpoint must be named episode_NNNN.pkl")
    if len(args.opponents) > 3:
        parser.error("At most three opponents are allowed.")
    if args.curriculum in {"mixed", "staged-combat", *TOURNAMENT_CURRICULA,
                           *LEAGUE_CURRICULA} and args.opponents:
        parser.error("Curriculum schedules manage opponents internally; omit --opponents.")
    if not (SOURCE_DIR / "agent_code" / args.agent / "callbacks.py").is_file():
        parser.error(f"Unknown training agent: {args.agent}")
    if min(args.rounds, args.max_steps, args.eval_every,
           args.eval_workers, args.eval_scenario_workers) < 1:
        parser.error("Round and step counts must be positive.")
    if args.board_stride is not None and args.board_stride < args.rounds:
        parser.error("--board-stride must be at least --rounds.")
    for values in (args.seeds, args.eval_seeds, args.eval_seats):
        if len(set(values)) != len(values):
            parser.error("Seed and seat lists may not contain duplicates.")
    for opponent in args.opponents:
        if not (SOURCE_DIR / "agent_code" / opponent / "callbacks.py").is_file():
            parser.error(f"Unknown opponent: {opponent}")
    if args.curriculum in {"staged-combat", *TOURNAMENT_CURRICULA}:
        if len(args.classic_opponents) > 3:
            parser.error("At most three staged classic opponents are supported.")
        for opponent in args.classic_opponents:
            if not (SOURCE_DIR / "agent_code" / opponent / "callbacks.py").is_file():
                parser.error(f"Unknown staged classic opponent: {opponent}")
    if args.classic_lineup is not None:
        if args.curriculum not in (TOURNAMENT_CURRICULA | LEAGUE_CURRICULA):
            parser.error("--classic-lineup requires a tournament or league curriculum")
        if not args.classic_lineup:
            parser.error("At least one classic lineup is required")
        for lineup in args.classic_lineup:
            if not 1 <= len(lineup) <= 3:
                parser.error("Each classic lineup must contain one to three opponents")
            for opponent in lineup:
                if not (SOURCE_DIR / "agent_code" / opponent / "callbacks.py").is_file():
                    parser.error(f"Unknown classic lineup opponent: {opponent}")
    if args.curriculum in (TOURNAMENT_CURRICULA | LEAGUE_CURRICULA) and args.classic_lineup is None:
        if args.curriculum in COMPLETE_LINEUP_CURRICULA:
            args.classic_lineup = [["rule_based_agent"] * 3]
        else:
            args.classic_lineup = [
                ["peaceful_agent"],
                ["coin_collector_agent"],
                ["rule_based_agent"],
                ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
            ]
    args.classic_lineups = args.classic_lineup
    if args.curriculum in COMPLETE_LINEUP_CURRICULA:
        for lineup in args.classic_lineups:
            if len(lineup) != 3:
                parser.error(
                    f"{args.curriculum} requires complete three-opponent Classic lineups"
                )
            forbidden = set(lineup) & FINAL_FINETUNE_FORBIDDEN_OPPONENTS
            if forbidden:
                parser.error(
                    f"{args.curriculum} excludes " + ", ".join(sorted(forbidden))
                )
    if args.league_eval_lineup is not None and args.curriculum not in LEAGUE_CURRICULA:
        parser.error("--league-eval-lineup requires a league curriculum")
    if args.curriculum in LEAGUE_CURRICULA:
        args.league_eval_lineups = args.league_eval_lineup or [
            ["rule_based_agent"] * 3,
        ]
        for lineup in args.league_eval_lineups:
            if not 1 <= len(lineup) <= 3:
                parser.error("League evaluation lineups require one to three opponents")
            for opponent in lineup:
                if not (SOURCE_DIR / "agent_code" / opponent / "callbacks.py").is_file():
                    parser.error(f"Unknown league evaluation opponent: {opponent}")
    else:
        args.league_eval_lineups = None
    board_stride = args.board_stride or args.rounds
    training_boards = {
        board
        for index in range(len(args.seeds))
        for board in range(4000 + index * board_stride,
                           4000 + index * board_stride + args.rounds)
    }
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
        for name in ("agent", "scenario", "curriculum", "opponents",
                     "classic_opponents", "classic_lineups",
                     "league_eval_lineups"):
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
    if args.initial_checkpoints:
        config["initial_checkpoints"] = [
            str(path.resolve()) for path in args.initial_checkpoints
        ]
    config["evaluation_scenarios"] = (
        ["coin-heaven", "loot-crate"]
        if args.curriculum in {"mixed", "staged-combat", *TOURNAMENT_CURRICULA,
                               *LEAGUE_CURRICULA}
        else [args.scenario]
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
        "Agent_034_compact_fqi_agent",
    }:
        # These variants intentionally import the already audited combat
        # feature and safety implementations; hash those dependencies for
        # provenance.
        source_paths.extend([
            SOURCE_DIR / "agent_code" / "combat_fqi_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "combat_fqi_agent" / "safety.py",
        ])
    if args.agent in {
            "Agent_035_compact_fqi_symmetry_agent",
            "Agent_036_compact_fqi_robust_agent"}:
        source_paths.extend([
            SOURCE_DIR / "agent_code" / args.agent / "symmetry.py",
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
        "Agent_029_combat_ddqn_adversarial_window_agent",
        "Agent_030_combat_ddqn_escape_replay_agent",
        "Agent_031_combat_ddqn_offensive_escape_agent",
        "Agent_032_combat_ddqn_optimized_features_agent",
        "Agent_037_tournament_fast_ddqn_agent",
        "Agent_038_symmetric_population_ddqn_agent",
        "Agent_039_compact_audit_ddqn_agent",
        "Agent_040_optimized_compact_ddqn_agent",
        "Agent_041_dynamic_nav_ddqn_agent",
        "Agent_042_combat_progress_ddqn_agent",
        "Agent_043_novelty_credit_ddqn_agent",
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
            SOURCE_DIR / "agent_code" / "combat_fqi_history_antistag_agent" / "train.py",
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
    if args.agent in {
        "Agent_029_combat_ddqn_adversarial_window_agent",
        "Agent_030_combat_ddqn_escape_replay_agent",
        "Agent_031_combat_ddqn_offensive_escape_agent",
        "Agent_032_combat_ddqn_optimized_features_agent",
        "Agent_037_tournament_fast_ddqn_agent",
        "Agent_038_symmetric_population_ddqn_agent",
        "Agent_039_compact_audit_ddqn_agent",
        "Agent_040_optimized_compact_ddqn_agent",
        "Agent_041_dynamic_nav_ddqn_agent",
        "Agent_042_combat_progress_ddqn_agent",
        "Agent_043_novelty_credit_ddqn_agent",
    }:
        source_paths.extend([
            SOURCE_DIR / "agent_code" / args.agent / "replay.py",
            SOURCE_DIR / "agent_code" / "Agent_027_combat_ddqn_short_cycle_staged_replay_agent" / "train.py",
            SOURCE_DIR / "agent_code" / "Agent_027_combat_ddqn_short_cycle_staged_replay_agent" / "replay.py",
            SOURCE_DIR / "agent_code" / "Agent_025_combat_ddqn_short_cycle_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "combat_dqn_r_topology_agent" / "features.py",
        ])
    if args.agent in {"Agent_030_combat_ddqn_escape_replay_agent",
                      "Agent_031_combat_ddqn_offensive_escape_agent",
                      "Agent_032_combat_ddqn_optimized_features_agent"}:
        source_paths.extend([
            SOURCE_DIR / "agent_code" / "Agent_029_combat_ddqn_adversarial_window_agent" / "callbacks.py",
            SOURCE_DIR / "agent_code" / "Agent_029_combat_ddqn_adversarial_window_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "Agent_029_combat_ddqn_adversarial_window_agent" / "replay.py",
            SOURCE_DIR / "agent_code" / "Agent_030_combat_ddqn_escape_replay_agent" / "train.py",
            SOURCE_DIR / "agent_code" / "Agent_030_combat_ddqn_escape_replay_agent" / "replay.py",
        ])
    if args.agent in {
        "Agent_037_tournament_fast_ddqn_agent",
        "Agent_038_symmetric_population_ddqn_agent",
        "Agent_039_compact_audit_ddqn_agent",
        "Agent_040_optimized_compact_ddqn_agent",
        "Agent_041_dynamic_nav_ddqn_agent",
        "Agent_042_combat_progress_ddqn_agent",
        "Agent_043_novelty_credit_ddqn_agent",
    }:
        source_paths.extend([
            SOURCE_DIR / "agent_code" / "Agent_029_combat_ddqn_adversarial_window_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "Agent_030_combat_ddqn_escape_replay_agent" / "train.py",
            SOURCE_DIR / "agent_code" / "Agent_032_combat_ddqn_optimized_features_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "Agent_032_combat_ddqn_optimized_features_agent" / "replay.py",
        ])
    if args.agent in {
        "Agent_038_symmetric_population_ddqn_agent",
        "Agent_039_compact_audit_ddqn_agent",
        "Agent_040_optimized_compact_ddqn_agent",
        "Agent_041_dynamic_nav_ddqn_agent",
        "Agent_042_combat_progress_ddqn_agent",
        "Agent_043_novelty_credit_ddqn_agent",
    }:
        source_paths.extend([
            SOURCE_DIR / "agent_code" / args.agent / "checkpoint.py",
            SOURCE_DIR / "agent_code" / args.agent / "symmetry.py",
            SOURCE_DIR / "agent_code" / "Agent_036_compact_fqi_robust_agent" / "features.py",
            SOURCE_DIR / "agent_code" / "Agent_036_compact_fqi_robust_agent" / "safety.py",
            SOURCE_DIR / "agent_code" / "Agent_037_tournament_fast_ddqn_agent" / "features.py",
        ])
    if args.agent == "Agent_041_dynamic_nav_ddqn_agent":
        source_paths.append(
            SOURCE_DIR / "agent_code" /
            "Agent_040_optimized_compact_ddqn_agent" / "features.py"
        )
    if args.agent == "Agent_042_combat_progress_ddqn_agent":
        source_paths.extend([
            SOURCE_DIR / "agent_code" /
            "Agent_040_optimized_compact_ddqn_agent" / "features.py",
            SOURCE_DIR / "agent_code" /
            "Agent_040_optimized_compact_ddqn_agent" / "symmetry.py",
            SOURCE_DIR / "agent_code" /
            "Agent_040_optimized_compact_ddqn_agent" / "replay.py",
            SOURCE_DIR / "agent_code" /
            "Agent_041_dynamic_nav_ddqn_agent" / "features.py",
        ])
    if args.agent == "Agent_043_novelty_credit_ddqn_agent":
        source_paths.extend([
            SOURCE_DIR / "agent_code" /
            "Agent_040_optimized_compact_ddqn_agent" / "features.py",
            SOURCE_DIR / "agent_code" /
            "Agent_040_optimized_compact_ddqn_agent" / "replay.py",
            SOURCE_DIR / "agent_code" /
            "Agent_040_optimized_compact_ddqn_agent" / "symmetry.py",
            SOURCE_DIR / "agent_code" /
            "Agent_041_dynamic_nav_ddqn_agent" / "features.py",
            SOURCE_DIR / "agent_code" /
            "Agent_042_combat_progress_ddqn_agent" / "callbacks.py",
            SOURCE_DIR / "agent_code" /
            "Agent_042_combat_progress_ddqn_agent" / "checkpoint.py",
            SOURCE_DIR / "agent_code" /
            "Agent_042_combat_progress_ddqn_agent" / "features.py",
            SOURCE_DIR / "agent_code" /
            "Agent_042_combat_progress_ddqn_agent" / "replay.py",
            SOURCE_DIR / "agent_code" /
            "Agent_042_combat_progress_ddqn_agent" / "symmetry.py",
            SOURCE_DIR / "agent_code" /
            "Agent_042_combat_progress_ddqn_agent" / "train.py",
        ])
    if args.agent == "Agent_031_combat_ddqn_offensive_escape_agent":
        source_paths.append(SOURCE_DIR / "prepare_offensive_checkpoint.py")
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
            checkpoint = (
                args.initial_checkpoints[index].resolve()
                if args.initial_checkpoints else None
            )
            if checkpoint is not None:
                try:
                    checkpoint_episode = int(checkpoint.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError) as error:
                    raise ValueError(
                        f"Invalid initial checkpoint filename: {checkpoint}"
                    ) from error
                if checkpoint_episode >= args.rounds:
                    raise ValueError(
                        f"Initial checkpoint {checkpoint} is already at episode "
                        f"{checkpoint_episode}; --rounds must be larger."
                    )
            else:
                checkpoint_episode = 0
            previous_seed_config = {}
        run_config = config.copy()
        run_config["seed"] = seed
        run_config["output"] = str(directory)
        run_config["board_start"] = previous_seed_config.get(
            "board_start", 4000 + index * (args.board_stride or args.rounds)
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
