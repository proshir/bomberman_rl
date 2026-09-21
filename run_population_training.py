"""Train two Agent 033 population members with one active learner per game.

Each episode selects one member to learn.  The other member may appear as a
frozen checkpoint opponent; imported and built-in opponents fill the remaining
seats.  This keeps the population experiment separate from the single-agent
training runner and from all earlier agents.
"""

from __future__ import annotations

import json
import random
from argparse import ArgumentParser
from pathlib import Path
import shutil
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

import settings as s
from run_benchmark import BenchmarkWorld


ACTIVE = "Agent_033_population_replay_agent"
FROZEN = "Agent_033_population_frozen_agent"
EXTERNAL = (
    "imp_li_deep_killer", "peaceful_agent", "coin_collector_agent",
    "rule_based_agent",
)


def _args(seed, seat, scenario, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "agents").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        seed=seed, agent_seed=seed, seat=seat, scenario=scenario,
        no_gui=True, save_replay=False, save_stats=False, match_name=None,
        continue_without_training=False, silence_errors=False,
        log_dir=str(log_dir), agent_log_dir=str(log_dir / "agents"),
    )


def _new_member(seed, checkpoint, output):
    """Create a learner once and save its initialized network."""
    from agent_code.Agent_033_population_replay_agent import callbacks

    callbacks.MODEL_PATH = Path(checkpoint).resolve()
    callbacks.RESUME_PATH = None
    world = BenchmarkWorld(
        _args(seed, 0, "coin-heaven", output / f"init_{seed}"),
        [(ACTIVE, True)],
    )
    learner = world.agents[0].backend.runner.fake_self
    callbacks.save_checkpoint(learner, checkpoint)
    world.end()
    return learner


def _make_world(active, frozen_checkpoint, lineup, episode, seed, output):
    """Create a fresh game and bind the persistent learner to its first seat."""
    from agent_code.Agent_033_population_replay_agent import callbacks as active_cb
    from agent_code.Agent_033_population_frozen_agent import callbacks as frozen_cb

    # The active runner is only a shell; its persistent fake_self is rebound
    # immediately below.  Use a unique nonexistent path so setup does not
    # collide with the persistent learner's checkpoint.
    active_cb.MODEL_PATH = (output / "shell" / f"episode_{episode:04d}.pt").resolve()
    frozen_cb.MODEL_PATH = Path(frozen_checkpoint).resolve()
    specs = [(ACTIVE, True)]
    for name in lineup:
        specs.append((FROZEN if name == FROZEN else name, False))
    scenario = "classic" if lineup else "coin-heaven"
    args = _args(seed, (episode - 1) % 4, scenario,
                 output / "logs" / f"episode_{episode:04d}")
    world = BenchmarkWorld(args, specs)
    world.agents[0].backend.runner.fake_self = active
    world.round = episode - 1
    world.rng = np.random.default_rng(seed * 1_000_003 + episode)
    return world, args


def _plan(episode, active_index, seed):
    """Return solo/combat scenario and a valid four-seat opponent lineup."""
    rng = random.Random((seed + 17) * 1_000_003 + episode)
    phase = (episode - 1) % 5
    if phase == 3:
        return "coin-heaven", []
    if phase == 4:
        return "loot-crate", []

    # Combat occupies 60% of rounds.  Half of combat games expose the active
    # member to the other member's current frozen checkpoint.
    use_frozen = rng.random() < 0.5
    lineup = [FROZEN] if use_frozen else []
    available = list(EXTERNAL)
    rng.shuffle(available)
    external_count = rng.randrange(0, 3 if use_frozen else 4)
    lineup.extend(available[:external_count])
    return "classic", lineup


def train_member_population(output, rounds, seeds=(0, 1), max_steps=400):
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = [output / "checkpoints" / f"member_{i}.pt" for i in range(2)]
    for path in checkpoints:
        path.parent.mkdir(parents=True, exist_ok=True)

    members = [
        _new_member(seeds[0], checkpoints[0], output),
        _new_member(seeds[1], checkpoints[1], output),
    ]
    s.MAX_STEPS = max_steps
    records_path = output / "rounds.jsonl"
    with records_path.open("w") as records:
        for episode in tqdm(range(1, rounds + 1), desc="Agent 033 population"):
            active_index = (episode - 1) % 2
            active = members[active_index]
            scenario, lineup = _plan(episode, active_index, seeds[active_index])
            if FROZEN in lineup:
                other = 1 - active_index
                frozen_checkpoint = checkpoints[other]
            else:
                # The value is unused when no frozen member is in the lineup.
                frozen_checkpoint = checkpoints[1 - active_index]
            started = perf_counter()
            world, args = _make_world(
                active, frozen_checkpoint, lineup, episode,
                seeds[active_index], output,
            )
            args.scenario = scenario
            world.new_round()
            while world.running:
                world.do_step()
            agent = world.agents[0]
            learner = agent.backend.runner.fake_self
            stats = agent.statistics
            record = {
                "episode": episode,
                "active_member": active_index,
                "scenario": scenario,
                "opponents": lineup,
                "seed": seeds[active_index],
                "steps": world.step,
                "score": agent.score,
                "coins": stats["coins"],
                "kills": stats["kills"],
                "suicides": stats["suicides"],
                "invalid_actions": stats["invalid"],
                "survived": not agent.dead,
                "replay_buffer_size": len(learner.replay_buffer),
                "optimizer_steps": int(learner.optimizer_steps),
                "epsilon": float(learner.epsilon),
                "wall_seconds": perf_counter() - started,
            }
            records.write(json.dumps(record) + "\n")
            records.flush()
            # The active member's end_of_round callback has already persisted
            # its model to its own checkpoint path.
            world.end()
            if episode % 100 == 0:
                for member_index, checkpoint in enumerate(checkpoints):
                    shutil.copy2(
                        checkpoint,
                        output / "checkpoints" /
                        f"member_{member_index}_episode_{episode:04d}.pt",
                    )


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=400)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.max_steps < 1:
        parser.error("rounds and max-steps must be positive")
    train_member_population(args.output.resolve(), args.rounds, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
