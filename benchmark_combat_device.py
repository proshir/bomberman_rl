"""Short CPU/CUDA timing diagnostic using the existing combat training loop.

Launch each device in a fresh process with CUDA_VISIBLE_DEVICES set explicitly.
Checkpoint evaluation is skipped; rewards, warmup, updates and episodes are
unchanged. Timing checkpoints are written only below the requested output.
"""

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import run_combat_training as runner
from agent_code.combat_dqn_agent import train as base_train
from agent_code.combat_dqn_agent.model import DEVICE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()
    if DEVICE.type != args.device:
        raise RuntimeError(f"Requested {args.device}, detected {DEVICE}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    config.update(output=str(args.output.resolve()), rounds=600 + args.rounds,
                  start_episode=600, eval_every=args.rounds + 601,
                  resume_checkpoint=config["initial_checkpoints"][config["seed"]])
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    measured = {"features_seconds": 0.0, "optimize_seconds": 0.0,
                "features_calls": 0, "optimizer_calls": 0}
    captured = {}
    feature_module = importlib.import_module(f"agent_code.{config['agent']}.features")
    original_features = feature_module.state_to_features
    original_optimize = base_train.optimize_model

    def timed_features(*positional, **keywords):
        started = perf_counter()
        result = original_features(*positional, **keywords)
        measured["features_seconds"] += perf_counter() - started
        measured["features_calls"] += 1
        return result

    def timed_optimize(learner):
        started = perf_counter()
        result = original_optimize(learner)
        measured["optimize_seconds"] += perf_counter() - started
        measured["optimizer_calls"] += result is not None
        return result

    def capture_checkpoint(config, learner, episode, interactions):
        captured["learner"] = learner
        captured.setdefault("initial_optimizer_steps", learner.optimizer_steps)
        return []

    feature_module.state_to_features = timed_features
    base_train.optimize_model = timed_optimize
    runner.evaluate_all = capture_checkpoint
    started = perf_counter()
    runner.train(config)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    wall = perf_counter() - started
    records = [json.loads(line) for line in
               (args.output / "rounds.jsonl").read_text().splitlines()]
    learner = captured["learner"]
    updates = learner.optimizer_steps - captured["initial_optimizer_steps"]
    if updates == 0:
        raise RuntimeError("Timing run did not pass replay warmup; increase --rounds")

    # Time the real optimizer on identical deterministic synthetic replay batches
    # to separate hardware overhead from diverging CPU/CUDA game trajectories.
    from agent_code.combat_dqn_agent.replay import ReplayBuffer
    from agent_code.combat_dqn_agent.model import load_checkpoint
    checkpoint = load_checkpoint(config["resume_checkpoint"])
    learner.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    learner.target_net.load_state_dict(checkpoint["target_state_dict"])
    learner.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    learner.optimizer_steps = int(checkpoint["optimizer_steps"])
    learner.replay_buffer = ReplayBuffer(6000, 123)
    rng = np.random.default_rng(123)
    dimension = learner.policy_net.input_dim
    for _ in range(5000):
        learner.replay_buffer.add(rng.normal(size=dimension).astype(np.float32),
                                 int(rng.integers(6)), float(rng.normal()),
                                 rng.normal(size=dimension).astype(np.float32),
                                 False, np.ones(6, dtype=bool))
    for _ in range(30):
        original_optimize(learner)
    samples = []
    for _ in range(3):
        started = perf_counter()
        for _ in range(200):
            original_optimize(learner)
        samples.append((perf_counter() - started) / 200)
    result = {
        "host": platform.node(), "torch": torch.__version__, "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=runner.SOURCE_DIR, text=True).strip(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(
            Path(config["resume_checkpoint"]).read_bytes()).hexdigest(),
        "agent": config["agent"], "rounds": args.rounds,
        "learner_transitions": learner.combat_env_steps,
        "world_steps": sum(record["steps"] for record in records),
        "optimizer_updates": updates, "wall_seconds": wall,
        "training_seconds": records[-1]["training_seconds"],
        "training_ms_per_transition": 1000 * records[-1]["training_seconds"] / learner.combat_env_steps,
        "micro_optimizer_ms": [1000 * sample for sample in samples],
        **measured,
    }
    (args.output / "timing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
