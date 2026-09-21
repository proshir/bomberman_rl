"""Measure uncached Agent 036 inference latency on dense synthetic boards.

The framework's action cache is intentionally bypassed before every call.
This measures route generation, safety searches, and frozen-tree selection
inside the normal callbacks.act() path on one CPU.
"""

import argparse
import json
import pickle
import tempfile
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

import settings as s
from agent_code.Agent_036_compact_fqi_robust_agent import callbacks, train


def dense_state(index):
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    # Leave a broad connected component while exposing many bomb targets.
    for x in range(3, 15, 3):
        for y in range(2, 15, 3):
            if (x, y) not in {(1, 1), (15, 15), (8, 8)}:
                field[x, y] = 1
    self_position = (1 + (index % 3), 1 + ((index // 3) % 3))
    field[self_position] = 0
    others = [
        ("enemy-a", 0, True, (15, 15)),
        ("enemy-b", 0, True, (15, 1)),
    ]
    bombs = [
        ((8, 8), 1 + index % 3),
        ((12, 12), 2 + (index % 2)),
    ]
    explosion_map = np.zeros_like(field)
    explosion_map[7, 8] = 1
    explosion_map[8, 7] = 2
    return {
        "round": 1,
        "step": index + 1,
        "field": field,
        "self": ("learner", 0, True, self_position),
        "others": others,
        "bombs": bombs,
        "coins": [(13, 13), (14, 13), (13, 14)],
        "explosion_map": explosion_map,
    }


def _write_empty_checkpoint(path):
    with Path(path).open("wb") as stream:
        pickle.dump({
            "schemas": train.CHECKPOINT_SCHEMAS,
            "trees": [None] * len(callbacks.ACTIONS),
        }, stream, protocol=pickle.HIGHEST_PROTOCOL)


def measure(checkpoint, samples):
    agent = SimpleNamespace(train=False, seed=0, model_path=checkpoint)
    callbacks.setup(agent)
    timings = []
    for index in range(samples):
        state = dense_state(index)
        # Force a fresh decision-time feature/safety computation.
        agent.observations.clear()
        agent.round_id = None
        agent.positions.clear()
        agent.last_progress = None
        started = perf_counter()
        callbacks.act(agent, state)
        timings.append(perf_counter() - started)
    milliseconds = np.asarray(timings, dtype=float) * 1000.0
    return {
        "samples": int(samples),
        "scenario": "dense-crates-bombs-opponents",
        "cache": "cleared before every act",
        "cpu": "single process; set thread env vars externally",
        "median_ms": float(np.median(milliseconds)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "max_ms": float(np.max(milliseconds)),
        "min_ms": float(np.min(milliseconds)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.checkpoint is not None:
        result = measure(args.checkpoint, args.samples)
    else:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "empty.pkl"
            _write_empty_checkpoint(checkpoint)
            result = measure(checkpoint, args.samples)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
