"""Measure Agent032 replay sampling and CPU/device batch handoff."""

import argparse
import json
from time import perf_counter

import numpy as np
import torch

from agent_code.Agent_030_combat_ddqn_escape_replay_agent.replay import (
    CombatEscapeReplayBuffer as OldReplay,
)
from agent_code.Agent_032_combat_ddqn_optimized_features_agent.replay import (
    CombatEscapeReplayBuffer as NewReplay,
)


def fill(buffer, count):
    state = np.arange(113, dtype=np.float32)
    next_state = state + 1.0
    mask = np.ones(6, dtype=bool)
    buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
    for index in range(count):
        buffer.transition_tag = "combat_escape" if index % 5 == 0 else None
        buffer.add(state, index % 6, float(index), next_state, False, mask)


def old_batch(buffer, batch_size, device):
    batch = buffer.sample(batch_size)
    dtypes = (torch.float32, torch.long, torch.float32, torch.float32,
              torch.float32, torch.bool)
    return tuple(
        torch.as_tensor(array, dtype=dtype, device=device)
        for array, dtype in zip(batch, dtypes)
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(function, repeats, device):
    for _ in range(20):
        function()
    synchronize(device)
    started = perf_counter()
    for _ in range(repeats):
        function()
    synchronize(device)
    return (perf_counter() - started) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--transitions", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=300)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    old = OldReplay(args.transitions, seed=11)
    new = NewReplay(args.transitions, seed=11)
    fill(old, args.transitions)
    fill(new, args.transitions)
    old_seconds = timed(
        lambda: old_batch(old, 128, device), args.repeats, device
    )
    new_seconds = timed(
        lambda: new.sample_torch(128, device), args.repeats, device
    )
    old_payload = sum(
        item.nbytes for row in old.storage for item in row
        if hasattr(item, "nbytes")
    )
    new_payload = sum(array.nbytes for array in (
        new.states, new.next_states, new.actions, new.rewards,
        new.dones, new.next_action_masks,
    ))
    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "old_ms_per_batch": 1000.0 * old_seconds,
        "new_ms_per_batch": 1000.0 * new_seconds,
        "speedup": old_seconds / new_seconds,
        "old_numpy_payload_bytes": old_payload,
        "new_preallocated_payload_bytes": new_payload,
        "transitions": args.transitions,
        "repeats": args.repeats,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
