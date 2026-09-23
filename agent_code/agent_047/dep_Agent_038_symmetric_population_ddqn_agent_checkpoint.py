"""Checkpoint compatibility and policy-only Agent 038 warm starts."""

import argparse
import copy
import hashlib
from pathlib import Path

import torch

from .dep_Agent_038_symmetric_population_ddqn_agent_features import AGENT_037_FEATURE_SIZE, FEATURE_SIZE


def expand_agent037_checkpoint(checkpoint):
    """Zero-pad 101-input first-layer weights and Adam moments to 117.

    Zero columns make the migrated policy and target networks exactly preserve
    their Agent 037 Q-values before Agent 038 learns to use the new suffix.
    Already-migrated 117-input checkpoints are returned unchanged.
    """
    old_dim = int(checkpoint["input_dim"])
    if old_dim == FEATURE_SIZE:
        return checkpoint
    if old_dim != AGENT_037_FEATURE_SIZE:
        raise ValueError(
            f"Agent 038 accepts {AGENT_037_FEATURE_SIZE}- or "
            f"{FEATURE_SIZE}-input checkpoints, got {old_dim}"
        )

    result = copy.deepcopy(checkpoint)
    added = FEATURE_SIZE - old_dim

    def pad(tensor):
        if tensor.ndim != 2 or tensor.shape[1] != old_dim:
            raise ValueError(
                "Unexpected Agent 037 first-layer checkpoint shape: "
                f"{tuple(tensor.shape)}"
            )
        return torch.cat(
            (tensor, tensor.new_zeros((tensor.shape[0], added))), dim=1
        )

    first_weight = "network.0.weight"
    old_shape = result["policy_state_dict"][first_weight].shape
    for name in ("policy_state_dict", "target_state_dict"):
        result[name][first_weight] = pad(result[name][first_weight])

    optimizer = result.get("optimizer_state_dict", {})
    groups = optimizer.get("param_groups", [])
    if groups and groups[0].get("params"):
        first_parameter = groups[0]["params"][0]
        state = optimizer.get("state", {}).get(first_parameter, {})
        for name, value in list(state.items()):
            if torch.is_tensor(value) and value.shape == old_shape:
                state[name] = pad(value)

    result["input_dim"] = FEATURE_SIZE
    result["migration"] = (
        f"{old_dim}-to-{FEATURE_SIZE} zero route columns; "
        "Adam moments preserved and padded"
    )
    return result


def policy_only_warm_start(checkpoint, source=None):
    """Keep learned policy weights while resetting all training state."""
    result = expand_agent037_checkpoint(checkpoint)
    result = copy.deepcopy(result)
    result["target_state_dict"] = copy.deepcopy(result["policy_state_dict"])
    optimizer = result["optimizer_state_dict"]
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = 1e-4
        if "initial_lr" in group:
            group["initial_lr"] = 1e-4
    result["epsilon"] = 0.30
    result["env_steps"] = 0
    result["combat_env_steps"] = 0
    result["optimizer_steps"] = 0
    result["warm_start"] = (
        "Agent 037 policy weights; target cloned; Adam/replay/counters/"
        "exploration reset for Agent 038 population training"
    )
    if source is not None:
        source = Path(source).resolve()
        result["warm_start_source"] = str(source)
        result["warm_start_source_sha256"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    result = policy_only_warm_start(checkpoint, args.source)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("xb") as output:
        torch.save(result, output)
    print(args.destination)


__all__ = ["expand_agent037_checkpoint", "policy_only_warm_start"]


if __name__ == "__main__":
    main()
