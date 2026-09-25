"""Checkpoint widening for optional training-time warm starts."""

import copy

import torch

from .features import (
    AGENT_040_FEATURE_SIZE, AGENT_041_FEATURE_SIZE, AGENT_042_FEATURE_SIZE,
    FEATURE_SIZE,
)


def _pad_inputs(checkpoint, new_size, label):
    old_size = int(checkpoint["input_dim"])
    result = copy.deepcopy(checkpoint)
    added = new_size - old_size
    first_weight = "network.0.weight"
    old_shape = result["policy_state_dict"][first_weight].shape

    def pad(tensor):
        if tensor.ndim != 2 or tensor.shape[1] != old_size:
            raise ValueError(
                f"Unexpected first-layer checkpoint shape: {tuple(tensor.shape)}"
            )
        return torch.cat(
            (tensor, tensor.new_zeros((tensor.shape[0], added))), dim=1
        )

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

    result["input_dim"] = new_size
    result["migration"] = f"{old_size}-to-{new_size} zero-padded {label} inputs"
    return result


def expand_checkpoint(checkpoint):
    """Widen supported earlier policy inputs while preserving old Q values."""
    old_size = int(checkpoint["input_dim"])
    if old_size == FEATURE_SIZE:
        return checkpoint
    if old_size not in (
        AGENT_040_FEATURE_SIZE, AGENT_041_FEATURE_SIZE,
        AGENT_042_FEATURE_SIZE,
    ):
        raise ValueError(
            f"Agent 047 accepts {AGENT_040_FEATURE_SIZE}, "
            f"{AGENT_041_FEATURE_SIZE}, {AGENT_042_FEATURE_SIZE}, or "
            f"{FEATURE_SIZE}-input checkpoints; got {old_size}"
        )
    if old_size != AGENT_042_FEATURE_SIZE:
        checkpoint = _pad_inputs(checkpoint, AGENT_042_FEATURE_SIZE, "Agent 042")
    return _pad_inputs(checkpoint, FEATURE_SIZE, "Agent 043 novelty")


__all__ = ["expand_checkpoint"]
