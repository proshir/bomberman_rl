"""Checkpoint widening retained for training-time warm starts."""

import copy

import torch

from agent_code.Agent_042_combat_progress_ddqn_agent.checkpoint import (
    expand_checkpoint as _expand_agent042,
)

from .features import AGENT_042_FEATURE_SIZE, FEATURE_SIZE


def expand_checkpoint(checkpoint):
    old_dim = int(checkpoint["input_dim"])
    if old_dim == FEATURE_SIZE:
        return checkpoint
    result = _expand_agent042(checkpoint)
    current_dim = int(result["input_dim"])
    if current_dim != AGENT_042_FEATURE_SIZE:
        raise ValueError(
            f"Agent 043 expected the Agent 042 migration to produce "
            f"{AGENT_042_FEATURE_SIZE} inputs, got {current_dim}"
        )
    result = copy.deepcopy(result)
    added = FEATURE_SIZE - current_dim
    first_weight = "network.0.weight"
    old_shape = result["policy_state_dict"][first_weight].shape

    def pad(tensor):
        if tensor.ndim != 2 or tensor.shape[1] != current_dim:
            raise ValueError(
                "Unexpected Agent 042 first-layer checkpoint shape: "
                f"{tuple(tensor.shape)}"
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
    result["input_dim"] = FEATURE_SIZE
    result["migration"] = (
        f"{old_dim}-to-{FEATURE_SIZE} zero-padded Agent043 novelty columns"
    )
    return result


__all__ = ["expand_checkpoint"]
