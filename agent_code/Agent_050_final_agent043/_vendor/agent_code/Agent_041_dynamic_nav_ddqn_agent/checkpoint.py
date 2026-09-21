"""Checkpoint widening helpers for optional Agent040 warm starts."""

import copy

import torch

from .features import AGENT_040_FEATURE_SIZE, FEATURE_SIZE


def expand_agent040_checkpoint(checkpoint):
    """Zero-pad Agent040 first-layer weights for Agent041 inputs.

    This preserves the old Q-values when the new navigation columns are zero.
    Scratch training remains the primary comparison because the new columns
    are intended to be learned, not merely initialized.
    """
    old_dim = int(checkpoint["input_dim"])
    if old_dim == FEATURE_SIZE:
        return checkpoint
    if old_dim != AGENT_040_FEATURE_SIZE:
        raise ValueError(
            f"Agent 041 accepts {AGENT_040_FEATURE_SIZE}- or "
            f"{FEATURE_SIZE}-input checkpoints, got {old_dim}"
        )

    result = copy.deepcopy(checkpoint)
    added = FEATURE_SIZE - old_dim

    def pad(tensor):
        if tensor.ndim != 2 or tensor.shape[1] != old_dim:
            raise ValueError(
                "Unexpected Agent040 first-layer checkpoint shape: "
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
        f"{old_dim}-to-{FEATURE_SIZE} zero-padded dynamic navigation columns"
    )
    return result


__all__ = ["expand_agent040_checkpoint"]
