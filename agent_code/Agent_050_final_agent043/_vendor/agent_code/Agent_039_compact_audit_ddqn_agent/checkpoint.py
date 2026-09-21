"""Checkpoint migration from Agent 038's 117-input schema to Agent 039."""

import argparse
import copy
import hashlib
from pathlib import Path

import torch

from .features import AGENT_038_FEATURE_SIZE, FEATURE_SIZE


def _column_mapping():
    """Return old 117-input columns retained by the 104-input schema."""
    return (
        list(range(0, 29))
        + list(range(30, 36))
        + list(range(37, 65))
        + [65 + 6 * action + offset
           for action in range(6) for offset in range(2, 6)]
        + list(range(101, 117))
    )


def compact_agent038_checkpoint(checkpoint):
    """Convert a 117-input policy checkpoint to the 104-input contract.

    The new global armed-opponent column is initialized to the mean of the six
    old action-conditioned armed-opponent columns. This is only an initializer;
    the primary Agent 039 comparison should still be scratch-to-scratch.
    """
    old_dim = int(checkpoint["input_dim"])
    if old_dim == FEATURE_SIZE:
        return checkpoint
    if old_dim != AGENT_038_FEATURE_SIZE:
        raise ValueError(
            f"Agent 039 accepts {FEATURE_SIZE}- or {AGENT_038_FEATURE_SIZE}-input "
            f"checkpoints, got {old_dim}"
        )

    result = copy.deepcopy(checkpoint)
    mapping = _column_mapping()
    old_shape = result["policy_state_dict"]["network.0.weight"].shape

    def compact_weight(weight):
        retained = weight[:, mapping]
        armed_columns = torch.stack(
            [weight[:, 66 + 6 * action] for action in range(6)], dim=1
        )
        global_column = armed_columns.mean(dim=1, keepdim=True)
        # The retained columns currently end with the route suffix. Insert the
        # replacement global flag immediately before the response blocks.
        return torch.cat((retained[:,:63], global_column, retained[:,63:]), dim=1)

    for name in ("policy_state_dict", "target_state_dict"):
        result[name]["network.0.weight"] = compact_weight(
            result[name]["network.0.weight"]
        )

    optimizer = result.get("optimizer_state_dict", {})
    groups = optimizer.get("param_groups", [])
    if groups and groups[0].get("params"):
        first_parameter = groups[0]["params"][0]
        state = optimizer.get("state", {}).get(first_parameter, {})
        for name, value in list(state.items()):
            if torch.is_tensor(value) and value.shape == old_shape:
                state[name] = compact_weight(value)

    result["input_dim"] = FEATURE_SIZE
    result["migration"] = (
        "Agent 038 117-to-Agent 039 104 compact audit schema; "
        "constant/duplicate columns removed and global armed-opponent column "
        "initialized from six old columns"
    )
    return result


def policy_only_warm_start(checkpoint, source=None):
    result = compact_agent038_checkpoint(checkpoint)
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
    result["warm_start"] = "Agent 038 policy weights with Agent 039 input migration"
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


__all__ = [
    "compact_agent038_checkpoint", "policy_only_warm_start",
]


if __name__ == "__main__":
    main()
