"""Extend a 101-input checkpoint with zero-weight offensive inputs."""

import argparse
import copy
import hashlib
from pathlib import Path

import torch


def expand_checkpoint(checkpoint, input_dim=113):
    result = copy.deepcopy(checkpoint)
    old_dim = int(result["input_dim"])
    if old_dim != 101 or input_dim != 113:
        raise ValueError("Only the controlled 101-to-113 input extension is supported")
    shape = result["policy_state_dict"]["network.0.weight"].shape

    def pad(tensor):
        return torch.cat((tensor, tensor.new_zeros((shape[0], input_dim - old_dim))), dim=1)

    for name in ("policy_state_dict", "target_state_dict"):
        result[name]["network.0.weight"] = pad(result[name]["network.0.weight"])
    optimizer = result["optimizer_state_dict"]
    first_parameter = optimizer["param_groups"][0]["params"][0]
    for name, value in optimizer["state"].get(first_parameter, {}).items():
        if torch.is_tensor(value) and value.shape == shape:
            optimizer["state"][first_parameter][name] = pad(value)
    result["input_dim"] = input_dim
    result["migration"] = "101-to-113 zero columns; Adam moments preserved and padded"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    result = expand_checkpoint(checkpoint)
    result["initial_checkpoint_source"] = str(args.source.resolve())
    result["initial_checkpoint_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("xb") as output:
        torch.save(result, output)
    print(args.destination)


if __name__ == "__main__":
    main()
