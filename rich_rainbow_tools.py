"""Small reproducible utilities for Agent 023 development runs.

This module deliberately does not launch jobs: cluster launchers supply their
own GPU allocation, thread caps, scratch path, and resolved configuration.
"""

import argparse
import json
from pathlib import Path
from agent_code.Agent_023_spatial_hybrid_rainbow_agent.config import FEATURE_SIZE, QUANTILES, RESIDUAL_BLOCKS, TRUNK_WIDTH

def write_resolved_config(path, **overrides):
    config = {"agent": "Agent_023_spatial_hybrid_rainbow_agent", "feature_size": FEATURE_SIZE, "spatial_channels": 20, "global_features": 36, "action_features_per_action": 16, "width": TRUNK_WIDTH, "residual_blocks": RESIDUAL_BLOCKS, "quantiles": QUANTILES, "overrides": overrides}
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    resolved = sub.add_parser("write-resolved-config"); resolved.add_argument("path"); resolved.add_argument("--seed", type=int, default=0); resolved.add_argument("--actors", type=int, default=1)
    args = parser.parse_args()
    if args.command == "write-resolved-config": write_resolved_config(args.path, seed=args.seed, actors=args.actors)

if __name__ == "__main__": main()
