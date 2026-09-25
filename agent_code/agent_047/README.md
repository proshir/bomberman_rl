# agent_047 — standalone Agent 043-derived package

This directory contains Agent 047, built from Agent 043's observation
pipeline and DDQN policy. Its feature code is grouped into a compact encoder
and one navigation, combat-progress, and novelty layer. Training, replay,
safety, symmetry, and model code remain package-local. Agent 047 imports no
Python modules from other agent folders.

## Running

From the original framework root:

```bash
python main.py play --my-agent agent_047
```

The package loads the bundled Agent 043 seed-1 mixed-300 continuation
checkpoint from `training.pkl`. The path is resolved relative to
`callbacks.py`, so the package does not depend on the training machine or its
scratch filesystem.

## Final-project compliance

- The agent itself is machine-learning based: a PyTorch DDQN network is used
  for every action decision.
- `act` returns only framework actions and applies no multiprocessing or
  external worker process.
- Agent-specific callbacks, checkpoint handling, features, model, replay,
  safety, symmetry, and training code are package-local Python files.
- The compact feature encoder contains the helpers it needs directly, so the
  package does not carry unused historical feature wrappers.
- The runtime checkpoint is the bundled `training.pkl` file (SHA-256
  `bec39a0c68e4e74d7af3382164b22eec6f89001ae62e8cb8eb7c0cc9d85f5aa0`).
- `agent.png` and `bomb.png` are the package-local 30x30 presentation assets.
- Training uses the same package-local modules and remains available for
  reproducibility; official play loads only the inference path.
- The framework outside this directory is not modified by the agent package.

The policy/checkpoint combination was evaluated under the original framework
rules in solo and four-player games. The package layout and checkpoint checksum
are recorded in the manifest.
