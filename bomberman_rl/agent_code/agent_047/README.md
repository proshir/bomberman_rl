# agent_047 — packaged Agent 043

This directory records the metadata for the final-project runtime package in
the repository-level `agent_code/agent_047/` directory. That package keeps the
Agent 043 observation pipeline, DDQN policy, safety mask, temporal route
features, and bounded novelty intervention, with flattened `dep_*.py` runtime
dependencies.

## Running

From the original framework root:

```bash
python main.py play --my-agent agent_047
```

The package loads the bundled Agent 043 seed-1 mixed-300 continuation
checkpoint from `training.pkl`, resolved relative to the package itself.

## Final-project compliance

- The agent itself is machine-learning based: a PyTorch DDQN network is used
  for every action decision.
- `act` returns only framework actions and applies no multiprocessing or
  external worker process.
- All custom Python runtime dependencies are flattened into root-level
  `dep_*.py` modules.
- The runtime checkpoint is bundled and has SHA-256
  `bec39a0c68e4e74d7af3382164b22eec6f89001ae62e8cb8eb7c0cc9d85f5aa0`.
- The package includes the custom `agent.png` and `bomb.png` assets.
- Training-only modules are retained for reproducibility, but official play
  imports only the callback/runtime path.
- The framework outside this directory is not modified by the agent package.

The package was tested by importing the callbacks, constructing a probe game
state, loading the configured checkpoint, and requesting an action.
