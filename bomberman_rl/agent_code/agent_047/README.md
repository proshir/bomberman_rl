# agent_047 — standalone Agent 043 package

This directory records the metadata for the final-project runtime package in
the repository-level `agent_code/agent_047/` directory. That package keeps the
Agent 043 observation pipeline, DDQN policy, safety mask, temporal route
features, and bounded novelty intervention. Its runtime and training modules
are self-contained in the package, with no imports from other agent folders.

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
- Callbacks, checkpoint, model, replay, safety, symmetry, training, and feature
  helpers are package-local modules. Duplicate ancestor trainers and safety
  aliases have been merged into the local implementation.
- The runtime checkpoint is bundled and has SHA-256
  `bec39a0c68e4e74d7af3382164b22eec6f89001ae62e8cb8eb7c0cc9d85f5aa0`.
- The package includes the custom `agent.png` and `bomb.png` assets.
- Training uses the same local modules and remains available for
  reproducibility; official play loads only the inference path.
- The framework outside this directory is not modified by the agent package.

The bundled checkpoint was evaluated with the protocol recorded in the final
package evaluation note.
