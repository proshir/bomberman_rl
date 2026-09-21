# Agent 050 — packaged Agent 043

This directory is the final-project runtime package for Agent 043. It keeps
the Agent 043 observation pipeline, DDQN policy, safety mask, temporal route
features, and bounded novelty intervention, while vendoring the custom
experimental dependencies required by those modules.

## Running

From the original framework root:

```bash
python main.py play --my-agent Agent_050_final_agent043
```

The package loads `tournament_checkpoint.pt` automatically. The checkpoint is
the 10,000-step dataset-pretrained Agent 043 seed-0 episode-650 checkpoint.
The later 300-round online continuation was still running when this package
was assembled, so this is the best complete checkpoint currently available;
it can be replaced with a completed episode-950 checkpoint without changing
the runtime code.

## Final-project compliance

- The agent itself is machine-learning based: a PyTorch DDQN network is used
  for every action decision.
- `act` returns only framework actions and applies no multiprocessing or
  external worker process.
- All custom Python runtime dependencies are under `_vendor/`.
- The trained parameters are included as `tournament_checkpoint.pt`.
- Training-only modules are retained for reproducibility, but official play
  imports only the callback/runtime path.
- The framework outside this directory is not modified by the agent package.

The package was tested by importing the callbacks, constructing a probe game
state, loading the bundled checkpoint, and requesting an action.
