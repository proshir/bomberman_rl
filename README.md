# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

## Project agents

The repository contains tabular Q-learning, linear SARSA(lambda), and tree-based
fitted-Q agents for the coin-navigation task. `combat_fqi_agent` is the first
bomb-aware extension. It combines fitted Q iteration with a deterministic escape
check, so the learner chooses among actions that have a known route through the
current bomb schedule.

The combat agent is implemented but not trained for submission. Evaluation needs
a fitted `trees.pkl`; the final file must be placed inside
`agent_code/combat_fqi_agent/` before packaging the agent.

## Local setup and checks

The current code uses Python, NumPy, scikit-learn, tqdm, and pygame. With the
local environment created for this project:

```bash
source .venv/bin/activate
python -m unittest -v test.py test_combat_safety.py
```

A deliberately small end-to-end crate smoke run is:

```bash
python run_combat_training.py \
  --scenario loot-crate \
  --seeds 0 \
  --rounds 2 \
  --max-steps 20 \
  --eval-every 2 \
  --eval-seeds 31000 \
  --eval-seats 0 \
  --output /tmp/combat_fqi_smoke
```

Smoke runs verify the software path only. They are not performance experiments
and should not be reported as evidence that an agent is better.
