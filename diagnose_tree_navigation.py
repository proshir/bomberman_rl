"""Replay frozen policies to inspect navigation failures without changing actions."""

import hashlib
import importlib
import json
import logging
from collections import Counter, deque
from pathlib import Path

import numpy as np

from run_benchmark import play_game, save_json

SOURCE = Path('experiments/tree_fqi_loop_fresh_confirmation')
OUTPUT = Path('experiments/tree_navigation_diagnosis')
MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]


def distances(field, coins):
    result = {tuple(coin): 0 for coin in coins}
    queue = deque(result)
    while queue:
        x, y = queue.popleft()
        for dx, dy in MOVES[:4]:
            nxt = (x + dx, y + dy)
            if (0 <= nxt[0] < field.shape[0] and 0 <= nxt[1] < field.shape[1]
                    and field[nxt] == 0 and nxt not in result):
                result[nxt] = result[(x, y)] + 1
                queue.append(nxt)
    return result


def main():
    OUTPUT.mkdir(exist_ok=False)
    report = {'games': [], 'variants': {}, 'feature_conflicts': []}
    witnesses = {}
    for variant, name in [('base', 'tree_fqi_agent'), ('loop', 'tree_fqi_loop_agent')]:
        module = importlib.import_module(f'agent_code.{name}.callbacks')
        original = module.act
        totals = Counter()
        for seed in range(3):
            root = SOURCE / 'steps_0400' / variant / f'seed_{seed}'
            expected = [json.loads(line) for line in (root / 'games.jsonl').read_text().splitlines()]
            expected100 = [json.loads(line) for line in
                           (SOURCE / 'steps_0100' / variant / f'seed_{seed}' / 'games.jsonl').read_text().splitlines()]
            for index, saved in enumerate(expected):
                with open(root / 'games' / f'{index:04d}' / 'config.json') as file:
                    config = json.load(file)
                checkpoint = Path(config['model_path'])
                before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                directory = OUTPUT / variant / f'seed_{seed}' / f'{index:04d}'
                directory.mkdir(parents=True)
                config['log_dir'] = str(directory)
                config['agent_log_dir'] = str(directory / 'agents')
                trace = []
                cache = [None, None]

                def inspect(policy, state):
                    coins = tuple(sorted(state['coins']))
                    if coins != cache[0]:
                        cache[:] = [coins, distances(state['field'], coins)]
                    position = state['self'][3]
                    features = module.state_to_features(state)
                    values = module.predict_values(policy.trees, [features])[0]
                    legal = module.legal_action_indices(state)
                    distance = cache[1].get(position, 0)
                    toward = [i for i in legal if cache[1].get(
                        (position[0] + MOVES[i][0], position[1] + MOVES[i][1]), 999) < distance]
                    prior = getattr(policy, 'loop_interventions', 0)
                    action = original(policy, state)
                    record = dict(step=state['step'], position=position, remaining=len(coins),
                                  features=features, values=values.tolist(), legal=legal,
                                  action=action, distance=distance, toward=toward,
                                  intervention=getattr(policy, 'loop_interventions', 0) > prior)
                    trace.append(record)
                    if coins:
                        totals['actions'] += 1
                        totals['waits'] += action == 'WAIT'
                        totals['toward_nearest_coin'] += module.ACTIONS.index(action) in toward
                        totals['interventions'] += record['intervention']
                        # Incompatible shortest-distance choices for exactly identical inputs.
                        examples = witnesses.setdefault(features, {})
                        key = tuple(toward)
                        if key not in examples:
                            examples[key] = dict(variant=variant, training_seed=seed,
                                board=saved['seed'], seat=saved['seat'], **record)
                    return action

                module.act = inspect
                result = play_game(config)
                module.act = original
                assert result['agents'][0]['coins'] == saved['agents'][0]['coins']
                assert result['agents'][0]['repeated_states'] == saved['agents'][0]['repeated_states']
                assert 50 - trace[100]['remaining'] == expected100[index]['agents'][0]['coins']
                assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
                last_coin = max([b['step'] - 1 for a, b in zip(trace, trace[1:])
                                 if b['remaining'] < a['remaining']] or [0])
                # Exact state recurrence with no coin progress is a real movement cycle.
                recent = {}
                cycles = Counter()
                for row in trace:
                    key = (row['position'], row['remaining'])
                    if key in recent:
                        cycles[row['step'] - recent[key]] += 1
                    recent[key] = row['step']
                game = dict(variant=variant, training_seed=seed, board=saved['seed'],
                            seat=saved['seat'], coins=result['agents'][0]['coins'],
                            last_coin_step=last_coin, recurrence_gaps=dict(cycles),
                            waits=sum(row['action'] == 'WAIT' for row in trace),
                            interventions=sum(row['intervention'] for row in trace))
                report['games'].append(game)
                save_json(directory / 'trace.json', trace)
                # Each replay owns its log files; do not accumulate handlers across games.
                for logger in logging.Logger.manager.loggerDict.values():
                    if isinstance(logger, logging.Logger):
                        for handler in logger.handlers[:]:
                            if isinstance(handler, logging.FileHandler):
                                logger.removeHandler(handler)
                                handler.close()
            print(variant, seed, '64 replays matched at 100 and 400 steps', flush=True)
        report['variants'][variant] = dict(totals)
    for features, examples in witnesses.items():
        keys = list(examples)
        for i, first in enumerate(keys):
            conflict = next((second for second in keys[i + 1:]
                             if first and second and not set(first).intersection(second)), None)
            if conflict is not None:
                report['feature_conflicts'].append(dict(features=features,
                    first=examples[first], second=examples[conflict]))
                break
    report['observed_feature_tuples'] = len(witnesses)
    save_json(OUTPUT / 'summary.json', report)
    print(report['variants'])
    print('Feature tuples with incompatible nearest-coin directions:',
          len(report['feature_conflicts']), '/', len(witnesses))


if __name__ == '__main__':
    main()
