"""Evaluate a loop intervention on saved Q-table or linear SARSA policies."""

import hashlib
import importlib
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from run_benchmark import SOURCE_DIR, confidence_interval, save_json


def read_json(path):
    with open(path) as file:
        return json.load(file)


def read_games(path):
    with open(path) as file:
        return [json.loads(line) for line in file]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path,
                        default=Path('experiments/q_table_masked_pilot_confirm'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--loop-agent', default='q_table_loop_agent')
    args = parser.parse_args()
    original = read_json(args.baseline / 'config.json')
    loop = importlib.import_module(f'agent_code.{args.loop_agent}.callbacks')
    args.output.mkdir(parents=True, exist_ok=False)
    agents = [original['agent'], args.loop_agent]
    sources = [Path(__file__), SOURCE_DIR / 'run_benchmark.py']
    sources += [SOURCE_DIR / 'agent_code' / name / 'callbacks.py' for name in agents]
    for name in agents:
        features = SOURCE_DIR / 'agent_code' / name / 'features.py'
        if features.is_file():
            sources.append(features)
    protocol = {
        'baseline': str(args.baseline.resolve()), 'training_seeds': original['seeds'],
        'board_seeds': original['eval_seeds'], 'action_seeds': [0], 'seats': [0, 1, 2, 3],
        'max_steps': original['max_steps'], 'episode': original['rounds'],
        'agents': agents,
        'history_length': loop.HISTORY_LENGTH, 'repeat_threshold': loop.REPEAT_THRESHOLD,
        'source_hashes': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        'checkpoint_hashes': {},
    }
    for seed in original['seeds']:
        checkpoint = args.baseline / f'seed_{seed}/checkpoints/episode_{original["rounds"]:04d}.pkl'
        protocol['checkpoint_hashes'][str(seed)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    save_json(args.output / 'config.json', protocol)

    rows = []
    for seed in original['seeds']:
        checkpoint = args.baseline / f'seed_{seed}/checkpoints/episode_{original["rounds"]:04d}.pkl'
        for name in agents:
            output = args.output / f'seed_{seed}' / name
            output.parent.mkdir(exist_ok=True)
            command = [sys.executable, str(SOURCE_DIR / 'run_benchmark.py'),
                       '--agents', name, '--model-path', str(checkpoint),
                       '--seeds', *map(str, original['eval_seeds']),
                       '--max-steps', str(original['max_steps']), '--output', str(output)]
            with open(output.with_suffix('.log'), 'w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            summary = read_json(output / 'summary.json')['agents'][name]
            games = read_games(output / 'games.jsonl')
            if name == agents[0]:
                old_path = args.baseline / f'seed_{seed}/evaluation/episode_{original["rounds"]:04d}/games.jsonl'
                old_games = read_games(old_path)
                key = lambda game: (game['seed'], game['agent_seed'], game['seat'])
                assert {key(g): g['agents'][0]['coins'] for g in games} == {
                    key(g): g['agents'][0]['coins'] for g in old_games}
            rows.append({'training_seed': seed, 'agent': name, **summary})
            print(seed, name, round(summary['mean'], 3), flush=True)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == protocol['checkpoint_hashes'][str(seed)]

    baseline = [row for row in rows if row['agent'] == agents[0]]
    variant = [row for row in rows if row['agent'] == agents[1]]
    differences = [b['mean'] - a['mean'] for a, b in zip(baseline, variant)]
    board_differences = np.mean([
        np.asarray(b['board_means']) - a['board_means']
        for a, b in zip(baseline, variant)], axis=0)
    report = {
        'status': 'exploratory frozen-policy comparison', 'runs': rows,
        'per_checkpoint_coin_differences': differences,
        'mean_coin_difference': float(np.mean(differences)),
        'board_paired_ci95_conditional_on_checkpoints': confidence_interval(board_differences, 10000, 0),
        'limitation': 'Reused development boards; interval conditions on these three fixed policies, not new training runs.',
    }
    report['agents'] = {}
    for name in agents:
        selected = [row for row in rows if row['agent'] == name]
        report['agents'][name] = {
            key: float(np.mean([row.get(key, 0.0) for row in selected]))
            for key in ('mean', 'mean_repeated_states', 'mean_invalid_actions', 'mean_loop_interventions')}
    save_json(args.output / 'summary.json', report)
    print(json.dumps(report['agents'], indent=2))


if __name__ == '__main__':
    main()
