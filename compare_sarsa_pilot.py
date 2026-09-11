"""Compare the fixed SARSA pilot with the preserved masked Q-table pilot."""

import csv
import json
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from run_benchmark import save_json


def read_json(path):
    with open(path) as file:
        return json.load(file)


def read_games(path):
    with open(path) as file:
        return [json.loads(line) for line in file]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--sarsa', type=Path, default=Path('experiments/linear_sarsa_pilot'))
    parser.add_argument('--baseline', type=Path,
                        default=Path('experiments/q_table_masked_pilot_confirm'))
    parser.add_argument('--diagnostics', type=Path,
                        default=Path('experiments/q_table_masked_sarsa_diagnostics'))
    parser.add_argument('--output', type=Path, default=Path('experiments/linear_sarsa_comparison'))
    args = parser.parse_args()
    config = read_json(args.sarsa / 'config.json')
    baseline_config = read_json(args.baseline / 'config.json')
    for key in ('seeds', 'rounds', 'max_steps', 'eval_every', 'eval_seeds', 'feature_mode'):
        assert config[key] == baseline_config[key], key
    for key, value in baseline_config['hyperparameters'].items():
        assert config['hyperparameters'][key] == value, key

    rows = []
    final = {}
    for name, directory in [('masked_q', args.baseline), ('linear_sarsa', args.sarsa)]:
        curve = read_json(directory / 'learning_curve.json')
        for point in curve:
            rows.append({'agent': name, 'seed': point['seed'], 'episode': point['episode'],
                         'interactions': point['interactions'], 'mean_coins': point['mean']})
        final[name] = []
        for seed in config['seeds']:
            episode = config['rounds']
            point = next(p for p in curve if p['seed'] == seed and p['episode'] == episode)
            run = directory / f'seed_{seed}'
            evaluation = run / 'evaluation' / f'episode_{episode:04d}'
            games = read_games(evaluation / 'games.jsonl')
            if name == 'masked_q':
                replay = read_games(args.diagnostics / f'seed_{seed}' / 'games.jsonl')
                original = {(g['seed'], g['agent_seed'], g['seat']): g['agents'][0]['coins']
                            for g in games}
                replayed = {(g['seed'], g['agent_seed'], g['seat']): g['agents'][0]['coins']
                            for g in replay}
                assert original == replayed, f'Baseline replay changed for seed {seed}'
                games = replay
            outcomes = [game['agents'][0] for game in games]
            rounds = read_games(run / 'rounds.jsonl')
            assert len(rounds) == config['rounds']
            for row in rounds:
                assert np.isclose(row['reward'], row['coins'] - 0.01 * row['steps'])
            assert np.isclose(np.mean([g['coins'] for g in outcomes]), point['mean'])
            if name == 'linear_sarsa':
                for checkpoint in (run / 'checkpoints').glob('*.pkl'):
                    with open(checkpoint, 'rb') as file:
                        weights = pickle.load(file)
                    assert weights.shape == (5, 26) and np.isfinite(weights).all()
            final[name].append({
                'seed': seed, 'mean_coins': point['mean'],
                'mean_repeated_states': float(np.mean([g['repeated_states'] for g in outcomes])),
                'mean_invalid_actions': float(np.mean([g['invalid_actions'] for g in outcomes])),
                'interactions': rounds[-1]['interactions'],
                'training_seconds': rounds[-1]['training_seconds'],
            })

    summary = {'status': 'exploratory development pilot', 'runs': final, 'agents': {}}
    for name, runs in final.items():
        summary['agents'][name] = {
            key: float(np.mean([run[key] for run in runs]))
            for key in ('mean_coins', 'mean_repeated_states', 'mean_invalid_actions')
        }
        summary['agents'][name]['coins_sd_across_runs'] = float(
            np.std([run['mean_coins'] for run in runs], ddof=1))
    summary['mean_coin_difference'] = (
        summary['agents']['linear_sarsa']['mean_coins'] - summary['agents']['masked_q']['mean_coins'])
    summary['limitation'] = ('Three training runs on reused development boards; descriptive results, '
                             'not a confirmed ranking or an isolated algorithm-component effect.')
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'summary.json', summary)
    with open(args.output / 'learning_curves.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
