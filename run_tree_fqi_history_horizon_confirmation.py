"""Compare history FQI checkpoints trained with 100- and 400-step episodes."""

import hashlib
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from run_benchmark import SOURCE_DIR, confidence_interval, save_json

BOARD_SEEDS = list(range(22048, 22064))
STEP_BUDGETS = [100, 400]
PILOTS = {
    'train_100': Path('experiments/tree_fqi_history_pilot'),
    'train_400': Path('experiments/tree_fqi_history_400step_pilot'),
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoints(pilot):
    with open(pilot / 'config.json') as file:
        config = json.load(file)
    if config['agent'] != 'tree_fqi_history_agent' or config['seeds'] != [0, 1, 2]:
        raise ValueError(f'Unexpected pilot protocol: {pilot}')
    if config['rounds'] != 300:
        raise ValueError(f'Expected 300 training rounds: {pilot}')
    return [(seed, pilot / f'seed_{seed}/checkpoints/episode_0300.pkl')
            for seed in config['seeds']]


def run_policy(checkpoint, max_steps, output):
    command = [
        sys.executable, str(SOURCE_DIR / 'run_benchmark.py'),
        '--agents', 'tree_fqi_history_agent', '--model-path', str(checkpoint),
        '--seeds', *map(str, BOARD_SEEDS), '--max-steps', str(max_steps),
        '--agent-seeds', '0', '--seats', '0', '1', '2', '3',
        '--batch-size', str(4 * len(BOARD_SEEDS)), '--output', str(output),
    ]
    with open(output.with_suffix('.log'), 'w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(output / 'summary.json') as file:
        return json.load(file)['agents']['tree_fqi_history_agent']


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new --output directory.')
    args.output.mkdir(parents=True)

    sources = [Path(__file__), SOURCE_DIR / 'run_benchmark.py',
               SOURCE_DIR / 'agent_code/tree_fqi_history_agent/callbacks.py',
               SOURCE_DIR / 'agent_code/tree_fqi_history_agent/features.py']
    protocol = {
        'status': 'fresh-board frozen-policy comparison of training horizons',
        'board_seeds': BOARD_SEEDS,
        'agent_seeds': [0],
        'seats': [0, 1, 2, 3],
        'step_budgets': STEP_BUDGETS,
        'checkpoint_episode': 300,
        'source_hashes': {str(path.resolve()): file_hash(path) for path in sources},
        'note': ('Final checkpoints from matched three-seed runs; boards 22048--22063 '
                 'were not used for training or development evaluation.'),
    }
    save_json(args.output / 'config.json', protocol)

    rows = []
    for horizon, pilot in PILOTS.items():
        for seed, checkpoint in checkpoints(pilot):
            checkpoint_hash = file_hash(checkpoint)
            for budget in STEP_BUDGETS:
                directory = args.output / f'steps_{budget:04d}' / horizon / f'seed_{seed}'
                directory.parent.mkdir(parents=True, exist_ok=True)
                summary = run_policy(checkpoint, budget, directory)
                if file_hash(checkpoint) != checkpoint_hash:
                    raise RuntimeError(f'Checkpoint changed during evaluation: {checkpoint}')
                rows.append({'training_horizon': horizon, 'training_seed': seed,
                             'max_steps': budget, 'checkpoint': str(checkpoint.resolve()),
                             'checkpoint_hash': checkpoint_hash, **summary})
                print(f'{horizon} seed {seed}, {budget} steps: {summary["mean"]:.2f} coins',
                      flush=True)

    report = {'status': protocol['status'], 'rows': rows, 'policies': {}}
    for budget in STEP_BUDGETS:
        old = sorted([row for row in rows if row['training_horizon'] == 'train_100' and
                      row['max_steps'] == budget], key=lambda row: row['training_seed'])
        new = sorted([row for row in rows if row['training_horizon'] == 'train_400' and
                      row['max_steps'] == budget], key=lambda row: row['training_seed'])
        for name, selected in [('train_100', old), ('train_400', new)]:
            report['policies'][f'{budget}_steps/{name}'] = {
                'mean_coins': float(np.mean([row['mean'] for row in selected])),
                'run_means': [row['mean'] for row in selected],
                'mean_repeated_states': float(np.mean([row['mean_repeated_states'] for row in selected])),
                'mean_invalid_actions': float(np.mean([row['mean_invalid_actions'] for row in selected])),
                'completion_rate': float(np.mean([row['completion_rate'] for row in selected])),
            }
        differences = [later['mean'] - earlier['mean']
                       for earlier, later in zip(old, new)]
        board_differences = np.mean([
            np.asarray(later['board_means']) - np.asarray(earlier['board_means'])
            for earlier, later in zip(old, new)], axis=0)
        report[f'{budget}_steps/train_400_minus_train_100'] = {
            'per_checkpoint_coin_differences': differences,
            'mean_coin_difference': float(np.mean(differences)),
            'board_paired_ci95_conditional_on_checkpoints': confidence_interval(
                board_differences, 10000, 0),
        }
    report['limitation'] = ('Uncertainty is conditional on three frozen runs; this comparison '
                            'does not measure uncertainty over additional training seeds.')
    save_json(args.output / 'summary.json', report)
    print(json.dumps(report['policies'], indent=2))


if __name__ == '__main__':
    main()
