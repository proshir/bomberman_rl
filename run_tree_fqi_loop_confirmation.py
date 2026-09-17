"""Compare frozen tree FQI with its loop-escape wrapper on fresh boards."""

import hashlib
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from run_benchmark import SOURCE_DIR, confidence_interval, save_json

BOARD_SEEDS = list(range(20032, 20048))
STEP_BUDGETS = [100, 400]
PILOT = Path('experiments/tree_fqi_pilot')
VARIANTS = [('base', 'tree_fqi_agent'), ('loop', 'tree_fqi_loop_agent')]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_paths():
    with open(PILOT / 'config.json') as file:
        config = json.load(file)
    if config['seeds'] != [0, 1, 2] or config['rounds'] != 300:
        raise ValueError(f'Unexpected pilot protocol: {PILOT}')
    paths = []
    for seed in config['seeds']:
        checkpoint = PILOT / f'seed_{seed}/checkpoints/episode_0300.pkl'
        if not checkpoint.is_file():
            checkpoint = PILOT / f'seed_{seed}/training.pkl'
        paths.append((seed, checkpoint))
    return paths


def run_variant(agent, checkpoint, max_steps, output):
    command = [
        sys.executable, str(SOURCE_DIR / 'run_benchmark.py'),
        '--agents', agent, '--model-path', str(checkpoint),
        '--seeds', *map(str, BOARD_SEEDS), '--max-steps', str(max_steps),
        '--agent-seeds', '0', '--seats', '0', '1', '2', '3',
        '--batch-size', str(len(BOARD_SEEDS) * 4), '--output', str(output),
    ]
    with open(output.with_suffix('.log'), 'w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(output / 'summary.json') as file:
        return json.load(file)['agents'][agent]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new --output directory.')
    args.output.mkdir(parents=True)

    sources = [Path(__file__), SOURCE_DIR / 'run_benchmark.py']
    for _, agent in VARIANTS:
        sources.append(SOURCE_DIR / 'agent_code' / agent / 'callbacks.py')
        sources.append(SOURCE_DIR / 'agent_code' / agent / 'features.py')
    protocol = {
        'status': 'fresh-board frozen-policy loop comparison',
        'board_seeds': BOARD_SEEDS, 'action_seeds': [0], 'seats': [0, 1, 2, 3],
        'step_budgets': STEP_BUDGETS, 'checkpoint_episode': 300,
        'history_length': 8, 'repeat_threshold': 3,
        'source_hashes': {str(path.resolve()): file_hash(path) for path in sources},
        'note': ('The loop variant retains the learned trees and substitutes a random legal '
                 'movement only after a repeated-position detector fires.'),
    }
    save_json(args.output / 'config.json', protocol)

    rows = []
    for training_seed, checkpoint in checkpoint_paths():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hash = file_hash(checkpoint)
        for variant, agent in VARIANTS:
            for max_steps in STEP_BUDGETS:
                directory = args.output / f'steps_{max_steps:04d}' / variant / f'seed_{training_seed}'
                directory.parent.mkdir(parents=True, exist_ok=True)
                summary = run_variant(agent, checkpoint, max_steps, directory)
                if file_hash(checkpoint) != checkpoint_hash:
                    raise RuntimeError(f'Checkpoint changed during evaluation: {checkpoint}')
                rows.append({'variant': variant, 'agent': agent, 'training_seed': training_seed,
                             'max_steps': max_steps, 'checkpoint': str(checkpoint.resolve()),
                             'checkpoint_hash': checkpoint_hash, **summary})
                print(f'{variant} seed {training_seed}, {max_steps} steps: '
                      f'{summary["mean"]:.2f} coins', flush=True)

    report = {'status': protocol['status'], 'rows': rows, 'policies': {}}
    for max_steps in STEP_BUDGETS:
        base = sorted([row for row in rows if row['variant'] == 'base' and
                       row['max_steps'] == max_steps], key=lambda row: row['training_seed'])
        loop = sorted([row for row in rows if row['variant'] == 'loop' and
                       row['max_steps'] == max_steps], key=lambda row: row['training_seed'])
        for variant, selected in [('base', base), ('loop', loop)]:
            report['policies'][f'{max_steps}_steps/{variant}'] = {
                'mean_coins': float(np.mean([row['mean'] for row in selected])),
                'run_means': [row['mean'] for row in selected],
                'mean_repeated_states': float(np.mean([row['mean_repeated_states'] for row in selected])),
                'mean_invalid_actions': float(np.mean([row['mean_invalid_actions'] for row in selected])),
                'mean_loop_interventions': float(np.mean([row.get('mean_loop_interventions', 0.0)
                                                          for row in selected])),
            }
        differences = [changed['mean'] - original['mean'] for original, changed in zip(base, loop)]
        board_differences = np.mean([np.asarray(changed['board_means']) - np.asarray(original['board_means'])
                                     for original, changed in zip(base, loop)], axis=0)
        report[f'{max_steps}_steps/loop_minus_base'] = {
            'per_checkpoint_coin_differences': differences,
            'mean_coin_difference': float(np.mean(differences)),
            'board_paired_ci95_conditional_on_checkpoints': confidence_interval(board_differences, 10000, 0),
        }
    report['limitation'] = ('The boards are fresh, but intervals condition on three frozen checkpoints; '
                            'the loop wrapper is a hybrid policy rather than changed learning.')
    save_json(args.output / 'summary.json', report)
    print(json.dumps(report['policies'], indent=2))


if __name__ == '__main__':
    main()
