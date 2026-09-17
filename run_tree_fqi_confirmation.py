"""Compare frozen tree FQI and masked Q-learning on unused coin-heaven boards."""

import hashlib
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from run_benchmark import SOURCE_DIR, confidence_interval, save_json

BOARD_SEEDS = list(range(20016, 20032))
STEP_BUDGETS = [100, 400]
POLICIES = {
    'tree_fqi': ('tree_fqi_agent', Path('experiments/tree_fqi_pilot')),
    'masked_q_learning': ('q_table_masked_agent',
                           Path('experiments/q_table_masked_pilot_confirm')),
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_policy(agent, checkpoint, seeds, max_steps, output):
    command = [
        sys.executable, str(SOURCE_DIR / 'run_benchmark.py'),
        '--agents', agent, '--model-path', str(checkpoint),
        '--seeds', *map(str, seeds), '--max-steps', str(max_steps),
        '--agent-seeds', '0', '--seats', '0', '1', '2', '3',
        '--batch-size', str(len(seeds) * 4),
        '--output', str(output),
    ]
    with open(output.with_suffix('.log'), 'w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(output / 'summary.json') as file:
        return json.load(file)['agents'][agent]


def checkpoint_paths(pilot):
    with open(pilot / 'config.json') as file:
        config = json.load(file)
    if config['seeds'] != [0, 1, 2] or config['rounds'] != 300:
        raise ValueError(f'Unexpected pilot protocol: {pilot}')
    paths = []
    for seed in config['seeds']:
        checkpoint = pilot / f'seed_{seed}/checkpoints/episode_0300.pkl'
        if not checkpoint.is_file():
            checkpoint = pilot / f'seed_{seed}/training.pkl'
        paths.append((seed, checkpoint))
    return paths


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true',
                        help='Reuse completed per-policy summaries in --output.')
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error('Choose a new --output directory.')
    if args.resume and not args.output.is_dir():
        parser.error('--resume requires an existing output directory.')
    args.output.mkdir(parents=True, exist_ok=True)

    source_paths = [Path(__file__), SOURCE_DIR / 'run_benchmark.py']
    for agent, _ in POLICIES.values():
        source_paths.append(SOURCE_DIR / 'agent_code' / agent / 'callbacks.py')
        features = SOURCE_DIR / 'agent_code' / agent / 'features.py'
        if features.is_file():
            source_paths.append(features)
    protocol = {
        'status': 'fresh-board frozen-policy comparison',
        'board_seeds': BOARD_SEEDS,
        'action_seeds': [0],
        'seats': [0, 1, 2, 3],
        'step_budgets': STEP_BUDGETS,
        'checkpoint_episode': 300,
        'source_hashes': {str(path.resolve()): file_hash(path) for path in source_paths},
        'note': ('All learner checkpoints are frozen. Board seeds 20016--20031 were not used '
                 'by the training, development, or earlier fresh-board evaluations.'),
    }
    config_path = args.output / 'config.json'
    if args.resume:
        with open(config_path) as file:
            saved_protocol = json.load(file)
        if (saved_protocol['board_seeds'] != BOARD_SEEDS or
                saved_protocol['step_budgets'] != STEP_BUDGETS):
            raise ValueError('Existing output has a different protocol.')
    else:
        save_json(config_path, protocol)

    rows = []
    for family, (agent, pilot) in POLICIES.items():
        for training_seed, checkpoint in checkpoint_paths(pilot):
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint_hash = file_hash(checkpoint)
            for max_steps in STEP_BUDGETS:
                directory = (args.output / f'steps_{max_steps:04d}' / family /
                             f'seed_{training_seed}')
                directory.parent.mkdir(parents=True, exist_ok=True)
                summary_path = directory / 'summary.json'
                if args.resume and summary_path.is_file():
                    with open(summary_path) as file:
                        summary = json.load(file)['agents'][agent]
                    print(f'Reusing {family} seed {training_seed}, {max_steps} steps', flush=True)
                else:
                    summary = run_policy(agent, checkpoint, BOARD_SEEDS, max_steps, directory)
                if file_hash(checkpoint) != checkpoint_hash:
                    raise RuntimeError(f'Checkpoint changed during evaluation: {checkpoint}')
                rows.append({
                    'family': family, 'agent': agent, 'training_seed': training_seed,
                    'max_steps': max_steps, 'checkpoint': str(checkpoint.resolve()),
                    'checkpoint_hash': checkpoint_hash, **summary,
                })
                print(f'{family} seed {training_seed}, {max_steps} steps: '
                      f'{summary["mean"]:.2f} coins', flush=True)

    report = {'status': protocol['status'], 'rows': rows, 'policies': {}}
    for max_steps in STEP_BUDGETS:
        for family in POLICIES:
            selected = [row for row in rows
                        if row['family'] == family and row['max_steps'] == max_steps]
            report['policies'][f'{max_steps}_steps/{family}'] = {
                'mean_coins': float(np.mean([row['mean'] for row in selected])),
                'run_means': [row['mean'] for row in selected],
                'mean_repeated_states': float(np.mean([
                    row['mean_repeated_states'] for row in selected])),
                'mean_invalid_actions': float(np.mean([
                    row['mean_invalid_actions'] for row in selected])),
                'completion_rate': float(np.mean([row['completion_rate'] for row in selected])),
            }
        tree = sorted([row for row in rows if row['family'] == 'tree_fqi' and
                       row['max_steps'] == max_steps], key=lambda row: row['training_seed'])
        table = sorted([row for row in rows if row['family'] == 'masked_q_learning' and
                        row['max_steps'] == max_steps], key=lambda row: row['training_seed'])
        differences = [a['mean'] - b['mean'] for a, b in zip(tree, table)]
        board_differences = np.mean([
            np.asarray(a['board_means']) - np.asarray(b['board_means'])
            for a, b in zip(tree, table)], axis=0)
        report[f'{max_steps}_steps/tree_minus_masked_q'] = {
            'per_checkpoint_coin_differences': differences,
            'mean_coin_difference': float(np.mean(differences)),
            'board_paired_ci95_conditional_on_checkpoints':
                confidence_interval(board_differences, 10000, 0),
        }
    report['limitation'] = (
        'Boards are fresh, but the three frozen training runs are reused. The conditional '
        'board-bootstrap intervals do not estimate uncertainty over newly trained policies.')
    save_json(args.output / 'summary.json', report)
    print(json.dumps(report['policies'], indent=2))


if __name__ == '__main__':
    main()
