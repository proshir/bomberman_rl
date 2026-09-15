"""Evaluate frozen coin-navigation policies on boards never used for development."""

import hashlib
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from run_benchmark import SOURCE_DIR, confidence_interval, save_json


DEFAULT_SEEDS = list(range(20000, 20016))
STEP_BUDGETS = [100, 400]


def read_json(path):
    with open(path) as file:
        return json.load(file)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_benchmark(agent, checkpoint, seeds, max_steps, output):
    command = [
        sys.executable, str(SOURCE_DIR / 'run_benchmark.py'),
        '--agents', agent,
        '--seeds', *map(str, seeds),
        '--max-steps', str(max_steps),
        '--output', str(output),
    ]
    if checkpoint is not None:
        command.extend(['--model-path', str(checkpoint)])
    with open(output.with_suffix('.log'), 'w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    return read_json(output / 'summary.json')['agents'][agent]


def policy_rows(family, base_agent, loop_agent, pilot, seeds, output):
    config = read_json(pilot / 'config.json')
    if config['seeds'] != [0, 1, 2] or config['rounds'] != 300:
        raise ValueError(f'Unexpected checkpoint protocol: {pilot}')
    rows = []
    for training_seed in config['seeds']:
        checkpoint = pilot / f'seed_{training_seed}/checkpoints/episode_0300.pkl'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hash = hash_file(checkpoint)
        for agent, variant in ((base_agent, 'base'), (loop_agent, 'loop')):
            for max_steps in STEP_BUDGETS:
                directory = output / f'steps_{max_steps:04d}' / family / variant / f'seed_{training_seed}'
                directory.parent.mkdir(parents=True, exist_ok=True)
                summary = run_benchmark(agent, checkpoint, seeds, max_steps, directory)
                if hash_file(checkpoint) != checkpoint_hash:
                    raise RuntimeError(f'Checkpoint changed during evaluation: {checkpoint}')
                rows.append({
                    'family': family,
                    'variant': variant,
                    'training_seed': training_seed,
                    'max_steps': max_steps,
                    'checkpoint': str(checkpoint.resolve()),
                    'checkpoint_hash': checkpoint_hash,
                    **summary,
                })
                print(f'{family} {variant} seed {training_seed}, {max_steps} steps: '
                      f'{summary["mean"]:.2f} coins', flush=True)
    return rows


def collector_rows(seeds, output):
    rows = []
    for max_steps in STEP_BUDGETS:
        directory = output / f'steps_{max_steps:04d}' / 'reference' / 'coin_collector'
        directory.parent.mkdir(parents=True, exist_ok=True)
        summary = run_benchmark('coin_collector_agent', None, seeds, max_steps, directory)
        rows.append({
            'family': 'reference',
            'variant': 'coin_collector',
            'training_seed': None,
            'max_steps': max_steps,
            **summary,
        })
        print(f'coin collector, {max_steps} steps: {summary["mean"]:.2f} coins', flush=True)
    return rows


def summarize(rows):
    report = {'policies': {}, 'within_family_loop_differences': {}}
    for max_steps in STEP_BUDGETS:
        step_rows = [row for row in rows if row['max_steps'] == max_steps]
        for family, variant in sorted({(row['family'], row['variant']) for row in step_rows}):
            selected = [row for row in step_rows
                        if row['family'] == family and row['variant'] == variant]
            key = f'{max_steps}_steps/{family}/{variant}'
            report['policies'][key] = {
                'mean_coins': float(np.mean([row['mean'] for row in selected])),
                'mean_repeated_states': float(np.mean([
                    row.get('mean_repeated_states', 0.0) for row in selected])),
                'mean_invalid_actions': float(np.mean([
                    row.get('mean_invalid_actions', 0.0) for row in selected])),
                'mean_loop_interventions': float(np.mean([
                    row.get('mean_loop_interventions', 0.0) for row in selected])),
                'completion_rate': float(np.mean([row['completion_rate'] for row in selected])),
                'mean_completion_steps': (
                    float(np.mean([row['mean_completion_steps'] for row in selected
                                   if row['mean_completion_steps'] is not None]))
                    if any(row['mean_completion_steps'] is not None for row in selected)
                    else None),
                'run_means': [row['mean'] for row in selected],
            }
        for family in ('q_learning', 'linear_sarsa'):
            base = sorted([row for row in step_rows
                           if row['family'] == family and row['variant'] == 'base'],
                          key=lambda row: row['training_seed'])
            loop = sorted([row for row in step_rows
                           if row['family'] == family and row['variant'] == 'loop'],
                          key=lambda row: row['training_seed'])
            differences = [loop_row['mean'] - base_row['mean']
                           for base_row, loop_row in zip(base, loop)]
            board_differences = np.mean([
                np.asarray(loop_row['board_means']) - np.asarray(base_row['board_means'])
                for base_row, loop_row in zip(base, loop)], axis=0)
            report['within_family_loop_differences'][f'{max_steps}_steps/{family}'] = {
                'per_checkpoint': differences,
                'mean': float(np.mean(differences)),
                'board_paired_ci95_conditional_on_checkpoints':
                    confidence_interval(board_differences, 10000, 0),
            }
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new --output directory.')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Board seeds must be unique.')
    used_seeds = set(range(1000, 1900)) | set(range(10000, 10008))
    if used_seeds.intersection(args.seeds):
        parser.error('Fresh boards overlap prior training or development seeds.')

    args.output.mkdir(parents=True)
    pilots = {
        'q_learning': (Path('experiments/q_table_masked_pilot_confirm'),
                       'q_table_masked_agent', 'q_table_loop_agent'),
        'linear_sarsa': (Path('experiments/linear_sarsa_pilot'),
                         'linear_sarsa_agent', 'linear_sarsa_loop_agent'),
    }
    source_paths = [Path(__file__), SOURCE_DIR / 'run_benchmark.py']
    for _, (_, base, loop) in pilots.items():
        source_paths += [SOURCE_DIR / 'agent_code' / base / 'callbacks.py',
                         SOURCE_DIR / 'agent_code' / loop / 'callbacks.py']
        for agent in (base, loop):
            features = SOURCE_DIR / 'agent_code' / agent / 'features.py'
            if features.is_file():
                source_paths.append(features)
    source_paths.append(SOURCE_DIR / 'agent_code' / 'coin_collector_agent' / 'callbacks.py')
    protocol = {
        'status': 'fresh-board frozen-policy confirmation',
        'board_seeds': args.seeds,
        'action_seeds': [0],
        'seats': [0, 1, 2, 3],
        'step_budgets': STEP_BUDGETS,
        'checkpoint_episode': 300,
        'source_hashes': {str(path.resolve()): hash_file(path) for path in source_paths},
        'note': ('All learner checkpoints are frozen. The supplied collector is a non-learning '
                 'reference. Loop variants add evaluation-time history and random escape actions.'),
    }
    save_json(args.output / 'config.json', protocol)
    rows = []
    for family, (pilot, base, loop) in pilots.items():
        rows.extend(policy_rows(family, base, loop, pilot, args.seeds, args.output))
    rows.extend(collector_rows(args.seeds, args.output))
    report = summarize(rows)
    report['status'] = protocol['status']
    report['rows'] = rows
    report['limitation'] = (
        'Fresh boards are independent of the recorded training and development boards, but the '
        'same three frozen training runs are reused. Conditional bootstrap intervals are not '
        'uncertainty estimates over newly trained policies.')
    save_json(args.output / 'summary.json', report)
    print(json.dumps(report['policies'], indent=2))


if __name__ == '__main__':
    main()
