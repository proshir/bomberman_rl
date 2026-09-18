"""Compare frozen tree-FQI policies with and without learned movement history."""

import hashlib
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from run_benchmark import SOURCE_DIR, confidence_interval, save_json

BOARD_SEEDS = list(range(22016, 22032))
STEP_BUDGETS = [100, 400]
PILOTS = {
    'base': ('tree_fqi_agent', Path('experiments/tree_fqi_pilot')),
    'history': ('tree_fqi_history_agent', Path('experiments/tree_fqi_history_pilot')),
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoints(pilot):
    with open(pilot / 'config.json') as file:
        config = json.load(file)
    if config['seeds'] != [0, 1, 2] or config['rounds'] != 300:
        raise ValueError(f'Unexpected pilot protocol: {pilot}')
    result = []
    for seed in config['seeds']:
        path = pilot / f'seed_{seed}/checkpoints/episode_0300.pkl'
        if not path.is_file():
            path = pilot / f'seed_{seed}/training.pkl'
        result.append((seed, path))
    return result


def run_policy(agent, checkpoint, max_steps, output):
    command = [
        sys.executable, str(SOURCE_DIR / 'run_benchmark.py'),
        '--agents', agent, '--model-path', str(checkpoint),
        '--seeds', *map(str, BOARD_SEEDS), '--max-steps', str(max_steps),
        '--agent-seeds', '0', '--seats', '0', '1', '2', '3',
        '--batch-size', str(4 * len(BOARD_SEEDS)), '--output', str(output),
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
    for agent, _ in PILOTS.values():
        sources.append(SOURCE_DIR / 'agent_code' / agent / 'callbacks.py')
        sources.append(SOURCE_DIR / 'agent_code' / agent / 'features.py')
    protocol = {
        'status': 'fresh-board frozen-policy comparison',
        'board_seeds': BOARD_SEEDS, 'agent_seeds': [0], 'seats': [0, 1, 2, 3],
        'step_budgets': STEP_BUDGETS, 'checkpoint_episode': 300,
        'source_hashes': {str(path.resolve()): file_hash(path) for path in sources},
        'note': ('Final checkpoints from three matched 300-round pilots; boards 22016--22031 '
                 'were not used by these pilots or earlier evaluations.'),
    }
    save_json(args.output / 'config.json', protocol)

    rows = []
    for variant, (agent, pilot) in PILOTS.items():
        for seed, checkpoint in checkpoints(pilot):
            checkpoint_hash = file_hash(checkpoint)
            for budget in STEP_BUDGETS:
                directory = args.output / f'steps_{budget:04d}' / variant / f'seed_{seed}'
                directory.parent.mkdir(parents=True, exist_ok=True)
                summary = run_policy(agent, checkpoint, budget, directory)
                if file_hash(checkpoint) != checkpoint_hash:
                    raise RuntimeError(f'Checkpoint changed during evaluation: {checkpoint}')
                rows.append({'variant': variant, 'agent': agent, 'training_seed': seed,
                             'max_steps': budget, 'checkpoint': str(checkpoint.resolve()),
                             'checkpoint_hash': checkpoint_hash, **summary})
                print(f'{variant} seed {seed}, {budget} steps: {summary["mean"]:.2f} coins',
                      flush=True)

    report = {'status': protocol['status'], 'rows': rows, 'policies': {}}
    for budget in STEP_BUDGETS:
        base = sorted([row for row in rows if row['variant'] == 'base' and
                       row['max_steps'] == budget], key=lambda row: row['training_seed'])
        history = sorted([row for row in rows if row['variant'] == 'history' and
                          row['max_steps'] == budget], key=lambda row: row['training_seed'])
        for name, selected in [('base', base), ('history', history)]:
            report['policies'][f'{budget}_steps/{name}'] = {
                'mean_coins': float(np.mean([row['mean'] for row in selected])),
                'run_means': [row['mean'] for row in selected],
                'mean_repeated_states': float(np.mean([row['mean_repeated_states'] for row in selected])),
                'mean_invalid_actions': float(np.mean([row['mean_invalid_actions'] for row in selected])),
                'completion_rate': float(np.mean([row['completion_rate'] for row in selected])),
            }
        differences = [new['mean'] - old['mean'] for old, new in zip(base, history)]
        board_differences = np.mean([
            np.asarray(new['board_means']) - np.asarray(old['board_means'])
            for old, new in zip(base, history)], axis=0)
        report[f'{budget}_steps/history_minus_base'] = {
            'per_checkpoint_coin_differences': differences,
            'mean_coin_difference': float(np.mean(differences)),
            'board_paired_ci95_conditional_on_checkpoints': confidence_interval(
                board_differences, 10000, 0),
        }
    report['limitation'] = ('Fresh boards, but uncertainty is conditional on three frozen training runs; '
                            'the results do not measure uncertainty over new training seeds.')
    save_json(args.output / 'summary.json', report)
    print(json.dumps(report['policies'], indent=2))


if __name__ == '__main__':
    main()
