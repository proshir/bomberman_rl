"""Evaluate stagnation-feature checkpoints selected on development boards."""

import hashlib
import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from run_benchmark import SOURCE_DIR, save_json

BOARD_SEEDS = list(range(22016, 22032))
TRAINING_SEEDS = [0, 1, 2]
STEP_BUDGETS = [100, 400]
CANDIDATES = {
    'stagnation_h8_r300': (
        Path('experiments/tree_fqi_history_stagnation_600round/stagnation_h8'),
        300, {'TREE_USE_STAGNATION': '1'}),
    'stagnation_h16_r400': (
        Path('experiments/tree_fqi_history_stagnation_600round/stagnation_h16'),
        400, {'TREE_USE_STAGNATION': '1', 'TREE_HISTORY_LENGTH': '16'}),
    'stagnation_h16_r500': (
        Path('experiments/tree_fqi_history_stagnation_600round/stagnation_h16'),
        500, {'TREE_USE_STAGNATION': '1', 'TREE_HISTORY_LENGTH': '16'}),
}


def checkpoint_path(pilot, seed, episode):
    candidates = [
        pilot / f'seed_{seed}/seed_{seed}/checkpoints/episode_{episode:04d}.pkl',
        pilot / f'seed_{seed}/checkpoints/episode_{episode:04d}.pkl',
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'Missing checkpoint: {pilot}, seed {seed}, round {episode}')


def evaluate_one(args, candidate, seed, budget):
    pilot, episode, variant_env = CANDIDATES[candidate]
    checkpoint = checkpoint_path(pilot, seed, episode)
    output = args.output / f'steps_{budget:04d}' / candidate / f'seed_{seed}'
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(SOURCE_DIR / 'run_benchmark.py'),
        '--agents', 'tree_fqi_history_agent', '--model-path', str(checkpoint),
        '--seeds', *map(str, BOARD_SEEDS), '--max-steps', str(budget),
        '--agent-seeds', '0', '--seats', '0', '1', '2', '3',
        '--batch-size', str(4 * len(BOARD_SEEDS)), '--output', str(output),
    ]
    environment = os.environ.copy()
    environment.update({
        'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'MPLBACKEND': 'Agg', **variant_env,
    })
    with open(output.with_suffix('.log'), 'w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env=environment)
    with open(output / 'summary.json') as file:
        summary = json.load(file)['agents']['tree_fqi_history_agent']
    return {
        'candidate': candidate, 'training_seed': seed, 'max_steps': budget,
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_hash': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        **summary,
    }


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Choose a new output directory.')
    args.output.mkdir(parents=True)

    jobs = [(candidate, seed, budget) for budget in STEP_BUDGETS
            for candidate in CANDIDATES for seed in TRAINING_SEEDS]
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(evaluate_one, args, candidate, seed, budget)
                   for candidate, seed, budget in jobs]
        for future in futures:
            row = future.result()
            rows.append(row)
            print(f"{row['candidate']} seed {row['training_seed']}, "
                  f"{row['max_steps']} steps: {row['mean']:.2f} coins", flush=True)

    report = {'status': 'held-out stagnation checkpoint comparison', 'rows': rows,
              'aggregates': {}}
    for candidate in CANDIDATES:
        for budget in STEP_BUDGETS:
            selected = [row for row in rows if row['candidate'] == candidate and
                        row['max_steps'] == budget]
            report['aggregates'][f'{candidate}/steps_{budget:04d}'] = {
                'mean_coins': float(np.mean([row['mean'] for row in selected])),
                'run_means': [row['mean'] for row in selected],
                'completion_rate': float(np.mean([row['completion_rate'] for row in selected])),
                'mean_repeated_states': float(np.mean([row['mean_repeated_states'] for row in selected])),
                'mean_invalid_actions': float(np.mean([row['mean_invalid_actions'] for row in selected])),
                'mean_self_deaths': float(np.mean([row['mean_suicides'] for row in selected])),
            }
    save_json(args.output / 'summary.json', report)
    print(json.dumps(report['aggregates'], indent=2))


if __name__ == '__main__':
    main()
