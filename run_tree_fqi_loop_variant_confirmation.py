"""Compare loop-focused history-agent variants on held-out boards."""

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
VARIANTS = {
    'baseline8': (Path('experiments/tree_fqi_history_400step_600round_pilot'), {}),
    'history16': (Path('experiments/tree_fqi_history_loop_variants_300round/history16'),
                  {'TREE_HISTORY_LENGTH': '16'}),
    'history32': (Path('experiments/tree_fqi_history_loop_variants_300round/history32'),
                  {'TREE_HISTORY_LENGTH': '32'}),
    'stagnation': (Path('experiments/tree_fqi_history_loop_variants_300round/stagnation'),
                   {'TREE_USE_STAGNATION': '1'}),
    'time': (Path('experiments/tree_fqi_history_loop_variants_300round/time'),
             {'TREE_USE_TIME': '1'}),
    'revisit_penalty': (
        Path('experiments/tree_fqi_history_loop_variants_300round/revisit_penalty'),
        {'TREE_REVISIT_PENALTY': '0.05'}),
}


def checkpoint_path(pilot, seed):
    candidates = [
        pilot / f'seed_{seed}/seed_{seed}/checkpoints/episode_0300.pkl',
        pilot / f'seed_{seed}/checkpoints/episode_0300.pkl',
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'Missing round-300 checkpoint for seed {seed}: {pilot}')


def evaluate_one(args, variant, seed, budget):
    pilot, variant_env = VARIANTS[variant]
    checkpoint = checkpoint_path(pilot, seed)
    output = args.output / f'steps_{budget:04d}' / variant / f'seed_{seed}'
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
        'variant': variant, 'training_seed': seed, 'max_steps': budget,
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

    jobs = [(variant, seed, budget) for budget in STEP_BUDGETS
            for variant in VARIANTS for seed in TRAINING_SEEDS]
    rows = []
    # Three concurrent games keeps the CPU load bounded while parallelising
    # independent frozen evaluations.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(evaluate_one, args, variant, seed, budget)
                   for variant, seed, budget in jobs]
        for future in futures:
            row = future.result()
            rows.append(row)
            print(f"{row['variant']} seed {row['training_seed']}, "
                  f"{row['max_steps']} steps: {row['mean']:.2f} coins", flush=True)

    report = {'status': 'held-out loop-variant comparison', 'rows': rows,
              'aggregates': {}}
    for variant in VARIANTS:
        for budget in STEP_BUDGETS:
            selected = [row for row in rows if row['variant'] == variant and
                        row['max_steps'] == budget]
            report['aggregates'][f'{variant}/steps_{budget:04d}'] = {
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
