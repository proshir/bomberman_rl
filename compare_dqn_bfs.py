"""Compare frozen DQN checkpoints with the supplied BFS-style coin collector."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from run_benchmark import SOURCE_DIR, save_json


def read_summary(path: Path) -> dict:
    with open(path / 'summary.json') as file:
        return json.load(file)['agents']


def evaluate(agent: str, seeds: list[int], steps: int, output: Path,
             checkpoint: Path | None = None) -> dict:
    command = [
        sys.executable, str(SOURCE_DIR / 'run_benchmark.py'),
        '--agents', agent,
        '--scenario', 'coin-heaven',
        '--max-steps', str(steps),
        '--seeds', *map(str, seeds),
        '--agent-seeds', '0',
        '--seats', '0', '1', '2', '3',
        '--batch-size', '8',
        '--output', str(output),
    ]
    if checkpoint is not None:
        command.extend(['--model-path', str(checkpoint.resolve())])
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_suffix('.log'), 'w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    return read_summary(output)[agent]


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(20000, 20016)))
    parser.add_argument('--steps', type=int, nargs='+', default=[100, 400])
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Choose a new output directory.')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Evaluation seeds must be unique.')
    args.output.mkdir(parents=True)
    with open(args.pilot / 'config.json') as file:
        pilot_config = json.load(file)
    training_seeds = pilot_config['seeds']
    rows = []
    for steps in args.steps:
        reference = evaluate(
            'coin_collector_agent', args.seeds, steps,
            args.output / f'steps_{steps:04d}' / 'bfs_reference')
        for training_seed in training_seeds:
            checkpoint = args.pilot / f'seed_{training_seed}' / 'checkpoints' / f'episode_{pilot_config["rounds"]:04d}.pt'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            dqn = evaluate(
                'dqn_coin_agent', args.seeds, steps,
                args.output / f'steps_{steps:04d}' / f'dqn_seed_{training_seed}',
                checkpoint)
            rows.append({
                'steps': steps,
                'training_seed': training_seed,
                'checkpoint': str(checkpoint.resolve()),
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'dqn': dqn,
                'bfs_reference': reference,
                'mean_difference_dqn_minus_bfs': dqn['mean'] - reference['mean'],
            })
            print(f'{steps} steps, DQN seed {training_seed}: '
                  f'{dqn["mean"]:.2f} coins vs BFS {reference["mean"]:.2f}', flush=True)
    report = {
        'status': 'fresh-board frozen-policy comparison',
        'pilot': str(args.pilot.resolve()),
        'board_seeds': args.seeds,
        'action_seeds': [0],
        'seats': [0, 1, 2, 3],
        'rows': rows,
        'aggregate': {},
    }
    for steps in args.steps:
        selected = [row for row in rows if row['steps'] == steps]
        report['aggregate'][str(steps)] = {
            'dqn_mean_across_training_seeds': float(np.mean([
                row['dqn']['mean'] for row in selected])),
            'bfs_mean': float(selected[0]['bfs_reference']['mean']),
            'mean_difference_dqn_minus_bfs': float(np.mean([
                row['mean_difference_dqn_minus_bfs'] for row in selected])),
            'dqn_run_means': [row['dqn']['mean'] for row in selected],
            'dqn_completion_rates': [row['dqn']['completion_rate'] for row in selected],
            'bfs_completion_rate': selected[0]['bfs_reference']['completion_rate'],
            'dqn_mean_completion_steps': [row['dqn']['mean_completion_steps'] for row in selected],
            'bfs_mean_completion_steps': selected[0]['bfs_reference']['mean_completion_steps'],
        }
    save_json(args.output / 'summary.json', report)
    print(json.dumps(report['aggregate'], indent=2))


if __name__ == '__main__':
    main()
