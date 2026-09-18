"""Reproducible coin-heaven DQN pilot runner.

Sahand was here.

The runner keeps training boards separate from frozen evaluation boards, rotates
the learner through all four starting corners, records the training curve, and
saves CPU-loadable checkpoints.  The environment is still stepped on the CPU;
only batched DQN updates use the selected device.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import random
import subprocess
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

import settings as s
from run_benchmark import BenchmarkWorld, SOURCE_DIR, save_json


AGENT = 'dqn_coin_agent'


def save_checkpoint(policy, path: Path, metadata: dict | None = None) -> None:
    """Save model weights on CPU so the file is portable to tournament hosts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': 1,
        'model_state': {name: value.detach().cpu() for name, value in policy.model.state_dict().items()},
        'feature_size': policy.model.network[0].in_features,
        'action_count': policy.model.network[-1].out_features,
    }
    if metadata:
        payload['metadata'] = metadata
    torch.save(payload, path)


def evaluate(config: dict, policy, episode: int, interactions: int) -> dict:
    """Evaluate a frozen checkpoint on held-out boards in fresh processes."""
    directory = Path(config['output'])
    checkpoint = directory / 'checkpoints' / f'episode_{episode:04d}.pt'
    save_checkpoint(policy, checkpoint, {
        'training_seed': config['seed'],
        'episode': episode,
        'interactions': interactions,
        'device_used_for_training': str(policy.device),
    })
    output = directory / 'evaluation' / f'episode_{episode:04d}'
    command = [
        sys.executable, str(SOURCE_DIR / 'run_benchmark.py'),
        '--agents', AGENT,
        '--scenario', 'coin-heaven',
        '--model-path', str(checkpoint.resolve()),
        '--max-steps', str(config['eval_steps']),
        '--seeds', *map(str, config['eval_seeds']),
        '--agent-seeds', '0',
        '--seats', '0', '1', '2', '3',
        '--batch-size', '8',
        '--output', str(output),
    ]
    output.parent.mkdir(exist_ok=True)
    with open(output.with_suffix('.log'), 'w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(output / 'summary.json') as file:
        summary = json.load(file)['agents'][AGENT]
    return {
        'episode': episode,
        'interactions': interactions,
        'checkpoint': str(checkpoint),
        **summary,
    }


def train_one(config: dict) -> None:
    """Train one independent seed and save frozen checkpoints."""
    os.environ['DQN_DEVICE'] = config['device']
    callbacks = importlib.import_module(f'agent_code.{AGENT}.callbacks')
    callbacks.MODEL_PATH = Path(config['output']) / 'training.pt'
    s.MAX_STEPS = config['train_steps']
    s.LOG_GAME = s.LOG_AGENT_WRAPPER = s.LOG_AGENT_CODE = logging.WARNING
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    directory = Path(config['output'])
    (directory / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (directory / 'logs').mkdir(exist_ok=True)
    args = SimpleNamespace(
        seed=config['seed'], agent_seed=config['seed'], seat=0,
        scenario='coin-heaven', no_gui=True, save_replay=False,
        save_stats=False, match_name=None, continue_without_training=True,
        silence_errors=False, log_dir=str(directory / 'logs'),
        agent_log_dir=str(directory / 'logs' / 'agents'),
    )
    world = BenchmarkWorld(args, [(AGENT, True)])
    policy = world.agents[0].backend.runner.fake_self
    curve = [evaluate(config, policy, 0, 0)]
    save_json(directory / 'learning_curve.json', curve)
    interactions = 0
    training_seconds = 0.0
    with open(directory / 'rounds.jsonl', 'w') as records:
        for episode in tqdm(range(1, config['rounds'] + 1), desc=f"DQN seed {config['seed']}"):
            board_seed = config['board_start'] + episode - 1
            world.rng = np.random.default_rng(board_seed)
            args.seat = (episode - 1) % 4
            epsilon = policy.epsilon
            started = perf_counter()
            world.new_round()
            while world.running:
                world.do_step()
            elapsed = perf_counter() - started
            training_seconds += elapsed
            interactions += world.step
            agent = world.agents[0]
            record = {
                'episode': episode,
                'board_seed': board_seed,
                'agent_seed': config['seed'],
                'seat': args.seat,
                'steps': world.step,
                'interactions': interactions,
                'coins': agent.statistics['coins'],
                'score': agent.score,
                'reward': policy.last_round_reward,
                'epsilon': epsilon,
                'last_loss': policy.last_loss,
                'replay_size': len(policy.replay),
                'updates': policy.update_count,
                'training_seconds': training_seconds,
            }
            records.write(json.dumps(record) + '\n')
            records.flush()
            if episode % config['eval_every'] == 0 or episode == config['rounds']:
                curve.append(evaluate(config, policy, episode, interactions))
                save_json(directory / 'learning_curve.json', curve)
    world.end()


def source_hashes() -> dict[str, str]:
    paths = [
        SOURCE_DIR / 'run_dqn_coin_training.py',
        SOURCE_DIR / 'run_benchmark.py',
        SOURCE_DIR / 'environment.py',
        SOURCE_DIR / 'agents.py',
        SOURCE_DIR / 'settings.py',
        SOURCE_DIR / 'events.py',
    ]
    paths.extend(SOURCE_DIR / 'agent_code' / AGENT / name
                 for name in ('callbacks.py', 'features.py', 'model.py', 'train.py'))
    return {str(path.relative_to(SOURCE_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--rounds', type=int, default=300)
    parser.add_argument('--train-steps', type=int, default=400)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=list(range(10000, 10008)))
    parser.add_argument('--board-start', type=int, default=1000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--worker', type=Path, help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return args
    if args.output is None or args.output.exists():
        parser.error('Choose a new --output directory.')
    if min(args.rounds, args.train_steps, args.eval_steps, args.eval_every) < 1:
        parser.error('Round counts and step limits must be positive.')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error('Seeds must be unique within each list.')
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds + args.eval_seeds):
        parser.error('Seeds must be between 0 and 2**32 - 1.')
    training_boards = set(range(args.board_start, args.board_start + args.rounds * len(args.seeds)))
    if training_boards.intersection(args.eval_seeds):
        parser.error('Evaluation seeds overlap the training board seeds.')
    return args


def run_experiment(args) -> None:
    requested = args.device
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false.')
    device = 'cuda' if requested == 'cuda' or (requested == 'auto' and torch.cuda.is_available()) else 'cpu'
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    config = vars(args).copy()
    config.pop('worker', None)
    config['output'] = str(args.output)
    config['agent'] = AGENT
    config['device_selected'] = device
    config['torch'] = torch.__version__
    config['cuda_available'] = bool(torch.cuda.is_available())
    config['cuda_device'] = (torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else None)
    config['source_hashes'] = source_hashes()
    save_json(args.output / 'config.json', config)

    for index, seed in enumerate(args.seeds):
        directory = args.output / f'seed_{seed}'
        directory.mkdir()
        run_config = config.copy()
        run_config.update({
            'seed': seed,
            'output': str(directory),
            'board_start': args.board_start + index * args.rounds,
            'device': device,
        })
        config_path = directory / 'config.json'
        save_json(config_path, run_config)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(config_path)]
        subprocess.run(command, check=True)

    print(f'Finished {len(args.seeds)} DQN runs on {device}. Results: {args.output}')


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        with open(args.worker) as file:
            train_one(json.load(file))
    else:
        run_experiment(args)


if __name__ == '__main__':
    main()
