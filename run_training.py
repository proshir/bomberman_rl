import hashlib
import importlib
import json
import logging
import pickle
import random
import subprocess
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

import settings as s
from run_benchmark import BenchmarkWorld, SOURCE_DIR, save_json


def evaluate(config, learner, episode, interactions):
    """Save the current Q-table and evaluate it in separate game processes."""
    directory = Path(config['output'])
    checkpoint = directory / 'checkpoints' / f'episode_{episode:04d}.pkl'
    with open(checkpoint, 'wb') as file:
        pickle.dump(learner.q_table, file)
    output = directory / 'evaluation' / f'episode_{episode:04d}'
    command = [
        sys.executable, str(SOURCE_DIR / 'run_benchmark.py'),
        '--agents', config['agent'], '--scenario', 'coin-heaven',
        '--feature-mode', config['feature_mode'],
        '--model-path', str(checkpoint), '--max-steps', str(config['max_steps']),
        '--seeds', *map(str, config['eval_seeds']),
        '--agent-seeds', '0', '--seats', '0', '1', '2', '3',
        '--output', str(output),
    ]
    output.parent.mkdir(exist_ok=True)
    with open(output.with_suffix('.log'), 'w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(output / 'summary.json') as file:
        result = json.load(file)['agents'][config['agent']]
    return {
        'episode': episode,
        'interactions': interactions,
        'checkpoint': str(checkpoint),
        **result,
    }


def train(config):
    """Train one independent table and record each round and evaluation."""
    directory = Path(config['output'])
    callbacks = importlib.import_module(f"agent_code.{config['agent']}.callbacks")
    callbacks.MODEL_PATH = directory / 'training.pkl'
    callbacks.FEATURE_MODE = config['feature_mode']
    s.MAX_STEPS = config['max_steps']
    s.LOG_GAME = s.LOG_AGENT_WRAPPER = s.LOG_AGENT_CODE = logging.WARNING
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    args = SimpleNamespace(seed=config['seed'], agent_seed=config['seed'],
                           seat=0, scenario='coin-heaven', no_gui=True,
                           save_replay=False, save_stats=False, match_name=None,
                           continue_without_training=True, silence_errors=False,
                           log_dir=str(directory / 'logs'),
                           agent_log_dir=str(directory / 'logs' / 'agents'))
    (directory / 'checkpoints').mkdir()
    (directory / 'logs').mkdir()
    world = BenchmarkWorld(args, [(config['agent'], True)])
    learner = world.agents[0].backend.runner.fake_self
    curve = [evaluate(config, learner, 0, 0)]
    save_json(directory / 'learning_curve.json', curve)
    interactions = 0
    training_seconds = 0.0
    with open(directory / 'rounds.jsonl', 'w') as file:
        for episode in tqdm(range(1, config['rounds'] + 1), desc=f"Seed {config['seed']}"):
            board_seed = config['board_start'] + episode - 1
            world.rng = np.random.default_rng(board_seed)
            args.seat = (episode - 1) % 4
            epsilon = learner.epsilon
            started = perf_counter()
            world.new_round()
            while world.running:
                world.do_step()
            training_seconds += perf_counter() - started
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
                'reward': learner.last_round_reward,
                'epsilon': epsilon,
                'table_size': len(learner.q_table),
                'training_seconds': training_seconds,
            }
            file.write(json.dumps(record) + '\n')
            file.flush()
            if episode % config['eval_every'] == 0 or episode == config['rounds']:
                curve.append(evaluate(config, learner, episode, interactions))
                save_json(directory / 'learning_curve.json', curve)
    world.end()


def parse_args(argv=None):
    parser = ArgumentParser(description='Train and evaluate the coin Q-table agent.')
    parser.add_argument('--agent', default='q_table_agent')
    parser.add_argument('--feature-mode', choices=['compact', 'position', 'distance', 'rich'],
                        default='distance')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--rounds', type=int, default=300)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=list(range(10000, 10008)))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path, help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return args
    if args.output is None or args.output.exists():
        parser.error('Choose a new --output directory.')
    if min(args.rounds, args.max_steps, args.eval_every) < 1:
        parser.error('Round counts and step limits must be positive.')
    for seeds in (args.seeds, args.eval_seeds):
        if len(set(seeds)) != len(seeds):
            parser.error('Seeds must be unique within each list.')
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds + args.eval_seeds):
        parser.error('Seeds must be between 0 and 2**32 - 1.')
    board_seeds = range(1000, 1000 + args.rounds * len(args.seeds))
    if set(board_seeds).intersection(args.eval_seeds):
        parser.error('Evaluation seeds overlap the training board seeds.')
    return args


def run_training(args):
    module = importlib.import_module(f'agent_code.{args.agent}.train')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    config = vars(args).copy()
    config['output'] = str(args.output)
    config['hyperparameters'] = {
        name: value for name, value in vars(module).items()
        if name.isupper() and isinstance(value, (int, float))
    }
    config['code_version'] = subprocess.check_output(
        ['git', '-C', str(SOURCE_DIR), 'rev-parse', 'HEAD'], text=True).strip()
    paths = [SOURCE_DIR / 'run_training.py', SOURCE_DIR / 'run_benchmark.py',
             SOURCE_DIR / 'environment.py', SOURCE_DIR / 'agents.py',
             SOURCE_DIR / 'settings.py', SOURCE_DIR / 'events.py',
             SOURCE_DIR / 'agent_code' / args.agent / 'callbacks.py',
             SOURCE_DIR / 'agent_code' / args.agent / 'train.py']
    config['source_hashes'] = {
        str(path.relative_to(SOURCE_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    config['python'] = sys.version
    config['numpy'] = np.__version__
    save_json(args.output / 'config.json', config)
    curve = []
    for index, seed in enumerate(args.seeds):
        directory = args.output / f'seed_{seed}'
        directory.mkdir()
        run_config = config.copy()
        run_config['seed'] = seed
        run_config['output'] = str(directory)
        run_config['board_start'] = 1000 + index * args.rounds
        config_path = directory / 'config.json'
        save_json(config_path, run_config)
        # Each training run starts with fresh agent state and random generators.
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(config_path)]
        subprocess.run(command, check=True)
        with open(directory / 'learning_curve.json') as file:
            curve.extend({'seed': seed, **row} for row in json.load(file))
    save_json(args.output / 'learning_curve.json', curve)
    for row in curve:
        print(f"Seed {row['seed']}, episode {row['episode']}: coins = {row['mean']:.2f}")


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        with open(args.worker) as file:
            config = json.load(file)
        train(config)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
