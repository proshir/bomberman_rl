import json
import importlib
import logging
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
from environment import BombeRLeWorld

SOURCE_DIR = Path(__file__).resolve().parent


class BenchmarkWorld(BombeRLeWorld):
    """A game world that rotates the agent lineup through the four corners."""

    def build_arena(self):
        arena, coins, agents = super().build_arena()
        corners = [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]
        for i, agent in enumerate(agents):
            agent.x, agent.y = corners[(self.args.seat + i) % len(corners)]
        return arena, coins, agents


def save_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def play_game(config):
    """Play one game and return the results for every agent."""
    s.MAX_STEPS = config['max_steps']
    random.seed(config['agent_seed'])
    if config.get('model_path'):
        callbacks = importlib.import_module(f"agent_code.{config['agents'][0]}.callbacks")
        callbacks.MODEL_PATH = Path(config['model_path']).resolve()
    if config.get('feature_mode'):
        callbacks = importlib.import_module(f"agent_code.{config['agents'][0]}.callbacks")
        callbacks.FEATURE_MODE = config['feature_mode']
    s.LOG_GAME = logging.WARNING
    s.LOG_AGENT_WRAPPER = logging.WARNING
    s.LOG_AGENT_CODE = logging.WARNING
    args = SimpleNamespace(**config, no_gui=True, save_replay=False,
                           save_stats=False, match_name=None,
                           continue_without_training=True, silence_errors=False)
    world = BenchmarkWorld(args, [(name, False) for name in config['agents']])
    world.new_round()
    starts = [agent.get_state()[-1] for agent in world.agents]
    coin_task = config['scenario'] == 'coin-heaven' and len(world.agents) == 1
    completion_steps = None
    visited = set()
    repeated_states = 0
    started = perf_counter()
    while world.running:
        if coin_task:
            position = world.agents[0].get_state()[-1]
            coins = tuple(sorted(coin.get_state() for coin in world.coins if coin.collectable))
            state = (position, coins)
            if state in visited:
                repeated_states += 1
            visited.add(state)
        world.do_step()
        if (coin_task and completion_steps is None and
                world.agents[0].statistics['coins'] == len(world.coins)):
            completion_steps = world.step
    elapsed = perf_counter() - started
    results = []
    for agent, start in zip(world.agents, starts):
        stats = agent.statistics
        results.append({
            'name': agent.code_name,
            'start': start,
            'score': agent.score,
            'coins': stats['coins'],
            'crates': stats['crates'],
            'kills': stats['kills'],
            'suicides': stats['suicides'],
            'bombs': stats['bombs'],
            'steps': stats['steps'],
            'dead': agent.dead, 'invalid_actions': stats['invalid'],
        })
        if coin_task:
            results[-1]['repeated_states'] = repeated_states
            policy = agent.backend.runner.fake_self
            if hasattr(policy, 'loop_interventions'):
                results[-1]['loop_interventions'] = policy.loop_interventions
            results[-1]['completed'] = completion_steps is not None
            results[-1]['completion_steps'] = completion_steps
    world.end()
    return {'seed': config['seed'], 'agent_seed': config['agent_seed'],
            'seat': config['seat'], 'steps': world.step,
            'elapsed_seconds': elapsed, 'agents': results}


def confidence_interval(values, samples, seed):
    """Return a bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        sample = rng.choice(values, len(values), replace=True)
        means.append(sample.mean())
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(results, candidates, metric, seeds, samples, seed):
    summary = {'metric': metric, 'agents': {}, 'comparisons': {}}
    values_by_agent = {}
    for candidate in candidates:
        candidate_games = [game for game in results if game['candidate'] == candidate]
        # Average corners and action seeds first, then compare board averages.
        values = []
        for game_seed in seeds:
            scores = [game['agents'][0][metric] for game in candidate_games
                      if game['seed'] == game_seed]
            values.append(float(np.mean(scores)))
        values_by_agent[candidate] = values
        summary['agents'][candidate] = {
            'mean': float(np.mean(values)),
            'ci95': confidence_interval(values, samples, seed),
            'board_means': values,
        }
        games = [game['agents'][0] for game in candidate_games]
        # Sahand was here. Keep official outcomes beside the primary metric so
        # combat improvements cannot be hidden by one aggregate score.
        for diagnostic in ('score', 'coins', 'crates', 'kills', 'suicides',
                           'bombs', 'steps'):
            if all(diagnostic in game for game in games):
                summary['agents'][candidate]['mean_' + diagnostic] = float(
                    np.mean([game[diagnostic] for game in games]))
        if all('dead' in game for game in games):
            summary['agents'][candidate]['survival_rate'] = float(
                np.mean([not game['dead'] for game in games]))
        for diagnostic in ('invalid_actions', 'repeated_states', 'loop_interventions'):
            if all(diagnostic in game for game in games):
                summary['agents'][candidate]['mean_' + diagnostic] = float(
                    np.mean([game[diagnostic] for game in games]))
        if all('completed' in game for game in games):
            completed = [game['completion_steps'] for game in games if game['completed']]
            summary['agents'][candidate].update({
                'games': len(games),
                'successes': len(completed),
                'completion_rate': len(completed) / len(games),
                'mean_completion_steps': float(np.mean(completed)) if completed else None,
            })
    baseline = values_by_agent[candidates[0]]
    for candidate in candidates[1:]:
        difference = np.asarray(values_by_agent[candidate]) - baseline
        summary['comparisons'][candidate] = {
            'mean_difference': float(np.mean(difference)),
            'ci95': confidence_interval(difference, samples, seed),
        }
    return summary


def parse_args(argv=None):
    parser = ArgumentParser(description='Compare agents on matching Bomberman games.')
    parser.add_argument('--agents', nargs='+',
                        help='Agents to compare; the first one is the baseline.')
    parser.add_argument('--opponents', nargs='*', default=[],
                        help='Opponents used for every candidate.')
    parser.add_argument('--scenario', choices=s.SCENARIOS, default='coin-heaven')
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='Board seeds to evaluate.')
    parser.add_argument('--agent-seeds', type=int, nargs='+', default=[0],
                        help='Random seeds for agent decisions.')
    parser.add_argument('--seats', type=int, nargs='+', choices=range(4),
                        default=[0, 1, 2, 3], help='Starting corners to use.')
    parser.add_argument('--max-steps', type=int, default=s.MAX_STEPS)
    parser.add_argument('--metric', choices=['coins', 'score'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-path', type=Path, help='Checkpoint for a single candidate.')
    parser.add_argument('--feature-mode', choices=['compact', 'position', 'distance', 'rich'],
                        default='distance')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--analysis-seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Games evaluated in each worker process (default: 1).')
    parser.add_argument('--worker', type=Path, help=SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        return args

    if not args.agents or not args.seeds or not args.output:
        parser.error('--agents, --seeds, and --output are required.')
    names = args.agents + args.opponents
    if args.model_path and (len(args.agents) != 1 or args.agents[0] in args.opponents):
        parser.error('--model-path requires one candidate, not repeated as an opponent.')
    if len(args.opponents) >= s.MAX_AGENTS:
        parser.error('At most three opponents are allowed.')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Board seeds must be unique.')
    for name in names:
        if not (SOURCE_DIR / 'agent_code' / name / 'callbacks.py').is_file():
            parser.error(f'Unknown agent: {name}')
    if args.output.exists():
        parser.error('Output directory already exists.')
    if args.max_steps < 1:
        parser.error('The step limit must be positive.')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')

    return args


def run_benchmark(args):
    metric = args.metric or ('score' if args.opponents else 'coins')
    model_path = str(args.model_path.resolve()) if args.model_path else None
    args.output.mkdir(parents=True)
    saved_args = vars(args).copy()
    saved_args['output'] = str(args.output)
    saved_args['model_path'] = model_path
    save_json(args.output / 'config.json', saved_args)
    tasks = []
    for seed in args.seeds:
        for agent_seed in args.agent_seeds:
            for seat in args.seats:
                for candidate in args.agents:
                    game_dir = args.output / 'games' / f'{len(tasks):04d}'
                    game_dir.mkdir(parents=True)
                    config = {
                        'model_path': model_path,
                        'feature_mode': args.feature_mode,
                        'agents': [candidate] + args.opponents,
                        'scenario': args.scenario,
                        'seed': seed,
                        'agent_seed': agent_seed,
                        'seat': seat,
                        'max_steps': args.max_steps,
                        'log_dir': str(game_dir),
                        'agent_log_dir': str(game_dir / 'agents'),
                    }
                    save_json(game_dir / 'config.json', config)
                    tasks.append((game_dir, candidate, config))

    results = []
    with open(args.output / 'games.jsonl', 'w') as file:
        batches = [tasks[start:start + args.batch_size]
                   for start in range(0, len(tasks), args.batch_size)]
        for index, batch in enumerate(tqdm(batches)):
            batch_dir = args.output / 'batches'
            batch_dir.mkdir(exist_ok=True)
            batch_path = batch_dir / f'{index:04d}.json'
            result_path = batch_dir / f'{index:04d}_results.json'
            save_json(batch_path, {
                'games': [config for _, _, config in batch],
                'result_path': str(result_path),
            })
            command = [sys.executable, __file__, '--worker', str(batch_path)]
            subprocess.run(command, check=True)
            with open(result_path) as result_file:
                batch_results = json.load(result_file)
            if len(batch_results) != len(batch):
                raise RuntimeError('Worker returned the wrong number of game results.')
            for result, (_, candidate, _) in zip(batch_results, batch):
                result['candidate'] = candidate
                file.write(json.dumps(result) + '\n')
                results.append(result)
    summary = summarize(results, args.agents, metric, args.seeds,
                        args.bootstrap_samples, args.analysis_seed)
    save_json(args.output / 'summary.json', summary)
    print(f'Results saved to {args.output}')
    for candidate, result in summary['agents'].items():
        print(f'{candidate}: {metric} = {result["mean"]:.2f}')


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        with open(args.worker) as file:
            config = json.load(file)
        if 'games' in config:
            results = [play_game(game) for game in config['games']]
            save_json(Path(config['result_path']), results)
        else:
            result = play_game(config)
            save_json(args.worker.parent / 'result.json', result)
    else:
        run_benchmark(args)


if __name__ == '__main__':
    main()
