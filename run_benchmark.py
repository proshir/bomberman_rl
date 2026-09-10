import json
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
    started = perf_counter()
    while world.running:
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
            'steps': stats['steps'],
            'dead': agent.dead,
        })
        if coin_task:
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
    means = [rng.choice(values, len(values), replace=True).mean()
             for _ in range(samples)]
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(results, candidates, metric, seeds, samples, seed):
    summary = {'metric': metric, 'agents': {}, 'comparisons': {}}
    values_by_agent = {}
    for candidate in candidates:
        values = []
        for game_seed in seeds:
            games = [game['agents'][0][metric] for game in results
                     if game['candidate'] == candidate and game['seed'] == game_seed]
            values.append(float(np.mean(games)))
        values_by_agent[candidate] = values
        summary['agents'][candidate] = {
            'mean': float(np.mean(values)),
            'ci95': confidence_interval(values, samples, seed),
            'board_means': values,
        }
        games = [game['agents'][0] for game in results if game['candidate'] == candidate]
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


def main(argv=None):
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
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--analysis-seed', type=int, default=0)
    parser.add_argument('--worker', type=Path, help=SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        with open(args.worker) as file:
            config = json.load(file)
        save_json(args.worker.parent / 'result.json', play_game(config))
        return

    if not args.agents or not args.seeds or not args.output:
        parser.error('--agents, --seeds, and --output are required.')
    names = args.agents + args.opponents
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

    metric = args.metric or ('score' if args.opponents else 'coins')
    args.output.mkdir(parents=True)
    saved_args = vars(args).copy()
    saved_args['output'] = str(args.output)
    save_json(args.output / 'config.json', saved_args)
    results = []
    with open(args.output / 'games.jsonl', 'w') as file:
        for seed in tqdm(args.seeds):
            for agent_seed in args.agent_seeds:
                for seat in args.seats:
                    for candidate in args.agents:
                        game_dir = args.output / 'games' / f'{len(results):04d}'
                        game_dir.mkdir(parents=True)
                        config = {
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
                        subprocess.run([sys.executable, __file__, '--worker',
                                        str(game_dir / 'config.json')], check=True)
                        with open(game_dir / 'result.json') as result_file:
                            result = json.load(result_file)
                        result['candidate'] = candidate
                        file.write(json.dumps(result) + '\n')
                        results.append(result)
    summary = summarize(results, args.agents, metric, args.seeds,
                        args.bootstrap_samples, args.analysis_seed)
    save_json(args.output / 'summary.json', summary)
    print(f'Results saved to {args.output}')
    for candidate, result in summary['agents'].items():
        print(f'{candidate}: {metric} = {result["mean"]:.2f}')


if __name__ == '__main__':
    main()
