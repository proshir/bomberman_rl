"""Replay frozen DDQN checkpoints and record action/value/mask evidence.

Analysis runs after each engine step, outside the timed action callback.
This does not alter the policy, opponents, board, or training process.
"""

import argparse
import hashlib
import importlib
import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import settings as s
from run_benchmark import BenchmarkWorld
from agent_code.combat_dqn_agent.model import DEVICE
from agent_code.combat_fqi_agent.safety import bomb_is_useful, can_survive_action
from agent_code.combat_fqi_history_antistag_agent.train import reward_from_transition
from agent_code.Agent_040_optimized_compact_ddqn_agent.features import StateContext, ACTIONS, MOVE_DELTAS
from agent_code.Agent_041_dynamic_nav_ddqn_agent.features import _navigation_features


def encode(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=0, help='training seed')
    parser.add_argument('--board', type=int, default=34000)
    parser.add_argument('--seat', type=int, default=1)
    parser.add_argument('--scenario', default='classic')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cb = importlib.import_module(f'agent_code.{args.agent}.callbacks')
    original_act = cb.act
    captured = []

    def capture(self, game_state):
        action = original_act(self, game_state)
        captured.append((self, game_state, action))
        return action

    cb.act = capture
    summaries = []
    s.LOG_GAME = s.LOG_AGENT_CODE = s.LOG_AGENT_WRAPPER = logging.WARNING
    s.MAX_STEPS = 400
    for episode in args.episodes:
        checkpoint = args.run / f'seed_{args.seed}/checkpoints/episode_{episode:04d}.pkl'
        cb.MODEL_PATH = checkpoint
        random.seed(0)
        folder = args.output / f'episode_{episode:04d}'
        folder.mkdir(exist_ok=True)
        world_args = SimpleNamespace(seed=args.board, agent_seed=0, seat=args.seat,
            scenario=args.scenario, no_gui=True, save_replay=False, save_stats=False,
            match_name=None, continue_without_training=True, silence_errors=False,
            log_dir=str(folder), agent_log_dir=str(folder / 'agents'))
        lineup = [(args.agent, False)]
        if args.scenario == 'classic':
            lineup += [('rule_based_agent', False)] * 3
        world = BenchmarkWorld(world_args, lineup)
        world.new_round()
        records = []
        while world.running:
            captured.clear()
            world.do_step()
            if not captured:
                continue
            learner, state, action = captured[-1]
            feature, history, actions, successes, signature, last_progress = learner.feature_cache[cb.state_key(state)]
            context = StateContext(state)
            with torch.inference_mode():
                q = learner.policy_net(torch.as_tensor(feature, device=DEVICE).unsqueeze(0))[0].cpu().tolist()
            pos = tuple(state['self'][3])
            enemy_distances = context.distance_map(tuple(o[3] for o in state['others']))
            coin_distances = context.distance_map(state['coins'])
            destinations = [(pos[0]+MOVE_DELTAS[a][0], pos[1]+MOVE_DELTAS[a][1]) for a in ACTIONS]
            actor = world.agents[0]
            post = world.get_state_for_agent(actor)
            records.append(dict(step=state['step'], position=pos, action=action,
                actual_action=world.replay['actions'][actor.name][-1], events=list(actor.events),
                reward=reward_from_transition(state, action, post, actor.events),
                q=q, features=feature.tolist(), history=list(history),
                steps_since_global_progress=state['step']-last_progress,
                coins=state['coins'], crates=int(np.count_nonzero(state['field']==1)),
                opponents=[o[3] for o in state['others']], bombs=state['bombs'],
                danger_here=sorted(context.danger().get(pos, ())),
                legal=list(context.legal_indices()), candidates=list(context.candidate_indices()),
                bomb_useful=bomb_is_useful(state), bomb_survivable=can_survive_action(state,'BOMB'),
                coin_distance=coin_distances.get(pos), enemy_distance=enemy_distances.get(pos),
                enemy_destination_distances=[enemy_distances.get(p) for p in destinations],
                nav041=_navigation_features(state,context,tuple(history)+(pos,)).tolist(),
                field_hash=hashlib.sha256(state['field'].tobytes()).hexdigest()))
        stats = dict(world.agents[0].statistics)
        summary = dict(episode=episode, checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            board=args.board, seat=args.seat, scenario=args.scenario,
            score=world.agents[0].score, dead=world.agents[0].dead, stats=stats,
            steps=world.step, decisions=len(records))
        (folder/'trace.json').write_text(json.dumps(records, default=encode))
        (folder/'summary.json').write_text(json.dumps(summary,indent=2, default=encode))
        summaries.append(summary)
        print(json.dumps(summary, default=encode),flush=True)
        world.end()
        for logger in logging.Logger.manager.loggerDict.values():
            if isinstance(logger,logging.Logger):
                for handler in list(logger.handlers):
                    if isinstance(handler,logging.FileHandler):
                        logger.removeHandler(handler)
                        handler.close()
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2, default=encode))


if __name__ == '__main__':
    main()
