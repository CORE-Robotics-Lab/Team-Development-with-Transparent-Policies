import sys
sys.path.insert(0, '../overcooked_ai/src/overcooked_ai_py')
sys.path.insert(0, '../overcooked_ai/src')
import numpy as np
import pygame
import time
import random
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai.src.overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action

ae = AgentEvaluator.from_layout_name(
    mdp_params={"layout_name": "cramped_room"},
    env_params={"horizon": 400},
)

horizon_env = ae.env.copy()
horizon_env.start_state_fn = ae.env.start_state_fn
horizon_env.reset()
agent1 = RandomAgent()
agent2 = RandomAgent()
# agent.set_mdp(horizon_env.mdp)
agent_pair = AgentPair(agent1, agent2)
while not horizon_env.is_done():
    s_t = horizon_env.state
    print(s_t)
    all_actions = horizon_env.mdp.get_actions(horizon_env.state)
    # a_t, a_info_t = agent.action(s_t)
    joint_action_and_infos = agent_pair.joint_action(s_t)
    a_t, a_info_t = zip(*joint_action_and_infos)
    assert all(a in Action.ALL_ACTIONS for a in a_t)
    assert all(type(a_info) is dict for a_info in a_info_t)
    display_phi = False
    s_tp1, r_t, done, info = horizon_env.step(a_t, a_info_t, display_phi)
    print(horizon_env)
    # # Getting actions and action infos (optional) for both agents
    # joint_action_and_infos = agent_pair.joint_action(s_t)























