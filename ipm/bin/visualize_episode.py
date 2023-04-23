import argparse
import configparser
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pygame
import torch
from ipm.bin.utils import play_episode_together, play_episode_together_get_states
from ipm.gui.experiment_gui_utils import SettingsWrapper
from ipm.models.bc_agent import StayAgent
from ipm.models.decision_tree import sparse_ddt_to_decision_tree
from ipm.models.human_model import HumanModel
from ipm.models.robot_model import RobotModel
from ipm.overcooked.overcooked_envs import OvercookedPlayWithFixedPartner, OvercookedJointRecorderEnvironment
from stable_baselines3 import PPO
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from tqdm import tqdm
from ipm.bin.utils import visualize_state

def watch_episode(env, states) -> float:
    """
    Play an episode of the game with two agents

    :param env: environemnt so we can extract layout
    :param policy_a: policy of the first agent
    :param policy_b: policy of the second agent
    """

    pygame.init()
    width = 800
    height = 600
    screen = pygame.display.set_mode((width, height))
    visualizer = StateVisualizer()

    for state in states:
        visualize_state(visualizer=visualizer, screen=screen, env=env, state=state, width=width, height=height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune various hyperaparameters')
    parser.add_argument('--states_file', help='States file', type=str)
    parser.add_argument('--layout', help='States file', type=str)
    args = parser.parse_args()

    settings = SettingsWrapper()

    dummy_env = OvercookedPlayWithFixedPartner(partner=StayAgent(), layout_name=args.layout,
                                               behavioral_model='dummy',
                                               reduced_state_space_ego=True, reduced_state_space_alt=True,
                                               use_skills_ego=True, use_skills_alt=True)

    # load pickle file states
    with open(args.states_file, 'rb') as f:
        states = pickle.load(f)
    watch_episode(dummy_env, states)

