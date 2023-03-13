"""
This is a simple example training script.
"""
import argparse
import json
import os
import sys

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


sys.path.insert(0, '../../overcooked_ai/src/')
sys.path.insert(0, '../../overcooked_ai/src/overcooked_ai_py')
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.utils import set_random_seed
from ipm.algos import ddt_ppo_policy
from ipm.algos import binary_ddt_ppo_policy
from tqdm import tqdm
import sys

sys.path.insert(0, '../../overcooked_ai/src/')
sys.path.insert(0, '../../overcooked_ai/src/overcooked_ai_py')
from ipm.algos.genetic_algorithm import GA_DT_Optimizer
from ipm.models.idct import IDCT
from ipm.models.bc_agent import get_human_bc_partner
from ipm.overcooked.overcooked_envs import OvercookedSelfPlayEnv, OvercookedRoundRobinEnv, OvercookedPlayWithFixedPartner
from stable_baselines3.common.monitor import Monitor
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import shutil
from stable_baselines3.common.results_plotter import load_results, ts2xy


def get_agents(folder_name):
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    agents = []
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if file.endswith('.zip') and 'final' in file:
                agent = PPO.load(os.path.join(root, file), custom_objects=custom_objects)
                agents.append(agent)
    return agents


def main(layout_name, folder_name):
    reduce_state_space = False
    use_skills = False
    agents = get_agents(folder_name)
    all_rewards = []

    for agent in tqdm(agents):
        total_reward = 0.0
        env = OvercookedSelfPlayEnv(layout_name=layout_name, seed_num=0,
                                    reduced_state_space_ego=reduce_state_space,
                                    reduced_state_space_alt=reduce_state_space,
                                    use_skills_ego=use_skills,
                                    use_skills_alt=use_skills)
        obs = env.reset()
        done = False
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            reward = env.joint_reward
            total_reward += reward

        all_rewards.append(total_reward)
    print('Total reward mean:', np.mean(all_rewards))
    print('Total reward std:', np.std(all_rewards))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains self-play agent on overcooked with checkpointing')
    parser.add_argument('--layout_name', help='the name of the layout to train on', type=str, default='forced_coordination')
    parser.add_argument('--folder_name', help='folder name containing the self-play agents', type=str)
    args = parser.parse_args()
    main(layout_name=args.layout_name, folder_name=args.folder_name)

