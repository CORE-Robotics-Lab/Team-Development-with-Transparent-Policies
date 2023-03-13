from ipm.overcooked.overcooked_base import OvercookedMultiAgentEnv
import os
from abc import abstractmethod, ABC

import gym
import numpy as np
import sys
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS
from typing import List, Tuple, Dict, Optional

from sklearn.ensemble import RandomForestClassifier
from stable_baselines3 import PPO

from ipm.models.bc_agent import get_human_bc_partner

class OvercookedSelfPlayEnv(OvercookedMultiAgentEnv):
    def check_conditions(self):
        assert self.reduced_state_space_ego == self.reduced_state_space_alt

    def get_teammate_action(self):
        # for the self-play case, we just stand still while we
        # alternate between the agents
        # this is because we need a joint action at every timestep
        # but would like to use this same policy for a single agent
        STAY_IDX = 4
        return STAY_IDX

    def update_ego(self):
        # for the self-play case, we alternate between both the agents
        # where one agent makes a move, and the other stands still
        self.current_alt_idx = self.current_ego_idx
        self.current_ego_idx = (self.current_ego_idx + 1) % 2

class OvercookedPlayWithFixedPartner(OvercookedMultiAgentEnv):
    def __init__(self, partner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load in the potential teammates
        self.partner = partner

    def check_conditions(self):
        pass

    def get_teammate_action(self):
        obs = self._obs[self.current_alt_idx]
        teammate_action, _states = self.partner.predict(obs)
        return teammate_action

    def update_ego(self):
        # for round-robin, we never change these since we have 1 single agent that is learning
        pass

    def reset(self):
        self.ego_obs = super().reset()
        return self.ego_obs

class OvercookedRoundRobinEnv(OvercookedMultiAgentEnv):
    def __init__(self, teammate_locations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load in the potential teammates
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
        self.teammate_locations = teammate_locations
        # potential teammates are in the data folder.
        self.all_teammates = []
        # iterate through all subfolders and load all files

        custom_objects = {}
        if newer_python_version:
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }

        for root, dirs, files in os.walk( self.teammate_locations):
            for file in files:
                if file.endswith('.zip') and 'final' in file:
                    agent = PPO.load(os.path.join(root, file), custom_objects=custom_objects)
                    self.all_teammates.append(agent)
        self.teammate_idx = np.random.randint(len(self.all_teammates))

    def check_conditions(self):
        pass

    def get_teammate_action(self):
        obs = self._obs[self.current_alt_idx]
        teammate_action, _states = self.all_teammates[self.teammate_idx].predict(obs)
        return teammate_action

    def update_ego(self):
        # for round-robin, we never change these since we have 1 single agent that is learning
        pass

    def reset(self):
        self.ego_obs = super().reset()
        self.teammate_idx = np.random.randint(len(self.all_teammates))
        return self.ego_obs

class OvercookedJointEnvironment(OvercookedMultiAgentEnv):
    def __init__(self, layout_name, n_timesteps=200):
        super().__init__(layout_name, ego_idx=0, reduced_state_space_ego=False, use_skills_ego=False,
                            reduced_state_space_alt=False, use_skills_alt=False, seed_num=0, n_timesteps=n_timesteps,
                            behavioral_model=None, failed_skill_rew=0.0, double_cook_times=False)

    def step(self, joint_action: Tuple[int, int]):

        joint_action = Action.INDEX_TO_ACTION[joint_action[0]], Action.INDEX_TO_ACTION[joint_action[1]]

        next_state, reward, done, info = self.base_env.step(joint_action)
        self.state = next_state

        # reward shaping
        reward_ego = reward + info['shaped_r_by_agent'][self.current_ego_idx]
        reward_alt = reward + info['shaped_r_by_agent'][self.current_alt_idx]

        (obs_p0, obs_p1) = self.featurize_fn(next_state)

        joint_obs = (obs_p0, obs_p1)
        joint_rew = (reward_ego, reward_alt)

        if done:
            return joint_obs, joint_rew, done, info

        return joint_obs, joint_rew, done, info

    def reset(self):
        self.base_env.reset()
        self.state = self.base_env.state
        obs_p0, obs_p1 = self.featurize_fn(self.base_env.state)
        self._obs = (obs_p0, obs_p1)
        return self._obs

    def check_conditions(self):
        pass

    def get_teammate_action(self):
        pass

    def update_ego(self):
        pass
