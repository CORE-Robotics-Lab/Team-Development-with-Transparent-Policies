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

from ipm.models.bc_agent import get_pretrained_teammate_finetuned_with_bc

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
        if type(self.partner).__name__ == 'HumanModel':
            teammate_action = self.partner.predict(obs)
        else:
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
    def step(self, joint_action: Tuple[int, int]):

        a_p1, skill_rew_p1 = self.idx_to_skill_ego[joint_action[0]](agent_idx=0)
        a_p2, skill_rew_p2 = self.idx_to_skill_alt[joint_action[1]](agent_idx=1)

        joint_action = (a_p1, a_p2)

        next_state, reward, done, info = self.base_env.step(joint_action)
        self.state = next_state

        # reward shaping
        reward_p1 = reward + info['shaped_r_by_agent'][0] + skill_rew_p1
        reward_p2 = reward + info['shaped_r_by_agent'][1] + skill_rew_p2

        (obs_p0, obs_p1) = self.featurize_fn(next_state)
        self.raw_obs = (obs_p0, obs_p1)
        obs_p0, obs_p1 = self.reduced_featurize_fn(self.base_env.state)
        self.reduced_obs = (obs_p0, obs_p1)

        joint_obs = (obs_p0, obs_p1)
        joint_rew = (reward_p1, reward_p2)

        if done:
            return joint_obs, joint_rew, done, info

        # self.timestep += 1

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


class OvercookedJointRecorderEnvironment(OvercookedMultiAgentEnv):
    def step(self, macro_joint_action: Tuple[int, int], use_reduced=False):

        if self.use_skills_ego:
            p0_action, _ = self.idx_to_skill_ego[macro_joint_action[0]](agent_idx=0)
        else:
            p0_action = Action.INDEX_TO_ACTION[macro_joint_action[0]]

        if self.use_skills_alt:
            p1_action, _ = self.idx_to_skill_alt[macro_joint_action[1]](agent_idx=1)
        else:
            p1_action = Action.INDEX_TO_ACTION[macro_joint_action[1]]

        raw_joint_action = (p0_action, p1_action)

        next_state, reward, done, info = self.base_env.step(raw_joint_action)
        self.state = next_state

        self.timestep = next_state.timestep
        self.prev_macro_action = macro_joint_action

        # reward shaping
        reward_p0 = reward + info['shaped_r_by_agent'][0]
        reward_p1 = reward + info['shaped_r_by_agent'][1]

        (obs_p0, obs_p1) = self.featurize_fn(next_state)
        self.raw_obs = (obs_p0, obs_p1)
        (obs_p0, obs_p1) = self.reduced_featurize_fn(next_state)
        self.reduced_obs = (obs_p0, obs_p1)

        if self.behavioral_model is not None:
            obs_p0_with_intent = self.add_intent(obs_p0, obs_p1, 0)
            obs_p1_with_intent = self.add_intent(obs_p1, obs_p0, 1)
            obs_p0 = obs_p0_with_intent
            obs_p1 = obs_p1_with_intent

        joint_obs = (obs_p0, obs_p1)
        joint_rew = (reward_p0, reward_p1)

        return joint_obs, joint_rew, done, info

    def reset(self, use_reduced=False):
        self.base_env.reset()
        self.state = self.base_env.state
        if use_reduced:
            (obs_p0, obs_p1) = self.reduced_featurize_fn(self.base_env.state)
            if self.behavioral_model is not None:
                obs_p0_with_intent = self.add_intent(obs_p0, obs_p1, 0)
                obs_p1_with_intent = self.add_intent(obs_p1, obs_p0, 1)
                obs_p0 = obs_p0_with_intent
                obs_p1 = obs_p1_with_intent
        else:
            obs_p0, obs_p1 = self.featurize_fn(self.base_env.state)
        self._obs = (obs_p0, obs_p1)
        return self._obs

    def check_conditions(self):
        pass

    def get_teammate_action(self):
        pass

    def update_ego(self):
        pass
