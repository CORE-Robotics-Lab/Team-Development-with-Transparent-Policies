import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import sys

from stable_baselines3 import PPO

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    import pickle5 as pickle
else:
    import pickle
from sklearn.model_selection import train_test_split
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from sklearn.model_selection import train_test_split
from ipm.overcooked.observation_reducer import ObservationReducer
import torch
import torch.nn.functional as F

class FCPModel:
    def __init__(self, dummy_env, filepath):
        model = PPO("MlpPolicy", dummy_env)
        weights = torch.load(filepath)
        model.policy.load_state_dict(weights['ego_state_dict'])
        self.fcp_policy = model.policy
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fcp_policy.to(device)

    def predict(self, obs):
        """
        Args:
            obs: observation from environment

        Returns:
            action: action to take
        """
        # reshape into a torch batch of 1
        # obs = torch.from_numpy(obs).to(self.human_ppo_policy.device).float()
        obs = torch.tensor(obs)
        action, _ = self.fcp_policy.predict(obs)
        return action

        obs = obs.unsqueeze(0)
        actions, vals, log_probs = self.fcp_policy.forward(obs, deterministic=False)
        return actions[0].item()
