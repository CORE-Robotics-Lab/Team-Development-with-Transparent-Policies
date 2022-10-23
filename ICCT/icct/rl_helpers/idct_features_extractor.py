import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from ICCT.icct.core.idct import IDCT


class IDCTFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim, action_dim, ddt_kwargs):
        super(IDCTFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.idct = IDCT(input_dim=features_dim,
             output_dim=action_dim,
             weights=None,
             comparators=None,
             leaves=ddt_kwargs['num_leaves'],
             alpha=None,
             use_individual_alpha=ddt_kwargs['use_individual_alpha'],
             device=ddt_kwargs['device'],
             hard_node=ddt_kwargs['hard_node'],
             argmax_tau=ddt_kwargs['argmax_tau'],
             l1_hard_attn=ddt_kwargs['l1_hard_attn'],
             use_gumbel_softmax=ddt_kwargs['use_gumbel_softmax'],
             is_value=True,
             alg_type=ddt_kwargs['alg_type']).to(ddt_kwargs['device'])

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.idct(observations)
