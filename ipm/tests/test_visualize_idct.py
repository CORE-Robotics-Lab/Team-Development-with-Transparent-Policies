import gym
import numpy as np
import copy
import argparse
import random
import os
import torch
from ipm.algos import ddt_sac_policy
from ipm.algos import ddt_td3_policy
from ipm.models.idct import IDCT
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.preprocessing import get_action_dim
from ipm.models.icct_helpers import convert_to_crisp
from ipm.gui.tree_gui_utils import TreeCreationPage
from ipm.algos.save_after_ep_callback import EpCheckPointCallback
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor
)

from ipm.algos.sac import SAC
from ipm.algos.td3 import TD3
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

def make_env(env_name, seed):
    set_random_seed(seed)
    if env_name == 'cartpole':
        env = gym.make('CartPole-v1')
        name = 'CartPole-v1'
    else:
        raise Exception('No valid environment selected')
    env.seed(seed)
    return env, name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ipm Training')
    parser.add_argument('--env_name', help='environment to run on', type=str, default='lunar')
    parser.add_argument('--alg_type', help='ppo is the only supported algorithm', type=str, default='ppo')
    parser.add_argument('--policy_type', help='mlp or ddt', type=str, default='ddt')
    parser.add_argument('--visualization_output', help='file location to export a visualization of the ipm.', type=str,
                        default=None)
    parser.add_argument('--mlp_size', help='the size of mlp (small|medium|large)', type=str, default='medium')
    parser.add_argument('--seed', help='the seed number to use', type=int, default=42)
    parser.add_argument('--num_leaves', help='number of leaves used in ddt (2^n)', type=int, default=16)
    parser.add_argument('--submodels', help='if use sub-models in ddt', action='store_true', default=False)
    parser.add_argument('--hard_node', help='if use differentiable crispification', action='store_true', default=False)
    parser.add_argument('--gpu', help='if run on a GPU', action='store_true', default=False)
    parser.add_argument('--lr', help='learning rate', type=float, default=3e-4)
    parser.add_argument('--buffer_size', help='buffer size', type=int, default=1000000)
    parser.add_argument('--batch_size', help='batch size', type=int, default=256)
    parser.add_argument('--gamma', help='the discount factor', type=float, default=0.9999)
    parser.add_argument('--learning_starts',
                        help='how many steps of the model to collect transitions for before learning starts', type=int,
                        default=10000)
    parser.add_argument('--training_steps', help='total steps for training the model', type=int, default=500000)
    parser.add_argument('--argmax_tau', help='the temperature of the diff_argmax function', type=float, default=1.0)
    parser.add_argument('--ddt_lr', help='the learning rate of the ddt', type=float, default=3e-4)
    parser.add_argument('--use_individual_alpha', help='if use different alphas for different nodes',
                        action='store_true', default=False)
    parser.add_argument('--l1_reg_coeff', help='the coefficient of the l1 regularization when using l1-reg submodels',
                        type=float, default=5e-3)
    parser.add_argument('--l1_reg_bias', help='if consider biases in the l1 loss when using l1-reg submodels',
                        action='store_true', default=False)
    parser.add_argument('--l1_hard_attn',
                        help='if only sample one linear controller to perform L1 regularization for each update when using l1-reg submodels',
                        action='store_true', default=False)
    parser.add_argument('--use_gumbel_softmax',
                        help='if use gumble softmax instead of the differentiable argmax proposed in the paper',
                        action='store_true', default=False)
    # evaluation and model saving
    parser.add_argument('--min_reward', help='minimum reward to save the model', type=int)
    parser.add_argument('--save_path', help='the path of saving the model', type=str, default='test')
    parser.add_argument('--n_eval_episodes', help='the number of episodes for each evaluation during training',
                        type=int, default=5)
    parser.add_argument('--eval_freq', help='evaluation frequence of the model', type=int, default=1500)
    parser.add_argument('--log_interval', help='the number of episodes before logging', type=int, default=4)

    args = parser.parse_args()
    env, env_n = make_env(args.env_name, args.seed)
    eval_env = gym.make(env_n)
    eval_env.seed(args.seed)

    save_folder = args.save_path
    log_dir = '../../' + save_folder + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    method = 'default'
    monitor_file_path = log_dir + method + f'_seed{args.seed}'
    env = Monitor(env, monitor_file_path)
    eval_monitor_file_path = log_dir + 'eval_' + method + f'_seed{args.seed}'
    eval_env = Monitor(eval_env, eval_monitor_file_path)
    callback = EpCheckPointCallback(eval_env=eval_env, best_model_save_path=log_dir,
                                    n_eval_episodes=args.n_eval_episodes,
                                    eval_freq=args.eval_freq, minimum_reward=args.min_reward)
    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    features_extractor = FlattenExtractor

    feature_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if args.policy_type == 'ddt':
        ddt_kwargs = {
            'num_leaves': args.num_leaves,
            'hard_node': args.hard_node,
            'weights': None,
            'alpha': None,
            'comparators': None,
            'leaves': None,
            'fixed_idct': False,
            'device': args.device,
            'argmax_tau': args.argmax_tau,
            'ddt_lr': args.ddt_lr,
            'use_individual_alpha': args.use_individual_alpha,
            'l1_reg_coeff': args.l1_reg_coeff,
            'l1_reg_bias': args.l1_reg_bias,
            'l1_hard_attn': args.l1_hard_attn,
            'use_gumbel_softmax': args.use_gumbel_softmax,
            'alg_type': args.alg_type
        }
        policy_kwargs = dict(features_extractor_class=features_extractor, ddt_kwargs=ddt_kwargs)
    else:
        raise Exception('Not a valid policy type')

    if args.visualization_output is not None:
        state = torch.Tensor([[1, 0, 2, 3]])
        state = state.to(args.device)
        #
        # alpha = torch.Tensor([[-1], [1], [-1]])
        #
        # leaves = []
        # leaves.append([[2], [0], [2, -2]])
        # leaves.append([[], [0, 2], [-2, 2]])
        # leaves.append([[0, 1], [], [2, -2]])
        # leaves.append([[0], [1], [-2, 2]])
        #
        # weights = torch.Tensor([
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1],
        #     [0, 1, 0, 0]
        # ])
        #
        # comparators = torch.Tensor([[0.03], [-0.03], [0]])
        # args.num_leaves = leaves
        #
        # depth = 2

        alpha = torch.Tensor([[-1], [1], [-1], [-1], [-1]])

        leaves = [[[2], [0], [2, -2]], [[], [0, 2], [-2, 2]], [[0, 1, 3], [], [2, -2]], [[0, 1], [3], [-2, 2]],
                  [[0, 4], [1], [2, -2]], [[0], [1, 4], [-2, 2]]]

        weights = torch.Tensor([
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])

        comparators = torch.Tensor([[0.03], [-0.03], [0], [0], [0]])
        args.num_leaves = leaves
        args.fixed_idct = True


        input_dim = get_obs_shape(env.observation_space)[0]
        output_dim = get_action_dim(env.action_space)

        fresh_icct = IDCT(input_dim=input_dim,
                          output_dim=output_dim,
                          hard_node=args.hard_node,
                          device=args.device,
                          argmax_tau=args.argmax_tau,
                          use_individual_alpha=args.use_individual_alpha,
                          use_gumbel_softmax=args.use_gumbel_softmax,
                          alg_type=args.alg_type,
                          weights=weights,
                          comparators=comparators,
                          alpha=alpha,
                          leaves=leaves)

        visualizer = TreeCreationPage(fresh_icct, args.env_name)
        visualizer.modifiable_gui()
