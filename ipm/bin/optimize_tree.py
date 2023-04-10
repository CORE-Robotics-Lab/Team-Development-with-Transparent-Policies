"""
This is a simple example training script.
"""
import argparse
import json
import os
import pickle
import sys

import joblib
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ipm.models.decision_tree import sparse_ddt_to_decision_tree

sys.path.insert(0, '../../overcooked_ai/src/')
sys.path.insert(0, '../../overcooked_ai/src/overcooked_ai_py')
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.torch_layers import FlattenExtractor
from ipm.algos import ddt_ppo_policy
from ipm.algos import binary_ddt_ppo_policy
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from tqdm import tqdm
import sys
from ipm.bin.utils import CheckpointCallbackWithRew
import torch
from ipm.models.bc_agent import AgentWrapper

sys.path.insert(0, '../../overcooked_ai/src/')
sys.path.insert(0, '../../overcooked_ai/src/overcooked_ai_py')
from ipm.algos.legacy_genetic_algorithm import GA_DT_Structure_Optimizer
from ipm.models.idct import IDCT
from ipm.models.bc_agent import get_pretrained_teammate_finetuned_with_bc
from ipm.overcooked.overcooked_envs import OvercookedSelfPlayEnv, OvercookedRoundRobinEnv, OvercookedPlayWithFixedPartner
from stable_baselines3.common.monitor import Monitor
import gym
import numpy as np
from stable_baselines3 import PPO

def main(n_steps, layout_name, training_type, intent_model_file, save_dir, prior_tree_file, teammate_paths,
         n_parallel_envs=1, traj_directory=None, seed=0):

    checkpoint_freq = n_steps // 100
    # layouts of interest: 'forced_coordination'
    # 'counter_circuit', 'counter_circuit_tomato'
    save_models = True
    ego_idx = 0
    alt_idx = (ego_idx + 1) % 2

    all_rewards_across_seeds = []
    all_steps = []

    # this just calls instantiates the recipe class
    def instantiate_recipes():
        DEFAULT_ENV_PARAMS = {
            # add one because when we reset it takes up a timestep
            "horizon": 200 + 1,
            "info_level": 0,
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, rew_shaping_params=rew_shaping_params)
        base_env = OvercookedEnv.from_mdp(mdp, **DEFAULT_ENV_PARAMS)
        featurize_fn = base_env.featurize_state_mdp

    instantiate_recipes()

    intent_model = joblib.load(intent_model_file)
    intent_model = AgentWrapper(intent_model)

    seed = 0
    env = OvercookedRoundRobinEnv(teammate_locations=teammate_paths,
                                  layout_name=layout_name,
                                  behavioral_model=intent_model,
                                  seed_num=seed,
                                  ego_idx=ego_idx,
                                  reduced_state_space_ego=True,
                                  reduced_state_space_alt=False,
                                  use_skills_ego=True,
                                  use_skills_alt=True,
                                  use_true_intent=False)
    # assert traj_directory is not None
    # behavioral_model, bc_partner = get_human_bc_partner(traj_directory=traj_directory, layout_name=layout_name,
    #                                                     bc_agent_idx=alt_idx, get_intent_model=True)
    # env = OvercookedPlayWithFixedPartner(partner=bc_partner, layout_name=layout_name, seed_num=i,
    #                                      ego_idx=ego_idx,
    #                                      behavioral_model=behavioral_model,
    #                                      reduced_state_space_ego=reduce_state_space_ego,
    #                                      reduced_state_space_alt=reduce_state_space_teammate,
    #                                      use_skills_ego=high_level_actions_ego,
    #                                      use_skills_alt=high_level_actions_teammate)

    initial_model_path = os.path.join('data', layout_name, training_type + f'_optimize_tree_seed_{seed}', 'initial_model.zip')
    medium_model_path = os.path.join('data', layout_name, training_type + '_optimize_tree', 'medium_model.zip')
    final_model_path = os.path.join('data', layout_name, training_type + '_optimize_tree', 'final_model.zip')

    # if os.path.exists(save_dir):
    #     raise ValueError("Save directory already exists. Please delete it or backup the data.")
    # os.makedirs(save_dir)

    checkpoint_callback = CheckpointCallbackWithRew(
      n_steps = n_steps,
      save_freq=checkpoint_freq,
      save_path=save_dir,
      name_prefix="rl_model",
      save_replay_buffer=True,
      initial_model_path=initial_model_path,
      medium_model_path=medium_model_path,
      final_model_path=final_model_path,
      save_model=save_models,
      verbose=1
    )

    env = Monitor(env, "./" + save_dir + "/")

    optimizer = GA_DT_Structure_Optimizer(initial_depth=4, max_depth=5, env=env)
    optimizer.run()
    best_genes = optimizer.best_solution

    input_dim = get_obs_shape(env.observation_space)[0]
    output_dim = env.n_actions_ego

    # copy over the weights
    # need to debug here and extract each part and put it into the IDCT constructor below

    idct = IDCT.from_decision_tree(best_genes, input_dim, output_dim, device='cuda')

    ppo_lr = 0.0003
    ppo_batch_size = 64
    ppo_n_steps = 10000

    ddt_kwargs = {
        'num_leaves': len(model.leaf_init_information),
        'hard_node': False,
        'weights': model.layers,
        'alpha': 1.0,
        'comparators': model.comparators,
        'leaves': model.leaf_init_information,
        'fixed_idct': False,
        'device': 'cuda',
        'argmax_tau': 1.0,
        'ddt_lr': 0.001,  # this param is irrelevant for the IDCT
        'use_individual_alpha': True,
        'l1_reg_coeff': 1.0,
        'l1_reg_bias': 1.0,
        'l1_hard_attn': 1.0,
        'use_gumbel_softmax': False,
        'alg_type': 'ppo'
    }

    features_extractor = FlattenExtractor
    policy_kwargs = dict(features_extractor_class=features_extractor, ddt_kwargs=ddt_kwargs)

    agent = PPO("BinaryDDT_PPOPolicy", env,
                n_steps=ppo_n_steps,
                # batch_size=args.batch_size,
                # buffer_size=args.buffer_size,
                learning_rate=ppo_lr,
                policy_kwargs=policy_kwargs,
                tensorboard_log='log',
                gamma=0.99,
                verbose=1,
                seed=1
                )

    print(f'Agent training...')
    agent.learn(total_timesteps=n_steps, callback=checkpoint_callback)
    all_rewards_across_seeds.append(checkpoint_callback.all_rewards)
    all_steps = checkpoint_callback.all_steps
    print(f'Finished training agent with best average reward of {checkpoint_callback.best_mean_reward}')

    plt.clf()
    all_rewards_across_seeds = np.array(all_rewards_across_seeds)
    avg_rewards = np.mean(all_rewards_across_seeds, axis=0)
    avg_var = np.var(all_rewards_across_seeds, axis=0)
    x = all_steps
    y = avg_rewards
    plt.plot(x, y)
    upper_bound = y + avg_var
    lower_bound = y - avg_var
    # reward has to be greater than 0
    upper_bound[upper_bound < 0] = 0
    lower_bound[lower_bound < 0] = 0
    plt.grid()
    plt.xlabel('Timesteps')
    plt.ylabel('Avg. Reward')
    plt.title('Reward Curve (across seeds)')
    plt.savefig(f'{layout_name}_{training_type}_avg_reward_curve.png')

    plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)
    plt.savefig(f'{layout_name}_{training_type}_avg_reward_curve_with_var.png')

    print('Finished training all agents')

    # also save x and y to csv
    df = pd.DataFrame({'timesteps': x, 'y': y})
    df.to_csv(f'{layout_name}_{training_type}_avg_reward_curve.csv', index=False)

    # To visualize the agent:
    # python overcookedgym/overcooked-flask/app.py --modelpath_p0 ../logs/rl_model_500000_steps --modelpath_p1 ../logs/rl_model_50000_steps --layout_name simple


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains self-play agent on overcooked with checkpointing')
    parser.add_argument('--n_steps', help='the number of steps to train for', type=int, default=500000)
    parser.add_argument('--layout_name', help='the name of the layout to train on', type=str, default='forced_coordination')
    parser.add_argument('--training_type', help='the type of training to do', type=str, default='rl')
    parser.add_argument('--n_parallel_envs', help='the number of parallel environments to use', type=int, default=1)
    parser.add_argument('--save_directory', help='save directory to save the results', type=str)
    parser.add_argument('--prior_tree_file', help='filepath for the initial tree policy to optimize', type=str)
    parser.add_argument('--intent_model_file', help='filepath for the intent model', type=str)
    parser.add_argument('--teammate_paths', help='directory for teammate files', type=str,
                        default=os.path.join('data',
                                             'forced_coordination',
                                             'raw_feats_and_high_level_actions',
                                             'self_play_training_models'))
    # os.path.join('data', layout_name, teammate_subidentifier, 'self_play_training_models')
    args = parser.parse_args()
    main(n_steps=args.n_steps,
         layout_name=args.layout_name,
         training_type=args.training_type,
         save_dir=args.save_directory,
         n_parallel_envs=args.n_parallel_envs,
         intent_model_file=args.intent_model_file,
         prior_tree_file=args.prior_tree_file,
         teammate_paths=args.teammate_paths)
