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
from ipm.overcooked.overcooked_multi import OvercookedSelfPlayEnv, OvercookedRoundRobinEnv, \
    OvercookedPlayWithFixedPartner
from stable_baselines3.common.monitor import Monitor
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import shutil
from stable_baselines3.common.results_plotter import load_results, ts2xy


class CheckpointCallbackWithRew(CheckpointCallback):
    def __init__(self, n_steps, save_freq, save_path, name_prefix, save_replay_buffer,
                 initial_model_path, medium_model_path, final_model_path, save_model, verbose, reward_threshold=0.0):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer)
        self.initial_model_path = initial_model_path
        self.medium_model_path = medium_model_path
        self.final_model_path = final_model_path
        self.n_steps = n_steps
        self.best_mean_reward = -np.inf
        self.all_rewards = []
        self.all_steps = []
        self.all_save_paths = []
        self.verbose = verbose
        self.save_model = save_model
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        super()._on_step()
        if self.n_calls % self.save_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.save_path), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                model_path = self.save_path + '/' + self.name_prefix + "_" + str(self.num_timesteps) + "_steps" + ".zip"
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.save_model and self.best_mean_reward > self.reward_threshold:
                        if self.verbose > 0:
                            print(f"Saving new best model to {model_path} with mean reward {mean_reward:.2f}")
                        self.model.save(self.final_model_path)
                self.all_rewards.append(mean_reward)
                self.all_steps.append(self.n_calls)
                self.all_save_paths.append(model_path)
            if self.n_calls == self.n_steps and self.save_model and self.best_mean_reward > self.reward_threshold:
                # save initial model
                shutil.copy(self.all_save_paths[0], self.initial_model_path)

                # save second best model
                def find_closest_idx(arr, val):
                    idx = np.abs(arr - val).argmin()
                    val = arr[idx]
                    return idx, val

                second_best_reward_idx, reward = find_closest_idx(np.array(self.all_rewards),
                                                                  self.best_mean_reward * 0.5)
                if self.verbose > 0:
                    print(f"Saving medium model to {self.medium_model_path} with mean reward {reward:.2f}")
                shutil.copy(self.all_save_paths[second_best_reward_idx], self.medium_model_path)

            if self.n_calls == self.n_steps:
                # matplotlib the reward curve
                # x is the number of timesteps in increments of self.save_freq
                x = self.all_steps
                y = self.all_rewards
                # x, y = ts2xy(load_results(self.save_path), "timesteps")
                plt.clf()
                plt.plot(x, y)
                plt.grid()
                plt.xlabel('Timesteps')
                plt.ylabel('Avg. Reward')
                plt.title('Reward Curve')
                plt.savefig(self.save_path + '/reward_curve.png')

        return True


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def main(n_steps, layout_name, training_type,
         agent_type, n_parallel_envs=1, traj_directory=None,
         reduce_state_space_ego=False, reduce_state_space_teammate=False,
         high_level_actions_ego=False, high_level_actions_teammate=False):
    n_agents = 1
    checkpoint_freq = n_steps // 100
    # layouts of interest: 'forced_coordination'
    # 'counter_circuit', 'counter_circuit_tomato'
    save_models = False
    ego_idx = 0
    alt_idx = (ego_idx + 1) % 2

    all_rewards_across_seeds = []
    all_steps = []

    def get_subidentifier(reduce_state_space, high_level_actions):
        if not reduce_state_space and not high_level_actions:
            subidentifier = 'raw_feats_and_raw_actions'
        elif not reduce_state_space and high_level_actions:
            subidentifier = 'raw_feats_and_high_level_actions'
        elif reduce_state_space and not high_level_actions:
            subidentifier = 'reduced_feats_and_raw_actions'
        elif reduce_state_space and high_level_actions:
            subidentifier = 'reduced_feats_and_high_level_actions'
        else:
            raise "Invalid combination of reduce_state_space and high_level_actions"
        return subidentifier

    ego_subidentifier = get_subidentifier(reduce_state_space_ego, high_level_actions_ego)
    teammate_subidentifier = get_subidentifier(reduce_state_space_teammate, high_level_actions_teammate)

    teammate_paths = os.path.join('data', layout_name, teammate_subidentifier, 'self_play_training_models')

    for i in tqdm(range(n_agents)):

        seed = i
        if training_type == 'round_robin':
            env = OvercookedRoundRobinEnv(teammate_locations=teammate_paths, layout_name=layout_name, seed_num=i,
                                          ego_idx=ego_idx,
                                          reduced_state_space_ego=reduce_state_space_ego,
                                          reduced_state_space_alt=reduce_state_space_teammate,
                                          use_skills_ego=high_level_actions_ego,
                                          use_skills_alt=high_level_actions_teammate)
        elif training_type == 'self_play':
            env = OvercookedSelfPlayEnv(layout_name=layout_name + '_demonstrations', seed_num=i,
                                        reduced_state_space_ego=reduce_state_space_ego,
                                        reduced_state_space_alt=reduce_state_space_ego,
                                        use_skills_ego=high_level_actions_ego,
                                        use_skills_alt=high_level_actions_ego)
        elif training_type == 'human_bc_teammate':
            assert traj_directory is not None
            behavioral_model, bc_partner = get_human_bc_partner(traj_directory=traj_directory, layout_name=layout_name,
                                                                bc_agent_idx=alt_idx, get_human_policy_estimator=True)
            env = OvercookedPlayWithFixedPartner(partner=bc_partner, layout_name=layout_name, seed_num=i,
                                                 ego_idx=ego_idx,
                                                 behavioral_model=behavioral_model,
                                                 reduced_state_space_ego=reduce_state_space_ego,
                                                 reduced_state_space_alt=reduce_state_space_teammate,
                                                 use_skills_ego=high_level_actions_ego,
                                                 use_skills_alt=high_level_actions_teammate)

        initial_model_path = os.path.join('data', layout_name, ego_subidentifier, training_type + '_training_models',
                                          'seed_' + str(seed), 'initial_model.zip')
        medium_model_path = os.path.join('data', layout_name, ego_subidentifier, training_type + '_training_models',
                                         'seed_' + str(seed), 'medium_model.zip')
        final_model_path = os.path.join('data', layout_name, ego_subidentifier, training_type + '_training_models',
                                        'seed_' + str(seed), 'final_model.zip')

        save_dir = os.path.join('data', 'ppo_' + training_type, ego_subidentifier, 'seed_{}'.format(seed))

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        checkpoint_callback = CheckpointCallbackWithRew(
            n_steps=n_steps,
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

        if agent_type == 'idct':
            input_dim = get_obs_shape(env.observation_space)[0]
            output_dim = env.action_space.n
            model = IDCT(input_dim=input_dim,
                         output_dim=output_dim,
                         hard_node=False,
                         device='cuda',
                         argmax_tau=1.0,
                         use_individual_alpha=True,
                         use_gumbel_softmax=False,
                         alg_type='ppo',
                         weights=None,
                         comparators=None,
                         alpha=None,
                         fixed_idct=False,
                         leaves=8)

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
        elif agent_type == 'nn':
            agent = PPO('MlpPolicy', env, verbose=0, seed=seed)
        elif agent_type == 'ga':
            teammate_paths = os.path.join('data', layout_name, 'self_play_training_models')
            optimizer = GA_DT_Optimizer(initial_depth=4, max_depth=5, env=env, initial_population=teammate_paths)
            optimizer.run()
            best_genes = optimizer.best_solution
        else:
            raise ValueError('agent_type must be either "idct" or "nn"')

        if agent_type == 'nn' or agent_type == 'idct':
            print(f'Agent {i} training...')
            agent.learn(total_timesteps=n_steps, callback=checkpoint_callback)
            all_rewards_across_seeds.append(checkpoint_callback.all_rewards)
            all_steps = checkpoint_callback.all_steps
            print(f'Finished training agent {seed} with best average reward of {checkpoint_callback.best_mean_reward}')
        # To visualize the agent:
        # python overcookedgym/overcooked-flask/app.py --modelpath_p0 ../logs/rl_model_500000_steps --modelpath_p1 ../logs/rl_model_50000_steps --layout_name simple
    if agent_type == 'nn' or agent_type == 'idct':
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
        plt.savefig(f'{layout_name}_{training_type}_{agent_type}_avg_reward_curve.png')

        plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)
        plt.savefig(f'{layout_name}_{training_type}_{agent_type}_avg_reward_curve_with_var.png')

        print('Finished training all agents')

        # also save x and y to csv
        df = pd.DataFrame({'timesteps': x, 'y': y})
        df.to_csv(f'{layout_name}_{training_type}_{agent_type}_avg_reward_curve.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains self-play agent on overcooked with checkpointing')
    parser.add_argument('--n_steps', help='the number of steps to train for', type=int, default=500000)
    parser.add_argument('--layout_name', help='the name of the layout to train on', type=str,
                        default='forced_coordination')
    parser.add_argument('--training_type', help='the type of training to do', type=str, default='round_robin')
    parser.add_argument('--agent_type', help='the type of agent to train', type=str, default='idct')
    parser.add_argument('--n_parallel_envs', help='the number of parallel environments to use', type=int, default=1)
    # trajectories is optional (required for human bcp training)
    parser.add_argument('--traj_directory', help='the directory of trajectories to use for human bc', type=str,
                        default=None)
    parser.add_argument('--reduce_state_space_ego', help='whether to reduce the state space for the ego agent',
                        action='store_true', default=True)
    parser.add_argument('--reduce_state_space_teammate',
                        help='whether to reduce the state space for the teammate agent', action='store_true',
                        default=True)
    parser.add_argument('--high_level_actions_ego', help='whether to use high level actions for the ego agent',
                        action='store_true', default=True)
    parser.add_argument('--high_level_actions_teammate',
                        help='whether to use high level actions for the teammate agent', action='store_true',
                        default=True)
    args = parser.parse_args()
    main(n_steps=args.n_steps, layout_name=args.layout_name, traj_directory=args.traj_directory,
         training_type=args.training_type, agent_type=args.agent_type, n_parallel_envs=args.n_parallel_envs,
         reduce_state_space_ego=args.reduce_state_space_ego,
         reduce_state_space_teammate=args.reduce_state_space_teammate,
         high_level_actions_ego=args.high_level_actions_ego,
         high_level_actions_teammate=args.high_level_actions_teammate)
