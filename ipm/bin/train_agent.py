"""
This is a simple example training script.
"""
import argparse
import json
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.torch_layers import FlattenExtractor
from ipm.algos import ddt_ppo_policy
from tqdm import tqdm

from ipm.algos.genetic_algorithm import GA_DT_Optimizer
from ipm.models.idct import IDCT
from ipm.overcooked.overcooked import OvercookedSelfPlayEnv, OvercookedRoundRobinEnv, OvercookedPlayWithFixedPartner
from stable_baselines3.common.monitor import Monitor
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import shutil
from stable_baselines3.common.results_plotter import load_results, ts2xy

class CheckpointCallbackWithRew(CheckpointCallback):
    def __init__(self, n_steps, save_freq, save_path, name_prefix, save_replay_buffer,
                 initial_model_path, medium_model_path, final_model_path, save_model, verbose, reward_threshold=200.0):
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

                second_best_reward_idx, reward = find_closest_idx(np.array(self.all_rewards), self.best_mean_reward * 0.5)
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

class BCAgent:
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions
        # by default, use sklearn random forest
        # self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
        self.model = DecisionTreeClassifier(max_depth=10, random_state=0)
        # self.model.fit(self.observations, self.actions)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.observations, self.actions, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        # check validation accuracy
        print("Validation accuracy for BC model: ", self.model.score(self.X_test, self.y_test))

        accuracy_threshold = 0.8
        if self.model.score(self.X_test, self.y_test) < accuracy_threshold:
            raise ValueError("BC model accuracy is too low! Please collect more data or use a different model.")

        # train on all the data
        self.model.fit(self.observations, self.actions)


    def predict(self, observation):
        _states = None
        return self.model.predict(observation.reshape(1, -1))[0], _states


def get_human_bc_partner(traj_directory, layout_name, alt_idx):
    # load each csv file into a dataframe
    dfs = []
    episode_num = 0
    for filename in os.listdir(traj_directory):
        if filename.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(traj_directory, filename)))
            dfs[-1]['episode_num'] = episode_num
            episode_num += 1
    # aggregate all dataframes into one
    df = pd.concat(dfs, ignore_index=True)
    # convert states to observations

    # we simply want to use the state -> observation fn from this env
    # env = OvercookedSelfPlayEnv(layout_name=layout_name, seed_num=0,
    #                             reduced_state_space_ego=False,
    #                             reduced_state_space_alt=False)
    # states = df['state']
    # for i in range(len(states)):
    #     state = json.loads(states[i])
    #     observations.append(env.featurize_fn(state))

    # only get rows where alt_idx is the alt agent
    df = df[df['agent_idx'] == alt_idx]

    # string obs to numpy array
    observations = []
    for obs_str in df['obs'].values:
        obs_str = obs_str.replace('\n', '')
        observations.append(np.fromstring(obs_str[1:-1], dtype=float, sep=' '))

    actions = df['action'].values
    return BCAgent(observations, actions)

def main(n_steps, training_type='self_play', traj_directory=None):
    n_agents = 32
    checkpoint_freq = n_steps // 100
    # layouts of interest: 'forced_coordination'
    # 'counter_circuit', 'counter_circuit_tomato'
    layout_name = 'forced_coordination'
    training_type = 'human_bc_teammate'
    agent_type = 'nn'
    save_models = True
    ego_idx = 0
    alt_idx = (ego_idx + 1) % 2

    all_rewards_across_seeds = []
    all_steps = []

    for i in tqdm(range(n_agents)):

        seed = i

        if agent_type == 'nn':
            reduce_state_space = False
        else:
            reduce_state_space = True

        if training_type == 'round_robin':
            teammate_paths = os.path.join('data', layout_name, 'self_play_training_models')
            env = OvercookedRoundRobinEnv(teammate_locations=teammate_paths, layout_name=layout_name, seed_num=i, ego_idx=ego_idx,
                                          reduced_state_space_ego=reduce_state_space, reduced_state_space_alt=False)
        elif training_type == 'self_play':
            env = OvercookedSelfPlayEnv(layout_name=layout_name, seed_num=i,
                                        reduced_state_space_ego=reduce_state_space,
                                        reduced_state_space_alt=reduce_state_space)
        elif training_type == 'human_bc_teammate':
            assert traj_directory is not None
            bc_partner = get_human_bc_partner(traj_directory=traj_directory, layout_name=layout_name,alt_idx=alt_idx)
            env = OvercookedPlayWithFixedPartner(partner=bc_partner, layout_name=layout_name, seed_num=i,
                                                 reduced_state_space_ego=reduce_state_space,
                                                 reduced_state_space_alt=False,
                                                 use_skills_ego=False,
                                                 use_skills_alt=False)

        initial_model_path = os.path.join('data', layout_name, training_type + '_training_models', 'seed_' + str(seed), 'initial_model.zip')
        medium_model_path = os.path.join('data', layout_name, training_type + '_training_models', 'seed_' + str(seed), 'medium_model.zip')
        final_model_path = os.path.join('data', layout_name, training_type + '_training_models', 'seed_' + str(seed), 'final_model.zip')

        save_dir = os.path.join('data', 'ppo_' + training_type, 'seed_{}'.format(seed))

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

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
                'alpha': model.alpha,
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

            agent = PPO("DDT_PPOPolicy", env,
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
            optimizer = GA_DT_Optimizer(initial_depth=3, max_depth=5, env=env, initial_population=teammate_paths)
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
    parser.add_argument('--trajectories', help='the directory of trajectories to use for human bc', type=str, default=None)
    args = parser.parse_args()
    main(n_steps=args.n_steps, traj_directory=args.trajectories)
