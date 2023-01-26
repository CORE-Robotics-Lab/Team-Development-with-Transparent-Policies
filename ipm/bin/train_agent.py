"""
This is a simple example training script.
"""
import argparse
import os

from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.torch_layers import FlattenExtractor
from ipm.algos import ddt_ppo_policy
from tqdm import tqdm

from ipm.algos.genetic_algorithm import GA_DT_Optimizer
from ipm.models.idct import IDCT
from ipm.overcooked.overcooked import OvercookedSelfPlayEnv, OvercookedRoundRobinEnv
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
                    if self.verbose > 0:
                        print(f"Saving new best model to {model_path} with mean reward {mean_reward:.2f}")
                    if self.save_model and self.best_mean_reward > self.reward_threshold:
                        self.model.save(self.final_model_path)
                self.all_rewards.append(mean_reward)
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
        return True

def main(N_steps, training_type='self_play'):
    n_agents = 32
    checkpoint_freq = N_steps // 100
    # layouts of interest: 'cramped_room_tomato', 'cramped_room', 'asymmetric_advantages', 'asymmetric_advantages_tomato',
    # 'counter_circuit', 'counter_circuit_tomato'
    layout_name = 'forced_coordination_tomato'
    training_type = 'self_play'
    agent_type = 'nn'
    save_models = True

    for i in tqdm(range(n_agents)):

        seed = i

        if agent_type == 'nn':
            reduce_state_space = False
        else:
            reduce_state_space = True

        if training_type == 'round_robin':
            teammate_paths = os.path.join('data', layout_name, 'self_play_training_models')
            env = OvercookedRoundRobinEnv(teammate_locations=teammate_paths, layout_name=layout_name, seed_num=i, ego_idx=0,
                                          reduced_state_space_ego=reduce_state_space, reduced_state_space_alt=False)
        elif training_type == 'self_play':
            env = OvercookedSelfPlayEnv(layout_name=layout_name, seed_num=i,
                                        reduced_state_space_ego=reduce_state_space,
                                        reduced_state_space_alt=reduce_state_space)

        initial_model_path = os.path.join('data', layout_name, training_type + '_training_models', 'seed_' + str(seed), 'initial_model.zip')
        medium_model_path = os.path.join('data', layout_name, training_type + '_training_models', 'seed_' + str(seed), 'medium_model.zip')
        final_model_path = os.path.join('data', layout_name, training_type + '_training_models', 'seed_' + str(seed), 'final_model.zip')

        save_dir = os.path.join('data', 'ppo_' + training_type, 'seed_{}'.format(seed))

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        checkpoint_callback = CheckpointCallbackWithRew(
          n_steps = N_steps,
          save_freq=checkpoint_freq,
          save_path=save_dir,
          name_prefix="rl_model",
          save_replay_buffer=True,
          initial_model_path=initial_model_path,
          medium_model_path=medium_model_path,
          final_model_path=final_model_path,
          save_model=save_models,
          verbose=0
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
            optimizer = GA_DT_Optimizer(initial_depth=7, max_depth=10, env=env, initial_population=teammate_paths)
            optimizer.run()
            best_genes = optimizer.best_solution
        else:
            raise ValueError('agent_type must be either "idct" or "nn"')

        if agent_type == 'nn' or agent_type == 'idct':
            print(f'Agent {i} training...')
            agent.learn(total_timesteps=N_steps, callback=checkpoint_callback)
            print(f'Finished training agent {seed} with best average reward of {checkpoint_callback.best_mean_reward}')
        # To visualize the agent:
        # python overcookedgym/overcooked-flask/app.py --modelpath_p0 ../logs/rl_model_500000_steps --modelpath_p1 ../logs/rl_model_50000_steps --layout_name simple


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains self-play agent on overcooked with checkpointing')
    parser.add_argument('--n_steps', help='the number of steps to train for', type=int, default=500000)
    args = parser.parse_args()
    main(args.n_steps)
