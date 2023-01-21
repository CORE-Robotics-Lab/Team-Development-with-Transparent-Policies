"""
This is a simple example training script.
"""
import argparse
import os

from tqdm import tqdm

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
                 initial_model_path, medium_model_path, final_model_path, verbose):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer)
        self.initial_model_path = initial_model_path
        self.medium_model_path = medium_model_path
        self.final_model_path = final_model_path
        self.n_steps = n_steps
        self.best_mean_reward = -np.inf
        self.all_rewards = []
        self.all_save_paths = []
        self.verbose = verbose

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
                    self.model.save(self.final_model_path)
                self.all_rewards.append(mean_reward)
                self.all_save_paths.append(model_path)
            if self.n_calls == self.n_steps:
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

def main(N_steps, agent_type='self_play'):
    n_agents = 20
    checkpoint_freq = N_steps // 100
    # layouts of interest: 'cramped_room_tomato', 'cramped_room', 'asymmetric_advantages', 'asymmetric_advantages_tomato',
    # 'counter_circuit', 'counter_circuit_tomato'
    layout_name = 'forced_coordination_tomato'
    agent_type = 'self_play'
    agent_type = 'round_robin'

    for i in tqdm(range(101, 100 + n_agents)):

        seed = i

        if agent_type == 'round_robin':
            teammate_paths = os.path.join('data', layout_name, 'self_play_training_models')
            env = OvercookedRoundRobinEnv(teammate_locations=teammate_paths, layout_name=layout_name, seed_num=i,
                                          reduced_state_space_ego=True, reduced_state_space_alt=True)
        elif agent_type == 'self_play':
            env = OvercookedSelfPlayEnv(layout_name=layout_name, seed_num=i, reduced_state_space_ego=True, reduced_state_space_alt=True)

        initial_model_path = os.path.join('data', layout_name, agent_type + '_training_models', 'seed_' + str(seed), 'initial_model.zip')
        medium_model_path = os.path.join('data', layout_name, agent_type + '_training_models', 'seed_' + str(seed), 'medium_model.zip')
        final_model_path = os.path.join('data', layout_name, agent_type + '_training_models', 'seed_' + str(seed), 'final_model.zip')

        save_dir = os.path.join('data', 'ppo_' + agent_type, 'seed_{}'.format(seed))

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
          verbose=0
        )

        env = Monitor(env, "./" + save_dir + "/")
        agent = PPO('MlpPolicy', env, verbose=1, seed=seed)
        agent.learn(total_timesteps=N_steps, callback=checkpoint_callback)
        # To visualize the agent:
        # python overcookedgym/overcooked-flask/app.py --modelpath_p0 ../logs/rl_model_500000_steps --modelpath_p1 ../logs/rl_model_50000_steps --layout_name simple


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains self-play agent on overcooked with checkpointing')
    parser.add_argument('--n_steps', help='the number of steps to train for', type=int, default=500000)
    args = parser.parse_args()
    main(args.n_steps)
