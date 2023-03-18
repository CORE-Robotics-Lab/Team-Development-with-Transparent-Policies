from stable_baselines3.common.callbacks import CheckpointCallback
import shutil
from stable_baselines3.common.results_plotter import load_results, ts2xy
from matplotlib import pyplot as plt
import numpy as np
import gym
from stable_baselines3.common.utils import set_random_seed


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
