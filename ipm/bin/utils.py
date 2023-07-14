from stable_baselines3.common.callbacks import CheckpointCallback
import shutil
from stable_baselines3.common.results_plotter import load_results, ts2xy
from matplotlib import pyplot as plt
import numpy as np
import gym
from stable_baselines3.common.utils import set_random_seed
import pygame
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState


def visualize_state(visualizer: StateVisualizer,
                    screen: pygame.display,
                    env: gym.Env,
                    state: OvercookedState,
                    width: int,
                    height: int):
    """
    Visualize the current state of the game in pygame

    :param visualizer: visualizer for OvercookedStates
    :param screen: pygame screen
    :param env: the environment (which contains the layout terrain)
    :param state: current OvercookedState
    :param width: width of the pygame screen
    :param height: height of the pygame screen
    """
    # wait 0.2 seconds
    pygame.time.wait(200)
    screen.fill((0, 0, 0))
    state_visualized_surf = visualizer.render_state(state=state, grid=env.base_env.mdp.terrain_mtx)
    screen.blit(pygame.transform.scale(state_visualized_surf, (width, height)), (0, 0))
    pygame.display.flip()


def play_episode_together_get_states(env, policy_a, policy_b) -> []:
    """
    Play an episode of the game with two agents

    :param env: joint environment
    :param policy_a: policy of the first agent
    :param policy_b: policy of the second agent
    :return: the states
    """

    states = []
    done = False
    (obs_a, obs_b) = env.reset(use_reduced=True)
    while not done:
        action_a = policy_a.predict(obs_a)
        action_b = policy_b.predict(obs_b)
        states.append(env.state)
        (obs_a, obs_b), (rew_a, rew_b), done, info = env.step(macro_joint_action=(action_a, action_b), use_reduced=True)
        env.prev_macro_action = [action_a, action_b]
    return states


def play_episode_together(env, policy_a, policy_b, render=False) -> float:
    """
    Play an episode of the game with two agents

    :param env: joint environment
    :param policy_a: policy of the first agent
    :param policy_b: policy of the second agent
    :return: total reward of the episode
    """

    idx_to_skill_strings = [
        ['stand_still'],
        ['get_onion_from_dispenser'], ['pickup_onion_from_counter'],
        ['get_dish_from_dispenser'], ['pickup_dish_from_counter'],
        ['get_soup_from_pot'], ['pickup_soup_from_counter'],
        ['serve_at_dispensary'],
        ['bring_to_closest_pot'], ['place_on_closest_counter']]

    if render:
        pygame.init()
        width = 800
        height = 600
        screen = pygame.display.set_mode((width, height))
        visualizer = StateVisualizer()

    done = False
    (obs_a, obs_b) = env.reset(use_reduced=True)
    total_reward = 0
    while not done:
        if render:
            visualize_state(visualizer=visualizer, screen=screen, env=env, state=env.state, width=width, height=height)
        action_a = policy_a.predict(obs_a)
        action_b = policy_b.predict(obs_b)
        # print(env.base_env)
        # print('Reward so far:', total_reward)
        # print('Action for human policy: ', idx_to_skill_strings[action_a])
        # print('Action for robot policy: ', idx_to_skill_strings[action_b])
        (obs_a, obs_b), (rew_a, rew_b), done, info = env.step(macro_joint_action=(action_a, action_b), use_reduced=True)
        env.prev_macro_action = [action_a, action_b]
        total_reward += rew_a + rew_b
    return total_reward


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
                        import copy
                        self.final_model_weights = copy.deepcopy(self.model.policy.state_dict())
                self.all_rewards.append(mean_reward)
                self.all_steps.append(self.n_calls)
                self.all_save_paths.append(model_path)
                import copy
                self.updated_model_weights = copy.deepcopy(self.model.policy.state_dict())
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

            # if self.n_calls == self.n_steps:
            #     # matplotlib the reward curve
            #     # x is the number of timesteps in increments of self.save_freq
            #     x = self.all_steps
            #     y = self.all_rewards
            #     # x, y = ts2xy(load_results(self.save_path), "timesteps")
            #     plt.clf()
            #     plt.plot(x, y)
            #     plt.grid()
            #     plt.xlabel('Timesteps')
            #     plt.ylabel('Avg. Reward')
            #     plt.title('Reward Curve')
            #     plt.savefig(self.save_path + '/reward_curve.png')

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
