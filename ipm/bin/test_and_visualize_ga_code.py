import os

import numpy as np
import pygame
import torch
from ipm.gui.experiment_gui_utils import SettingsWrapper, get_next_user_id
from ipm.models.bc_agent import StayAgent
from ipm.models.decision_tree import sparse_ddt_to_decision_tree
from ipm.overcooked.overcooked_envs import OvercookedPlayWithFixedPartner, OvercookedJointRecorderEnvironment
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from models.human_model import HumanModel
from models.robot_model import RobotModel
from stable_baselines3 import PPO
from tqdm import tqdm


def visualize_state(visualizer, screen, env, state, width, height):
    # wait 0.2 seconds
    pygame.time.wait(200)
    screen.fill((0, 0, 0))
    state_visualized_surf = visualizer.render_state(state=state, grid=env.base_env.mdp.terrain_mtx)
    screen.blit(pygame.transform.scale(state_visualized_surf, (width, height)), (0, 0))
    pygame.display.flip()

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


class EnvWrapper:
    def __init__(self, layout, data_folder):
        # wrapping this up in a class so that we can easily change the reward function
        # this acts like a pointer
        self.multipliers = [1, 1, 1]
        self.ego_idx = 0
        self.alt_idx = 1
        self.layout = layout
        self.data_folder = data_folder
        self.rewards = []
        self.save_chosen_as_prior = False
        self.latest_save_file = None

        dummy_env = OvercookedPlayWithFixedPartner(partner=StayAgent(), layout_name=layout,
                                                   behavioral_model='dummy',
                                                   reduced_state_space_ego=True, reduced_state_space_alt=True,
                                                   use_skills_ego=True, use_skills_alt=True)

        self.initial_policy_path = os.path.join('data', 'prior_tree_policies', layout + '.tar')
        intent_model_path = os.path.join('data', 'intent_models', layout + '.pt')

        model = PPO("MlpPolicy", dummy_env)
        weights = torch.load(self.initial_policy_path)
        model.policy.load_state_dict(weights['ego_state_dict'])
        human_ppo_policy = model.policy
        self.human_policy = HumanModel(layout, human_ppo_policy)

        input_dim = dummy_env.observation_space.shape[0]
        output_dim = dummy_env.n_actions_alt

        self.robot_policy = RobotModel(layout=layout,
                                       idct_policy_filepath=self.initial_policy_path,
                                       human_policy=self.human_policy,
                                       intent_model_filepath=intent_model_path,
                                       input_dim=input_dim,
                                       output_dim=output_dim)

        joint_environment = OvercookedJointRecorderEnvironment(behavioral_model=self.robot_policy.intent_model,
                                                               layout_name=layout, seed_num=0,
                                                               ego_idx=0,
                                                               failed_skill_rew=0,
                                                               reduced_state_space_ego=True,
                                                               reduced_state_space_alt=True,
                                                               use_skills_ego=True,
                                                               use_skills_alt=True,
                                                               use_true_intent_ego=True,
                                                               use_true_intent_alt=False,
                                                               double_cook_times=False)

        n_samples = 30

        all_rewards_initial = []
        for i in tqdm(range(n_samples)):
            reward = play_episode_together(joint_environment, self.human_policy, self.robot_policy, render=False)
            all_rewards_initial.append(reward)

        data_file = 'data/iteration_0.tar'
        self.human_policy.translate_recent_data_to_labels(recent_data_loc=data_file)
        self.human_policy.finetune_human_ppo_policy()

        all_rewards_finetuned_human = []
        for i in tqdm(range(n_samples)):
            reward = play_episode_together(joint_environment, self.human_policy, self.robot_policy, render=False)
            all_rewards_finetuned_human.append(reward)

        self.current_policy, tree_info = sparse_ddt_to_decision_tree(self.robot_policy.robot_idct_policy,
                                                                     self.robot_policy.env)
        self.intent_model = self.robot_policy.intent_model

        data_file = 'data/iteration_0.tar'
        self.robot_policy.translate_recent_data_to_labels(recent_data_loc=data_file)
        self.robot_policy.finetune_intent_model()

        # all_rewards_finetuned_intent = []
        # for i in range(n_samples):
        #     reward = play_episode_together(joint_environment, self.human_policy, self.robot_policy, render=False)
        #     all_rewards_finetuned_intent.append(reward)
        #
        self.robot_policy.finetune_robot_idct_policy()

        all_rewards_finetuned_ga_rl= []
        for i in range(n_samples):
            reward = play_episode_together(joint_environment, self.human_policy, self.robot_policy, render=False)
            all_rewards_finetuned_ga_rl.append(reward)

        print('Average reward for initial policy: ', round(np.mean(all_rewards_initial), 2))
        print('Average reward for fine-tuned human policy: ', round(np.mean(all_rewards_finetuned_human), 2))
        # print('Average reward for fine-tuned intent model: ', np.mean(all_rewards_finetuned_intent))
        print('Average reward for fine-tuned GA: ', round(np.mean(all_rewards_finetuned_ga_rl), 2))


    def initialize_env(self):
        # we keep track of the reward function that may change
        self.robot_policy.env.set_env(placing_in_pot_multiplier=self.multipliers[0],
                                      dish_pickup_multiplier=self.multipliers[1],
                                      soup_pickup_multiplier=self.multipliers[2])


if __name__ == '__main__':
    user_id = get_next_user_id()
    conditions = ['human_modifies_tree',
                  'optimization',
                  'optimization_while_modifying_reward',
                  'no_modification_bb',
                  'no_modification_interpretable']
    condition = 'human_modifies_tree'
    condition_num = conditions.index(condition) + 1
    data_folder = os.path.join('data',
                               'experiments',
                               condition,
                               'user_' + str(user_id))

    domain_names = ['tutorial', 'forced_coordination', 'two_rooms', 'two_rooms_narrow']
    domain_names = ['forced_coordination']
    for domain_name in domain_names:
        folder = os.path.join(data_folder, domain_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

    pygame.init()
    settings = SettingsWrapper()
    env_wrappers = [EnvWrapper(layout=layout, data_folder=data_folder) for layout in domain_names]
