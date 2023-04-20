import os

import pygame
import torch
from ipm.gui.experiment_gui_utils import SettingsWrapper, get_next_user_id
from ipm.models.bc_agent import StayAgent
from ipm.models.decision_tree import sparse_ddt_to_decision_tree
from ipm.overcooked.overcooked_envs import OvercookedPlayWithFixedPartner
from models.human_model import HumanModel
from models.robot_model import RobotModel
from stable_baselines3 import PPO
from ipm.algos import ddt_ppo_policy


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

        self.current_policy, tree_info = sparse_ddt_to_decision_tree(self.robot_policy.robot_idct_policy,
                                                                     self.robot_policy.env)
        self.intent_model = self.robot_policy.intent_model

        if self.layout == 'forced_coordination':
            data_file = 'data/iteration_0.tar'
            self.robot_policy.finetune_robot_idct_policy()
            self.robot_policy.translate_recent_data_to_labels(recent_data_loc=data_file)
            self.robot_policy.finetune_intent_model()
            self.human_policy.translate_recent_data_to_labels(recent_data_loc=data_file)
            self.human_policy.finetune_human_ppo_policy()

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
    for domain_name in domain_names:
        folder = os.path.join(data_folder, domain_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

    pygame.init()
    settings = SettingsWrapper()
    env_wrappers = [EnvWrapper(layout=layout, data_folder=data_folder) for layout in domain_names]
