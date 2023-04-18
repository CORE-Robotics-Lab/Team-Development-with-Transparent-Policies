import argparse
import math
import os
import pygame
import numpy as np
import cv2
import time
import torch
from typing import Callable
import pickle5 as pickle
from pygame import gfxdraw
from ipm.models.idct import IDCT
import torch.nn as nn

from ipm.gui.nasa_tlx import run_gui
from ipm.gui.pages import GUIPageCenterText, TreeCreationPage, EnvPage, EnvPerformancePage, OvercookedPage, \
    EnvRewardModificationPage, DecisionTreeCreationPage, GUIPageWithTwoTreeChoices, GUIPageWithImage, \
    GUIPageWithTextAndURL, GUIPageWithSingleTree
from ipm.gui.policy_utils import get_idct, finetune_model
from ipm.models.bc_agent import get_pretrained_teammate_finetuned_with_bc, StayAgent
from ipm.models.intent_model import get_pretrained_intent_model
from ipm.overcooked.overcooked_envs import OvercookedRoundRobinEnv, OvercookedPlayWithFixedPartner
from ipm.models.decision_tree import DecisionTree, sparse_ddt_to_decision_tree
from ipm.overcooked.overcooked_envs import OvercookedJointEnvironment
from stable_baselines3 import PPO


class EnvWrapper:
    def __init__(self, layout, idct_filepath):
        # wrapping this up in a class so that we can easily change the reward function
        # this acts like a pointer
        self.multipliers = [1, 1, 1]
        teammate_paths = os.path.join('data', layout, 'self_play_training_models')
        self.ego_idx = 0
        self.alt_idx = 1
        self.layout = layout

        # CODE BELOW NEEDED TO CONFIGURE RECIPES INITIALLY
        dummy_env = OvercookedJointEnvironment(layout_name=layout)


        self.bc_partner = get_pretrained_teammate_finetuned_with_bc(layout, self.alt_idx)
        intent_model_path = os.path.join('data', 'intent_models', layout + '.pt')
        self.intent_model = get_pretrained_intent_model(layout, intent_model_file=intent_model_path)
        self.rewards = []
        self.train_env = None  # for optimization conditions we want to use this

        self.team_env = OvercookedPlayWithFixedPartner(partner=self.bc_partner, layout_name=layout,
                                                       behavioral_model=self.intent_model,
                                                       reduced_state_space_ego=True, reduced_state_space_alt=False,
                                                       use_skills_ego=True, use_skills_alt=True,
                                                       failed_skill_rew=0)
        self.save_chosen_as_prior = False
        self.env = self.team_env  # need to change to train env
        self.initial_policy_path = idct_filepath

        self.idx_to_skill_ego = self.team_env.idx_to_skill_ego

        self.idx_to_skill_alt = self.team_env.idx_to_skill_alt
        self.idx_to_skill_strings = [
                                     ['stand_still'],
                                     ['get_onion_from_dispenser'], ['pickup_onion_from_counter'],
            ['get_dish_from_dispenser'], ['pickup_dish_from_counter'],
                                      ['get_soup_from_pot'], ['pickup_soup_from_counter'],
                                      ['serve_at_dispensary'],
                                      ['bring_to_closest_pot'], ['place_on_closest_counter']]

        def load_idct_from_torch(filepath):
            model = torch.load(filepath)['alt_state_dict']
            layers = model['action_net.layers']
            comparators = model['action_net.comparators']
            alpha = model['action_net.alpha']
            input_dim = self.env.observation_space.shape[0]
            output_dim = self.env.n_actions_ego
            # assuming an symmetric tree here
            n_nodes, n_feats = layers.shape
            assert n_feats == input_dim

            action_mus = model['action_net.action_mus']
            n_leaves, _ = action_mus.shape
            idct = IDCT(input_dim=input_dim, output_dim=output_dim, leaves=n_leaves, hard_node=False, device='cuda',
                        argmax_tau=1.0,
                        alpha=alpha, comparators=comparators, weights=layers)
            idct.action_mus = nn.Parameter(action_mus, requires_grad=True)
            idct.update_leaf_init_information()
            return idct

        def load_dt_from_idct(filepath):
            idct = load_idct_from_torch(filepath)
            dt, tree_info = sparse_ddt_to_decision_tree(idct, self.env)
            return dt

        def load_pols(filepath):
            idct = load_idct_from_torch(filepath)
            return idct

        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        idct = load_pols(idct_filepath)
        dt = load_dt_from_idct(idct_filepath)
        model = PPO("MlpPolicy", self.team_env)
        weights = torch.load(idct_filepath)
        model.policy.load_state_dict(weights['ego_state_dict'])

        self.team_env = OvercookedPlayWithFixedPartner(partner=model, layout_name=layout,
                                                    reduced_state_space_ego=True, reduced_state_space_alt=True,
                                                    use_skills_ego=True, use_skills_alt=True, failed_skill_rew=0)

        self.env.reset()
        dummy_env.reset()
        done = False
        p0 = model
        using_idct = True
        if using_idct:
            p1 = idct
        else:
            p1 = dt
        reduced_observations_p0 = []
        reduced_observations_p1 = []
        states_p0 = []
        states_p1 = []
        actions_p0 = []
        actions_p1 = []
        rewards = 0
        while not done:
            obs = dummy_env.mdp.featurize_state_reduced(dummy_env.state)
            obs0 = dummy_env.add_intent(obs[0], obs[1], 0)
            obs1 = dummy_env.add_intent(obs[1], obs[0], 1)
            # get the actions
            action_p0, _ = p0.predict(torch.tensor(obs0))
            if using_idct:
                action_p1 = p1.predict_proba(obs1)
            else:
                output_class = p1.predict(obs1)
                action_p1 = np.random.choice(output_class.indices, p=[output_class.values[0], output_class.values[1], 1-output_class.values[1]- output_class.values[0]])
            print(dummy_env.base_env)
            print(self.idx_to_skill_strings[action_p0], self.idx_to_skill_strings[action_p1])

            # ego_action, skill_rew_ego = self.idx_to_skill_ego[action_p0](agent_idx=0)
            # alt_action, skill_rew_alt = self.idx_to_skill_alt[action_p1](agent_idx=1)
            reduced_observations_p0.append(obs[0])
            reduced_observations_p1.append(obs[1])
            states_p0.append(self.env.state)
            states_p1.append(self.env.state)
            actions_p0.append(action_p0)
            actions_p1.append(action_p1)

            # take the actions
            obs, reward, done, info = dummy_env.step((action_p0, action_p1))
            dummy_env.timestep = dummy_env.state.timestep
            dummy_env.prev_macro_action = [action_p0, action_p1]
            rewards += reward[0]
        print('tot reward is ', rewards)
        self.current_policy = load_dt_from_idct(self.initial_policy_path)
        self.save_chosen_as_prior = False

    def initialize_env(self):
        # we keep track of the reward function that may change
        self.team_env.set_env(placing_in_pot_multiplier=self.multipliers[0],
                              dish_pickup_multiplier=self.multipliers[1],
                              soup_pickup_multiplier=self.multipliers[2])
class SettingsWrapper:
    def __init__(self):
        self.zoom = 1
        self.old_zoom = 1
        self.max_zoom = 3
        self.min_zoom = 1
        self.width, self.height = 1920, 1080
        self.offset_x, self.offset_y = 0, 0
        self.absolute_x, self.absolute_y = self.width // 2, self.height // 2
        self.options_menus_per_domain = {-1: [], 0: [], 1: [], 2: []}

    def check_if_options_menu_open(self, domain_idx) -> (bool, int):
        for i, menu_open in enumerate(self.options_menus_per_domain[domain_idx]):
            if menu_open:
                return True, i
        return False, -1

    def zoom_out(self):
        self.old_zoom = self.zoom
        self.zoom = max(self.zoom - 0.1, self.min_zoom)
        assert self.max_zoom >= self.zoom >= self.min_zoom

    def zoom_in(self):
        self.old_zoom = self.zoom
        self.zoom = min(self.zoom + 0.1, self.max_zoom)
        assert self.max_zoom >= self.zoom >= self.min_zoom

class IDCTVisualizer:
    def __init__(self, idct_filepath: str):
        pygame.init()
        self.pages = []
        self.current_page = 0
        self.settings = SettingsWrapper()
        # self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA | pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((self.settings.width, self.settings.height), pygame.SRCALPHA)# | pygame.FULLSCREEN)
                                              #pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen.fill('white')

        layout = 'forced_coordination'
        self.env_wrapper = EnvWrapper(layout=layout, idct_filepath=idct_filepath)
        self.tree_page = DecisionTreeCreationPage(self.env_wrapper, layout, -1, self.settings,
                                                  screen=self.screen,
                                                  X=self.settings.width, Y=self.settings.height,
                                                  bottom_left_button=False, bottom_right_button=True,
                                                  bottom_left_fn=None, bottom_right_fn=None)
        self.pages.append(self.tree_page)



    def launch(self):
        pygame.init()
        clock = pygame.time.Clock()
        self.is_running = True
        self.pages[0].show()
        pygame.display.flip()

        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                    break
                self.is_running = self.pages[self.current_page].process_event(event)
                if self.is_running is False:
                    break
            self.pages[self.current_page].process_standby()
            pygame.display.update()
            clock.tick(30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize an IDCT.')
    parser.add_argument('--idct_filepath', help='Filepath for idct', type=str, required=True)
    args = parser.parse_args()
    experiment = IDCTVisualizer(args.idct_filepath)
    experiment.launch()