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
from gui.tree_gui_utils import TreeInfo
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
from ipm.models.decision_tree import DecisionTree, sparse_ddt_to_decision_tree, decision_tree_to_ddt
from ipm.algos.genetic_algorithm import GA_DT_Structure_Optimizer

class EnvWrapper:
    def __init__(self, layout):
        # wrapping this up in a class so that we can easily change the reward function
        # this acts like a pointer
        self.multipliers = [1, 1, 1]
        teammate_paths = os.path.join('data', layout, 'self_play_training_models')
        self.ego_idx = 0
        self.alt_idx = 1
        self.layout = layout

        # CODE BELOW NEEDED TO CONFIGURE RECIPES INITIALLY
        dummy_env = OvercookedPlayWithFixedPartner(partner=StayAgent(), layout_name=layout,
                                                       reduced_state_space_ego=True, reduced_state_space_alt=False,
                                                       use_skills_ego=True, use_skills_alt=False, failed_skill_rew=0)

        self.bc_partner = get_pretrained_teammate_finetuned_with_bc(layout, self.alt_idx)
        self.intent_model = get_pretrained_intent_model(layout)
        self.rewards = []
        # TODO: reward shown on chosen page can be inaccurate if we go with the prior policy
        # this probably won't matter if we use human policy estimation to compute rewards for each tree
        self.train_env = None # for optimization conditions we want to use this

        self.team_env = OvercookedPlayWithFixedPartner(partner=self.bc_partner, layout_name=layout,
                                                       behavioral_model=self.intent_model,
                                                       reduced_state_space_ego=True, reduced_state_space_alt=False,
                                                       use_skills_ego=True, use_skills_alt=False, failed_skill_rew=0)

        input_dim = self.team_env.observation_space.shape[0]
        output_dim = self.team_env.n_actions_ego

        self.save_chosen_as_prior = False
        self.env = self.team_env # need to change to train env

        trajectories_file = 'data/11_trajs_tar'
        initial_depth = 1
        max_depth = 1
        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.n_actions_ego
        num_gens = 100
        seed = 1
        ga = GA_DT_Structure_Optimizer(trajectories_file=trajectories_file,
                                       initial_depth=initial_depth,
                                       max_depth=max_depth,
                                       n_vars=input_dim,
                                       n_actions=output_dim,
                                       num_gens=num_gens,
                                       seed=seed)
        ga.run()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_tree = ga.final_trees[0]
        #final_tree = ga.best_tree
        idct = decision_tree_to_ddt(tree=final_tree,
                                  input_dim=input_dim,
                                  output_dim=output_dim,
                                  device=device)
        self.decision_tree, self.tree_info = sparse_ddt_to_decision_tree(idct, self.env)
        self.save_chosen_as_prior = False

    def initialize_env(self):
        # we keep track of the reward function that may change
        self.team_env.set_env(self.multipliers[0], self.multipliers[1], self.multipliers[2])
        # self.train_env.set_env(self.multipliers[0], self.multipliers[1], self.multipliers[2])

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
    def __init__(self):
        pygame.init()
        self.pages = []
        self.current_page = 0
        self.settings = SettingsWrapper()
        # self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA | pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((self.settings.width, self.settings.height), pygame.SRCALPHA)# | pygame.FULLSCREEN)
                                              #pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen.fill('white')

        layout = 'forced_coordination'
        self.env_wrapper = EnvWrapper(layout=layout)
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
    # parser.add_argument('--idct_filepath', help='Filepath for idct', type=str, required=True)
    args = parser.parse_args()
    experiment = IDCTVisualizer()
    experiment.launch()
