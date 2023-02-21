import math
import os
import pygame
import numpy as np
import cv2
import time
import torch
from typing import Callable

from pygame import gfxdraw

from ipm.gui.nasa_tlx import run_gui
from ipm.gui.pages import GUIPageCenterText, TreeCreationPage, EnvPage, EnvPerformancePage, OvercookedPage, \
    EnvRewardModificationPage, DecisionTreeCreationPage, GUIPageWithTwoTreeChoices
from ipm.gui.policy_utils import get_idct, finetune_model
from ipm.models.bc_agent import get_human_bc_partner
from ipm.overcooked.overcooked import OvercookedRoundRobinEnv, OvercookedPlayWithFixedPartner
from ipm.models.decision_tree import DecisionTree

class EnvWrapper:
    def __init__(self, layout='forced_coordination', traj_directory='/home/mike/ipm/trajectories'):
        # wrapping this up in a class so that we can easily change the reward function
        # this acts like a pointer
        self.multipliers = [1, 1, 1]
        teammate_paths = os.path.join('data', layout, 'self_play_training_models')
        self.ego_idx = 0
        self.alt_idx = 1
        self.bc_partner = get_human_bc_partner(traj_directory, layout, self.alt_idx)
        self.eval_partner = get_human_bc_partner(traj_directory, layout, self.ego_idx)
        self.train_env = None # for optimization conditions we want to use this
        self.team_env = OvercookedPlayWithFixedPartner(partner=self.bc_partner, layout_name=layout,
                                                  reduced_state_space_ego=True, reduced_state_space_alt=True,
                                                       use_skills_ego=True, use_skills_alt=True, failed_skill_rew=0)
        self.env = self.team_env # need to change to train env
        self.decision_tree = DecisionTree.from_sklearn(self.bc_partner.model,
                                                       self.team_env.n_reduced_feats,
                                                       self.team_env.n_actions_ego)


    def initialize_env(self):
        # we keep track of the reward function that may change
        self.team_env.set_env(self.multipliers[0], self.multipliers[1], self.multipliers[2])
        # self.train_env.set_env(self.multipliers[0], self.multipliers[1], self.multipliers[2])


class SettingsWrapper:
    def __init__(self):
        self.zoom = 1
        self.max_zoom = 3
        self.min_zoom = 1
        self.width, self.height = 1920, 1080
        self.offset_x, self.offset_y = 0, 0
        self.absolute_x, self.absolute_y = self.width // 2, self.height // 2

    def zoom_out(self):
        self.zoom = max(self.zoom - 0.1, self.min_zoom)
        assert self.max_zoom >= self.zoom >= self.min_zoom

    def zoom_in(self):
        self.zoom = min(self.zoom + 0.1, self.max_zoom)
        assert self.max_zoom >= self.zoom >= self.min_zoom


class MainExperiment:
    def __init__(self, group: str):
        self.user_id = 0
        self.condition = 6
        self.experiment = 3


        pygame.init()
        self.pages = []
        self.current_page = 0
        self.settings = SettingsWrapper()
        # self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA | pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((self.settings.width, self.settings.height), pygame.SRCALPHA | pygame.RESIZABLE)
                                              #pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen.fill('white')
        env_wrapper = EnvWrapper()

        main_page = GUIPageCenterText(self.screen, 'Welcome to our experiment investigating the performance'
                                       ' of our AI-based overcooked player.', 24,
                                       bottom_left_button=False, bottom_right_button=True,
                                       bottom_right_fn=self.next_page)

        proceed_page = GUIPageCenterText(self.screen, 'Are you ready to proceed? (Press next when signed consent form)', 24,
                                       bottom_left_button=True, bottom_right_button=True,
                                       bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        self.pages.append(main_page)

        #self.pages.append(tutorial_vid_page)

        self.pages.append(proceed_page)

        model = get_idct(env_wrapper)

        tutorial_vid_page = GUIPageCenterText(self.screen, 'Tutorial images', 24,
                                       bottom_left_button=True, bottom_right_button=True,
                                       bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        self.pages.append(tutorial_vid_page)

        self.tree_page = DecisionTreeCreationPage(env_wrapper, 'overcooked', 'forced_coordination', self.settings, screen=self.screen,
                                     X=self.settings.width, Y=self.settings.height,
                                     bottom_left_button=True, bottom_right_button=True,
                                     bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        tree_choice_page = GUIPageWithTwoTreeChoices(self.screen, tree_page=self.tree_page, env_wrapper=env_wrapper, font_size=24,
                                       bottom_left_button=True, bottom_right_button=True,
                                       bottom_left_fn=self.pick_initial_policy, bottom_right_fn=self.pick_final_policy)

        env_perf_page = EnvPerformancePage(env_wrapper, self.tree_page, screen=self.screen,
                                           X=self.settings.width, Y=self.settings.height, font_size=24,
                                             bottom_left_button=True, bottom_right_button=True,
                                             bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)
        #
        # tree_page = TreeCreationPage(tree_page.tree, 'overcooked', self.settings, screen=self.screen,
        #                              X=self.settings.width, Y=self.settings.height,
        #                              bottom_left_button=True, bottom_right_button=True,
        #                              bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        env_page = OvercookedPage(self.screen, self.tree_page, layout='forced_coordination', text=' ', font_size=24,
                                         bottom_left_button=True, bottom_right_button=True,
                                         bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        perf = GUIPageCenterText(self.screen, 'The original policies performance: Reward of 120, Your modified policies performance: Reward of 100.', 24,
                                       bottom_left_button=True, bottom_right_button=True,
                                       bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        survey = GUIPageCenterText(self.screen, 'Please take survey. Press next when finished', 24,
                                       bottom_left_button=True, bottom_right_button=True,
                                       bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page,
                                       nasa_tlx=True)


        policy_performance_page = OvercookedPage(self.screen, self.tree_page, layout='forced_coordination', text=' ', font_size=24,
                                         bottom_left_button=True, bottom_right_button=True,
                                         bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)
        #
        env_reward_modification_page = EnvRewardModificationPage(env_wrapper, screen=self.screen, settings=self.settings,
                                                                 X=self.settings.width, Y=self.settings.height, font_size=24,
                                                bottom_left_button=True, bottom_right_button=True,
                                                bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        #self.pages.append(env_reward_modification_page)
        #self.pages.append(env_page)
        #self.pages.append(tree_page)
        #if group == 'reward_modification':
        #    self.pages.append(env_reward_modification_page)
        #self.pages.append(env_perf_page)
        #self.pages.append(env_page)
        self.pages.append(self.tree_page)
        self.pages.append(tree_choice_page)
        self.pages.append(self.tree_page)
        self.pages.append(survey)
        self.pages.append(env_page)
        self.pages.append(perf)
        self.pages.append(survey)
        self.pages.append(self.tree_page)
        self.pages.append(env_page)
        self.pages.append(perf)
        self.pages.append(survey)
        self.pages.append(self.tree_page)
        self.pages.append(env_page)
        self.pages.append(perf)
        self.pages.append(survey)
        #self.pages.append(GUIPageCenterText(self.screen, 'Thank you for participating in our experiment!', 24,
        #                                    bottom_left_button=False, bottom_right_button=False))

    def next_page(self):
        # save final tree if the prior page is of type DecisionTreeCreationPage
        if self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage':
            pygame.image.save(self.screen, 'final_tree.png')
        self.pages[self.current_page].hide()
        self.current_page += 1
        self.saved_first_tree = False
        self.showed_nasa_tlx = False
        self.pages[self.current_page].show()

    def previous_page(self):
        self.pages[self.current_page].hide()
        self.current_page -= 1
        self.pages[self.current_page].show()

    def pick_initial_policy(self):
        initial_policy = self.tree_page.decision_tree_history[0]
        self.tree_page.reset_initial_policy(initial_policy)
        self.next_page()

    def pick_final_policy(self):
        final_policy = self.tree_page.decision_tree_history[-1]
        self.tree_page.reset_initial_policy(final_policy)
        self.next_page()

    def launch(self):
        self.saved_first_tree = False
        self.showed_nasa_tlx = False

        pygame.init()
        clock = pygame.time.Clock()
        is_running = True
        self.pages[0].show()
        pygame.display.flip()
        previous_zoom = self.settings.zoom

        while is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False
                    break
                if event.type == pygame.KEYDOWN:
                    # if scroll in, zoom
                    if event.key == pygame.K_UP:
                        self.settings.zoom_in()
                    elif event.key == pygame.K_DOWN:
                        self.settings.zoom_out()

                is_running = self.pages[self.current_page].process_event(event)
                if is_running is False:
                    break
            self.pages[self.current_page].process_standby()

            # zoom in here
            if self.settings.zoom != 1:
                # create pygame subsurface
                wnd_w, wnd_h = self.screen.get_size()
                zoom_size = (round(wnd_w / self.settings.zoom), round(wnd_h / self.settings.zoom))
                # when fully zoomed in, make sure it is in bounds
                if self.settings.zoom != previous_zoom:
                    x, y = pygame.mouse.get_pos()
                else:
                    x = self.settings.absolute_x
                    y = self.settings.absolute_y
                x = (x + self.settings.offset_x) // self.settings.zoom
                y = (y + self.settings.offset_y) // self.settings.zoom

                # prevent any black borders
                x = max(x, zoom_size[0] // 2)
                y = max(y, zoom_size[1] // 2)
                x = min(x, wnd_w - zoom_size[0] // 2)
                y = min(y, wnd_h - zoom_size[1] // 2)

                self.settings.absolute_x = x
                self.settings.absolute_y = y
                self.settings.offset_x = int(self.settings.absolute_x * self.settings.zoom) - self.settings.width // 2
                self.settings.offset_y = int(self.settings.absolute_y * self.settings.zoom) - self.settings.height // 2

                zoom_area = pygame.Rect(0, 0, *zoom_size)
                # if self.settings.zoom == previous_zoom:
                #     x, y = wnd_w // 2, wnd_h // 2
                zoom_area.center = (x, y)
                zoom_surf = pygame.Surface(zoom_area.size)
                zoom_surf.blit(self.screen, (0, 0), zoom_area)
                zoom_surf = pygame.transform.scale(zoom_surf, (wnd_w, wnd_h))
                self.screen.blit(zoom_surf, (0, 0))
                previous_zoom = self.settings.zoom
            pygame.display.update()
            clock.tick(30)

            if not self.saved_first_tree and self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage':
                # save image of pygame window
                pygame.image.save(self.screen, 'initial_tree.png')
                self.saved_first_tree = True

            if not self.showed_nasa_tlx and self.pages[self.current_page].__class__.__name__ == 'GUIPageCenterText' \
                    and self.pages[self.current_page].nasa_tlx:
                self.showed_nasa_tlx = True
                run_gui(self.user_id, self.condition, self.experiment)





