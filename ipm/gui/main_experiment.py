import math
import os
import pygame
import numpy as np
import cv2
import time
import torch
from typing import Callable
import pickle
from pygame import gfxdraw

from ipm.gui.nasa_tlx import run_gui
from ipm.gui.pages import GUIPageCenterText, TreeCreationPage, EnvPage, EnvPerformancePage, OvercookedPage, \
    EnvRewardModificationPage, DecisionTreeCreationPage, GUIPageWithTwoTreeChoices, GUIPageWithImage
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
        self.layout = layout
        self.bc_partner = get_human_bc_partner(traj_directory, layout, self.alt_idx)
        self.eval_partner = get_human_bc_partner(traj_directory, layout, self.ego_idx)
        self.train_env = None # for optimization conditions we want to use this
        # self.team_env = OvercookedPlayWithFixedPartner(partner=self.bc_partner, layout_name=layout,
        #                                           reduced_state_space_ego=True, reduced_state_space_alt=True,
        #                                                use_skills_ego=True, use_skills_alt=True, failed_skill_rew=0)
        self.team_env = OvercookedRoundRobinEnv(teammate_locations=teammate_paths, layout_name=layout, seed_num=0,
                                                reduced_state_space_ego=True, reduced_state_space_alt=False,
                                               use_skills_ego=True, use_skills_alt=False, failed_skill_rew=0)
        self.env = self.team_env # need to change to train env
        # self.decision_tree = DecisionTree.from_sklearn(self.bc_partner.model,
        #                                                self.team_env.n_reduced_feats,
        #                                                self.team_env.n_actions_ego)
        with open('initial_policy.pkl', 'rb') as inp:
            self.decision_tree = pickle.load(inp)


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

        pygame.init()
        self.pages = []
        self.current_page = 0
        self.settings = SettingsWrapper()
        # self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA | pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((self.settings.width, self.settings.height), pygame.SRCALPHA | pygame.RESIZABLE)
                                              #pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen.fill('white')

        env_wrapper_easy = EnvWrapper(layout='forced_coordination')
        env_wrapper_med = EnvWrapper(layout='two_rooms')
        env_wrapper_hard = EnvWrapper(layout='two_rooms_narrow')


        main_page = GUIPageCenterText(self.screen, 'Welcome to our experiment.', 24,
                                       bottom_left_button=False, bottom_right_button=True,
                                       bottom_right_fn=self.next_page)

        proceed_page = GUIPageCenterText(self.screen, 'Are you ready to proceed? (Press next when signed consent form)', 24,
                                       bottom_left_button=False, bottom_right_button=True,
                                       bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(main_page)
        self.pages.append(proceed_page)

        oc_tutorial_page = GUIPageWithImage(self.screen, 'Overcooked Gameplay Overview', 'OvercookedTutorial.png',
                                        bottom_left_button=False, bottom_right_button=True,
                                        bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(oc_tutorial_page)

        dt_tutorial_page = GUIPageWithImage(self.screen, 'Decision Tree Modification Overview', 'DTTutorial.png',
                                        bottom_left_button=False, bottom_right_button=True,
                                        bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(dt_tutorial_page)

        proceed_dt_page = GUIPageCenterText(self.screen, 'You will now see and modify your teammate as you wish. '
                                                      'After you finish, you will play a game together.', 24,
                                       bottom_left_button=False, bottom_right_button=True,
                                       bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(proceed_dt_page)

        self.easy_tree_page = DecisionTreeCreationPage(env_wrapper_easy, 'overcooked', env_wrapper_easy.layout, self.settings,
                                                  screen=self.screen,
                                                  X=self.settings.width, Y=self.settings.height,
                                                  bottom_left_button=False, bottom_right_button=True,
                                                  bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.med_tree_page = DecisionTreeCreationPage(env_wrapper_med, 'overcooked', env_wrapper_med.layout, self.settings,
                                                  screen=self.screen,
                                                  X=self.settings.width, Y=self.settings.height,
                                                  bottom_left_button=False, bottom_right_button=True,
                                                  bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.hard_tree_page = DecisionTreeCreationPage(env_wrapper_hard, 'overcooked', env_wrapper_hard.layout, self.settings,
                                                  screen=self.screen,
                                                  X=self.settings.width, Y=self.settings.height,
                                                  bottom_left_button=False, bottom_right_button=True,
                                                  bottom_left_fn=None, bottom_right_fn=self.next_page)

        n_iterations = 3

        for env_wrapper in [env_wrapper_easy, env_wrapper_med, env_wrapper_hard]:

            # for i in range(n_iterations):

            if env_wrapper.layout == 'forced_coordination':
                tree_page = self.easy_tree_page
            elif env_wrapper.layout == 'two_rooms':
                tree_page = self.med_tree_page
            elif env_wrapper.layout == 'two_rooms_narrow':
                tree_page = self.hard_tree_page
            else:
                raise ValueError('Invalid layout')

            env_page = OvercookedPage(self.screen, tree_page, layout=env_wrapper.layout, text=' ', font_size=24,
                                             bottom_left_button=False, bottom_right_button=True,
                                             bottom_left_fn=None, bottom_right_fn=self.next_page)

            tree_choice_page = GUIPageWithTwoTreeChoices(self.screen, tree_page=tree_page, env_wrapper=env_wrapper, font_size=24,
                                           bottom_left_button=True, bottom_right_button=True,
                                           bottom_left_fn=self.pick_initial_policy, bottom_right_fn=self.pick_final_policy)

            survey = GUIPageCenterText(self.screen, 'Please take survey. Press next when finished', 24,
                                           bottom_left_button=False, bottom_right_button=True,
                                           bottom_left_fn=False, bottom_right_fn=self.next_page,
                                           nasa_tlx=True)

            survey_qual = GUIPageCenterText(self.screen, 'Please take the qualtrics survey provided by the researcher.', 24,
                                           bottom_left_button=False, bottom_right_button=True,
                                           bottom_left_fn=False, bottom_right_fn=self.next_page,
                                           nasa_tlx=False)

            self.pages.append(tree_page)
            self.pages.append(env_page)
            self.pages.append(tree_choice_page)
            self.pages.append(survey)
            self.pages.append(survey_qual) # this will be outside of the iterations loop

            # env_reward_modification_page = EnvRewardModificationPage(env_wrapper, screen=self.screen, settings=self.settings,
            #                                                          X=self.settings.width, Y=self.settings.height, font_size=24,
            #                                         bottom_left_button=True, bottom_right_button=True,
            #                                         bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

            #if group == 'reward_modification':
            #    self.pages.append(env_reward_modification_page)

        thank_you_page = GUIPageCenterText(self.screen, 'Thank you for participating in our study', 24,
                                           bottom_left_button=False, bottom_right_button=False,
                                           bottom_left_fn=False, bottom_right_fn=False,
                                           nasa_tlx=False)
        self.pages.append(thank_you_page)

    def next_page(self):
        # record time spent in prior page
        self.times.append(time.time() - self.start_time)

        if self.easy_tree_page == self.pages[self.current_page]:
            self.current_domain = 0
        elif self.med_tree_page == self.pages[self.current_page]:
            self.current_domain = 1
        elif self.hard_tree_page == self.pages[self.current_page]:
            self.current_domain = 2


        # save final tree if the prior page is of type DecisionTreeCreationPage
        if self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage':
            pygame.image.save(self.screen, 'final_tree.png')
        self.pages[self.current_page].hide()
        self.current_page += 1
        if self.current_page == len(self.pages) - 1:
            # save times and pages names to csv
            with open(str(self.user_id) + 'times.csv', 'w') as outp:
                outp.write('page,time\n')
                for i in range(len(self.pages_names)):
                    outp.write(f'{self.pages_names[i]},{self.times[i]}\n')
            # save initial reward from tutorial map
            # save rewards for each game
            self.pages[self.current_page].show()
        else:
            self.pages_names.append(self.pages[self.current_page].__class__.__name__)
            self.saved_first_tree = False
            self.showed_nasa_tlx = False
            self.pages[self.current_page].show()

    def previous_page(self):
        self.pages[self.current_page].hide()
        self.current_page -= 1
        self.pages[self.current_page].show()

    def pick_initial_policy(self):
        if self.current_domain == 0:
            initial_policy = self.easy_tree_page.decision_tree_history[0]
            self.easy_tree_page.reset_initial_policy(initial_policy)
        elif self.current_domain == 1:
            initial_policy = self.med_tree_page.decision_tree_history[0]
            self.med_tree_page.reset_initial_policy(initial_policy)
        elif self.current_domain == 2:
            initial_policy = self.hard_tree_page.decision_tree_history[0]
            self.hard_tree_page.reset_initial_policy(initial_policy)
        self.next_page()

    def pick_final_policy(self):
        if self.current_domain == 0:
            final_policy = self.easy_tree_page.decision_tree_history[-1]
            self.easy_tree_page.reset_initial_policy(final_policy)
        elif self.current_domain == 1:
            final_policy = self.med_tree_page.decision_tree_history[-1]
            self.med_tree_page.reset_initial_policy(final_policy)
        elif self.current_domain == 2:
            final_policy = self.hard_tree_page.decision_tree_history[-1]
            self.hard_tree_page.reset_initial_policy(final_policy)
        # with open('initial_policy.pkl', 'wb') as outp:
        #     pickle.dump(final_policy, outp, pickle.HIGHEST_PROTOCOL)
        self.next_page()

    def launch(self):
        self.saved_first_tree = False
        self.showed_nasa_tlx = False

        pygame.init()
        clock = pygame.time.Clock()


        self.is_running = True
        self.pages[0].show()
        pygame.display.flip()
        previous_zoom = self.settings.zoom

        # start recording time, so we can get seconds spent in each page
        self.start_time = time.time()
        self.times = []
        self.pages_names = [self.pages[self.current_page].__class__.__name__]

        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                    break
                # if event.type == pygame.KEYDOWN:
                    # if scroll in, zoom
                    # if event.key == pygame.K_UP:
                    #     self.settings.zoom_in()
                    # elif event.key == pygame.K_DOWN:
                    #     self.settings.zoom_out()
                self.is_running = self.pages[self.current_page].process_event(event)
                if self.is_running is False:
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
                run_gui(self.user_id, self.condition, self.current_domain)
