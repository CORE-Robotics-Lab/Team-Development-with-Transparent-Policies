import math
import os
import pygame
import numpy as np
import cv2
import time
import torch
from typing import Callable

from pygame import gfxdraw
from ipm.gui.pages import GUIPageCenterText, TreeCreationPage, EnvPage, EnvPerformancePage, OvercookedPage, EnvRewardModificationPage
from ipm.gui.policy_utils import get_idct, finetune_model
from ipm.overcooked.overcooked import OvercookedSelfPlayEnv

class EnvWrapper:
    def __init__(self, layout='forced_coordination'):
        # wrapping this up in a class so that we can easily change the reward function
        # this acts like a pointer
        self.multipliers = [1, 1, 1]
        self.env = OvercookedSelfPlayEnv(layout, reduced_state_space=True)

    def initialize_env(self):
        # we keep track of the reward function that may change
        self.env.set_env(self.multipliers[0], self.multipliers[1], self.multipliers[2])

class MainExperiment:
    def __init__(self):
        pygame.init()
        self.pages = []
        self.current_page = 0
        self.X, self.Y = 1600, 900
        # self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA | pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA | pygame.RESIZABLE)
        self.screen.fill('white')
        env_wrapper = EnvWrapper()

        self.pages.append(GUIPageCenterText(self.screen, 'Welcome to our experiment investigating the performance'
                                       ' of our AI-based overcooked player.', 24,
                                       bottom_left_button=False, bottom_right_button=True,
                                       bottom_right_fn=self.next_page))

        self.pages.append(GUIPageCenterText(self.screen, 'Tutorial video will go here', 24,
                                       bottom_left_button=True, bottom_right_button=True,
                                       bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page))

        self.pages.append(GUIPageCenterText(self.screen, 'Are you ready to proceed?', 24,
                                       bottom_left_button=True, bottom_right_button=True,
                                       bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page))

        # self.pages.append(GUIPageCenterText(self.screen, 'overcooked-ai env goes here', 24,
        #                                bottom_left_button=False, bottom_right_button=False))

        model = get_idct(env_wrapper)
        tree_page = TreeCreationPage(model, env_wrapper, screen=self.screen, X=self.X, Y=self.Y,
                                     bottom_left_button=True, bottom_right_button=True,
                                     bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        env_perf_page = EnvPerformancePage(env_wrapper, tree_page, screen=self.screen, X=self.X, Y=self.Y, font_size=24,
                                             bottom_left_button=True, bottom_right_button=True,
                                             bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        tree_page = TreeCreationPage(tree_page.tree, 'overcooked', screen=self.screen, X=self.X, Y=self.Y,
                                     bottom_left_button=True, bottom_right_button=True,
                                     bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        env_page = OvercookedPage(self.screen, tree_page, ' ', font_size=24,
                                         bottom_left_button=True, bottom_right_button=True,
                                         bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        env_reward_modification_page = EnvRewardModificationPage(env_wrapper, screen=self.screen, X=self.X, Y=self.Y, font_size=24,
                                                bottom_left_button=True, bottom_right_button=True,
                                                bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        # env_page = EnvPage('cartpole', tree_page, screen=self.screen, X=self.X, Y=self.Y)

        self.pages.append(env_reward_modification_page)
        self.pages.append(tree_page)
        self.pages.append(env_perf_page)
        self.pages.append(env_page)
        self.pages.append(tree_page)
        self.pages.append(env_page)
        self.pages.append(GUIPageCenterText(self.screen, 'Thank you for participating in our experiment!', 24,
                                            bottom_left_button=False, bottom_right_button=False))

    def next_page(self):
        self.pages[self.current_page].hide()
        self.current_page += 1
        self.pages[self.current_page].show()

    def previous_page(self):
        self.pages[self.current_page].hide()
        self.current_page -= 1
        self.pages[self.current_page].show()

    def launch(self):
        pygame.init()
        clock = pygame.time.Clock()
        is_running = True
        self.pages[0].show()
        pygame.display.flip()

        while is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False
                    break
                is_running = self.pages[self.current_page].process_event(event)
                if is_running is False:
                    break
            self.pages[self.current_page].process_standby()
            pygame.display.update()
            clock.tick(30)


