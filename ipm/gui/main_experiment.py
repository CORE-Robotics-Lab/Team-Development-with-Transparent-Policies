import math
import os
import pygame
import numpy as np
import cv2
import time
import torch
from typing import Callable

from pygame import gfxdraw
from ipm.gui.pages import GUIPageCenterText, TreeCreationPage, EnvPage, EnvPerformancePage, OvercookedPage
from ipm.gui.policy_utils import get_idct, finetune_model

class MainExperiment:
    def __init__(self):
        pygame.init()
        self.pages = []
        self.current_page = 0
        self.X, self.Y = 1800, 800
        self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA)
        self.screen.fill('white')

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

        env_name = 'overcooked'

        model = get_idct(env_name=env_name)
        tree_page = TreeCreationPage(model, env_name, screen=self.screen, X=self.X, Y=self.Y,
                                     bottom_left_button=True, bottom_right_button=True,
                                     bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

        env_perf_page = EnvPerformancePage(env_name, tree_page, screen=self.screen, X=self.X, Y=self.Y, font_size=24,
                                             bottom_left_button=True, bottom_right_button=True,
                                             bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)


        env_page = OvercookedPage(self.screen, ' ', font_size=24,
                                         bottom_left_button=True, bottom_right_button=True,
                                         bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)
        # env_page = EnvPage('cartpole', tree_page, screen=self.screen, X=self.X, Y=self.Y)

        self.pages.append(tree_page)
        self.pages.append(env_perf_page)
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


