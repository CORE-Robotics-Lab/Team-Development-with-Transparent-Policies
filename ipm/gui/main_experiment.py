import math
import os
import pygame
import numpy as np
import cv2
import time
import torch
from typing import Callable
import gym

from ipm.models.idct import IDCT
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.preprocessing import get_action_dim
from pygame import gfxdraw
from ipm.gui.pages import GUIPageCenterText, TreeCreationPage, EnvPage

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

        self.pages.append(GUIPageCenterText(self.screen, 'More tutorial text will go here...', 24,
                                       bottom_left_button=True, bottom_right_button=True,
                                       bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page))

        self.pages.append(GUIPageCenterText(self.screen, 'Are you ready to proceed?', 24,
                                       bottom_left_button=True, bottom_right_button=True,
                                       bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page))

        # self.pages.append(GUIPageCenterText(self.screen, 'overcooked-ai env goes here', 24,
        #                                bottom_left_button=False, bottom_right_button=False))

        model = get_oracle_idct_cartpole()
        tree_page = TreeCreationPage(model, 'cartpole', screen=self.screen, X=self.X, Y=self.Y,
                                     bottom_left_button=True, bottom_right_button=True,
                                     bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)
        env_page = EnvPage(model, 'cartpole', screen=self.screen, X=self.X, Y=self.Y)

        self.pages.append(tree_page)
        self.pages.append(env_page)


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


def get_oracle_idct_cartpole():
    env = gym.make('CartPole-v1')
    alpha = torch.Tensor([[-1], [1], [-1], [-1], [-1]])

    leaves = [[[2], [0], [2, -2]], [[], [0, 2], [-2, 2]], [[0, 1, 3], [], [2, -2]], [[0, 1], [3], [-2, 2]],
              [[0, 4], [1], [2, -2]], [[0], [1, 4], [-2, 2]]]

    weights = torch.Tensor([
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    comparators = torch.Tensor([[0.03], [-0.03], [0], [0], [0]])
    input_dim = get_obs_shape(env.observation_space)[0]
    output_dim = get_action_dim(env.action_space)

    return IDCT(input_dim=input_dim,
                 output_dim=output_dim,
                 hard_node=False,
                 device='cuda',
                 argmax_tau=1.0,
                 use_individual_alpha=True,
                 use_gumbel_softmax=False,
                 alg_type='ppo',
                 weights=weights,
                 comparators=comparators,
                 alpha=alpha,
                 leaves=leaves)

