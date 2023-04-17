import copy
import os
import time
import webbrowser
from abc import ABC, abstractmethod

import gym
import numpy as np
import pygame

from ipm.bin.overcooked_recorder import OvercookedPlayWithAgent
from ipm.gui.env_rendering import render_cartpole
from ipm.gui.page_components import GUIActionNodeDT, GUIDecisionNodeDT
from ipm.gui.page_components import GUIActionNodeICCT, GUIActionNodeIDCT, GUIDecisionNode, Arrow, Legend
from ipm.gui.page_components import GUIButton, Multiplier, GUITriggerButton
from ipm.gui.policy_utils import finetune_model
from ipm.gui.tree_gui_utils import Node, TreeInfo
from ipm.models.bc_agent import AgentWrapper
from ipm.models.decision_tree import BranchingNode, LeafNode


def get_button(screen, button_size, pos, button_text, button_fn):
    # surface: pygame.Surface, position: tuple, size: tuple, event_fn: Callable,
    return GUIButton(surface=screen, position=pos, event_fn=button_fn,
                     size=button_size, text=button_text, rect_color=(240, 240, 240),
                     text_color='black',
                     transparent=False,
                     border_color=(0, 0, 0), border_width=3)


def get_undo_button(screen, button_size, pos):
    return GUITriggerButton(surface=screen, position=pos,
                            size=button_size, text='Undo', rect_color=(240, 240, 240),
                            text_color='black', font_size=14,
                            transparent=False,
                            border_color=(0, 0, 0), border_width=3)


def get_reset_button(screen, button_size, pos):
    return GUITriggerButton(surface=screen, position=pos,
                            size=button_size, text='Reset', rect_color=(240, 240, 240),
                            text_color='black', font_size=14,
                            transparent=False,
                            border_color=(0, 0, 0), border_width=3)


class GUIPage(ABC):
    def __init__(self):
        self.X, self.Y = 1920, 1080
        self.gui_items = []
        self.showing = False

    @abstractmethod
    def show(self):
        pass

    def hide(self):
        self.showing = False

    @abstractmethod
    def process_event(self, event):
        pass

    @abstractmethod
    def process_standby(self):
        pass


class GUIPageCenterText(GUIPage):
    def __init__(self, screen, text, font_size, bottom_left_button=False, bottom_right_button=False,
                 bottom_left_fn=None, bottom_right_fn=None, nasa_tlx=False):
        GUIPage.__init__(self)
        self.screen = screen
        self.text = text
        self.main_font = pygame.font.Font('freesansbold.ttf', font_size)
        self.text_render = self.main_font.render(text, True, (0, 0, 0))
        self.bottom_left_button = bottom_left_button
        self.bottom_right_button = bottom_right_button
        self.bottom_left_fn = bottom_left_fn
        self.bottom_right_fn = bottom_right_fn

        self.button_size = (100, 50)
        self.button_size_x, self.button_size_y = self.button_size

        self.bottom_left_pos = (5 * self.button_size_x, self.Y - 2 * self.button_size_y)
        self.bottom_right_pos = (self.X - 5 * self.button_size_x, self.Y - 2 * self.button_size_y)

        self.nasa_tlx = nasa_tlx

    def show(self):
        self.screen.fill('white')
        self.screen.blit(self.text_render, self.text_render.get_rect(center=self.screen.get_rect().center))

        self.gui_items = []

        if self.bottom_left_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_left_pos, 'Previous', self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_right_pos, 'Next', self.bottom_right_fn))

        for item in self.gui_items:
            item.show()

        self.showing = True

    def process_event(self, event):
        for item in self.gui_items:
            result = item.process_event(event)
            if result is False:
                return False
        return True

    def process_standby(self):
        # self.show()
        for item in self.gui_items:
            item.process_standby()


class GUIPageWithTextAndURL(GUIPageCenterText):
    def __init__(self, screen, text: str, urls: list, font_size: int, bottom_left_button: bool = False,
                 bottom_right_button: bool = False,
                 bottom_left_fn=None, bottom_right_fn=None):
        GUIPageCenterText.__init__(self, screen, text, font_size, bottom_left_button, bottom_right_button,
                                   bottom_left_fn, bottom_right_fn)
        self.urls = urls
        self.url_opened = False

    def show(self):
        for url in self.urls:
            webbrowser.open(url)
        GUIPageCenterText.show(self)


class GUIPageWithImage(GUIPage):
    def __init__(self, screen, title, imagepath, bottom_left_button=False, bottom_right_button=False,
                 bottom_left_fn=None, bottom_right_fn=None):
        GUIPage.__init__(self)
        self.screen = screen
        self.text = title
        self.main_font = pygame.font.Font('freesansbold.ttf', 24)
        self.text_render = self.main_font.render(self.text, True, (0, 0, 0))
        self.bottom_left_button = bottom_left_button
        self.bottom_right_button = bottom_right_button
        self.bottom_left_fn = bottom_left_fn
        self.bottom_right_fn = bottom_right_fn

        self.button_size = (100, 50)
        self.button_size_x, self.button_size_y = self.button_size

        self.image = pygame.image.load(imagepath)

    def show(self):
        self.screen.fill('white')
        # self.screen.blit(self.text_render, self.text_render.get_rect(center=self.screen.get_rect().center))

        self.gui_items = []

        # put image in 80% of the screen, in the center
        image_size = self.image.get_size()
        image_size_x, image_size_y = image_size
        new_image_size_y = int(self.Y * 0.7)
        new_image_size_x = int(new_image_size_y * image_size_x / image_size_y)
        new_image = pygame.transform.scale(self.image, (new_image_size_x, new_image_size_y))
        self.screen.blit(new_image, (self.X / 2 - new_image_size_x / 2, 100))
        # put text in the top of the screen
        self.screen.blit(self.text_render, (self.X / 2 - self.text_render.get_size()[0] / 2, 50))

        # place buttons at the bottom of the image
        self.bottom_left_pos = (5 * self.button_size_x, self.Y - 2 * self.button_size_y)
        self.bottom_right_pos = (self.X - 5 * self.button_size_x, self.Y - 2 * self.button_size_y)

        if self.bottom_left_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_left_pos, 'Previous', self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_right_pos, 'Next', self.bottom_right_fn))

        for item in self.gui_items:
            item.show()

        self.showing = True

    def process_event(self, event):
        for item in self.gui_items:
            result = item.process_event(event)
            if result is False:
                return False
        return True

    def process_standby(self):
        # self.show()
        for item in self.gui_items:
            item.process_standby()


class GUIPageWithSingleTree(GUIPage):
    def __init__(self, is_initial, screen, tree_page, env_wrapper, font_size, bottom_left_button=False, bottom_right_button=False,
                 bottom_left_fn=None, bottom_right_fn=None):
        GUIPage.__init__(self)
        self.screen = screen
        self.tree_page = tree_page
        self.env_wrapper = env_wrapper
        self.main_font = pygame.font.Font('freesansbold.ttf', font_size)
        # self.text_render = self.main_font.render(text, True, (0, 0, 0))
        self.bottom_left_button = bottom_left_button
        self.bottom_right_button = bottom_right_button
        self.bottom_left_fn = bottom_left_fn
        self.bottom_right_fn = bottom_right_fn

        self.button_size = (100, 50)
        self.button_size_x, self.button_size_y = self.button_size

        self.is_initial = is_initial
        if is_initial:
            self.tree_filename = 'initial_tree.png'
        else:
            self.tree_filename = 'final_tree.png'

        self.tree_text = self.main_font.render('N/A', True,
                                               (0, 0, 0))

        self.loaded_images = False

    def get_performance(self, model, num_episodes=1):
        current_episode = 0
        all_rewards = []
        while current_episode < num_episodes:
            done = False
            total_reward = 0
            obs = self.env_wrapper.env.reset()
            while not done:
                action = model.predict(obs)
                obs, reward, done, info = self.env_wrapper.env.step(action)
                total_reward += reward
            all_rewards.append(total_reward)
            current_episode += 1
        return np.mean(all_rewards)

    def show(self):
        self.screen.fill('white')
        # self.screen.blit(self.text_render, self.text_render.get_rect(center=self.screen.get_rect().center))

        self.gui_items = []

        # place buttons at the bottom of the image
        self.bottom_left_pos = (5 * self.button_size_x, self.Y - 2 * self.button_size_y)
        self.bottom_right_pos = (self.X - 5 * self.button_size_x, self.Y - 2 * self.button_size_y)

        if self.bottom_left_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_left_pos, 'Previous', self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_right_pos, 'Next', self.bottom_right_fn))

        for item in self.gui_items:
            item.show()

        ratio = 16 / 9
        x = self.X // 2 - 100
        y = int(x / ratio)
        y_padding = 100
        x_padding = 100

        if not self.loaded_images:
            # show the images side by side in pygame
            self.tree_image = pygame.image.load(self.tree_filename)
            self.tree_image = self.tree_image.subsurface(
                (0, 0, self.tree_image.get_width(), self.tree_image.get_height() - 250))
            # # let's keep the aspect ratio
            #
            # self.tree_image = pygame.transform.scale(self.tree_image, (x, y))
            self.loaded_images = True

            # let's also estimate the reward performance for each tree
            if self.is_initial:
                tree_performance = self.env_wrapper.rewards[-2]
            else:
                tree_performance = self.env_wrapper.rewards[-1]

            if self.is_initial:
                tree_performance_text = "Initial Tree Performance: " + str(tree_performance)
            else:
                tree_performance_text = "Modified Tree Performance: " + str(tree_performance)

            # we want these to be displayed below the images
            self.performance_text = self.main_font.render(tree_performance_text, True, (0, 0, 0))

        # place image in horizontally centered
        # place text below that
        # self.screen.blit(self.tree_image, (x_padding, y_padding))
        # self.screen.blit(self.performance_text, (x_padding, y_padding + y + 10))

        # put image in 80% of the screen, in the center
        image_size = self.tree_image.get_size()
        image_size_x, image_size_y = image_size
        new_image_size_y = int(self.Y * 0.7)
        new_image_size_x = int(new_image_size_y * image_size_x / image_size_y)
        new_image = pygame.transform.scale(self.tree_image, (new_image_size_x, new_image_size_y))
        self.screen.blit(new_image, (self.X / 2 - new_image_size_x / 2, -25))
        self.screen.blit(self.performance_text, (self.X / 2 - self.performance_text.get_size()[0] / 2, new_image_size_y))

        self.showing = True

    def process_event(self, event):
        for item in self.gui_items:
            result = item.process_event(event)
            if result is False:
                return False
        return True

    def process_standby(self):
        # self.show()
        for item in self.gui_items:
            item.process_standby()

# this class will allow the user to choose between two options
# we will show two buttons and on top of those, we will show the respective trees (images)
class GUIPageWithTwoTreeChoices(GUIPage):
    def __init__(self, screen, tree_page, env_wrapper, font_size, bottom_left_button=False, bottom_right_button=False,
                 bottom_left_fn=None, bottom_right_fn=None):
        GUIPage.__init__(self)
        self.screen = screen
        self.tree_page = tree_page
        self.env_wrapper = env_wrapper
        self.main_font = pygame.font.Font('freesansbold.ttf', font_size)
        # self.text_render = self.main_font.render(text, True, (0, 0, 0))
        self.bottom_left_button = bottom_left_button
        self.bottom_right_button = bottom_right_button
        self.bottom_left_fn = bottom_left_fn
        self.bottom_right_fn = bottom_right_fn

        self.button_size = (400, 50)
        self.button_size_x, self.button_size_y = self.button_size

        # place buttons evenly
        self.bottom_left_pos = (0.25 * self.X - self.button_size_x // 2 + 50, self.Y - 3 * self.button_size_y)
        self.bottom_right_pos = (0.75 * self.X - self.button_size_x // 2 + 50, self.Y - 3 * self.button_size_y)

        self.initial_tree = 'initial_tree.png'
        self.final_tree = 'final_tree.png'

        self.initial_tree_text = self.main_font.render('N/A', True,
                                                       (0, 0, 0))
        self.final_tree_text = self.main_font.render('N/A', True, (0, 0, 0))

        self.loaded_images = False

    def get_performance(self, model, num_episodes=1):
        current_episode = 0
        all_rewards = []
        while current_episode < num_episodes:
            done = False
            total_reward = 0
            obs = self.env_wrapper.env.reset()
            while not done:
                action = model.predict(obs)
                obs, reward, done, info = self.env_wrapper.env.step(action)
                total_reward += reward
            all_rewards.append(total_reward)
            current_episode += 1
        return np.mean(all_rewards)

    def show(self):
        self.screen.fill('white')
        # self.screen.blit(self.text_render, self.text_render.get_rect(center=self.screen.get_rect().center))

        self.gui_items = []

        if self.bottom_left_button:
            self.gui_items.append(get_button(self.screen, self.button_size, self.bottom_left_pos, 'Choose Initial Tree',
                                             self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_right_pos, 'Choose Modified Tree',
                           self.bottom_right_fn))

        for item in self.gui_items:
            item.show()

        ratio = 16 / 9
        x = self.X // 2 - 100
        y = int(x / ratio)
        y_padding = 200
        x_padding = 100

        if not self.loaded_images:
            # show the images side by side in pygame
            self.initial_tree_image = pygame.image.load(self.initial_tree)
            self.final_tree_image = pygame.image.load(self.final_tree)
            # cut out bottom 400 pixels of image
            self.initial_tree_image = self.initial_tree_image.subsurface(
                (0, 0, self.initial_tree_image.get_width(), self.initial_tree_image.get_height() - 250))
            self.final_tree_image = self.final_tree_image.subsurface(
                (0, 0, self.final_tree_image.get_width(), self.final_tree_image.get_height() - 250))
            # let's keep the aspect ratio

            self.initial_tree_image = pygame.transform.scale(self.initial_tree_image, (x, y))
            self.final_tree_image = pygame.transform.scale(self.final_tree_image, (x, y))
            self.loaded_images = True

            # let's also estimate the reward performance for each tree
            initial_tree = self.tree_page.decision_tree_history[0]
            final_tree = self.tree_page.decision_tree_history[-1]
            # performance_initial = round(self.get_performance(initial_tree), 2)
            # performance_final = round(self.get_performance(final_tree), 2)
            performance_initial = self.env_wrapper.rewards[-2]
            performance_final = self.env_wrapper.rewards[-1]

            # we want these to be displayed below the images
            self.initial_tree_text = self.main_font.render('Initial Tree Performance: ' + str(performance_initial),
                                                           True, (0, 0, 0))
            self.final_tree_text = self.main_font.render('Modified Tree Performance: ' + str(performance_final), True,
                                                         (0, 0, 0))

        self.screen.blit(self.initial_tree_text, (x_padding + x // 2 - 130, y_padding + y + 15))
        self.screen.blit(self.final_tree_text, (self.X - x // 2 - 130, y_padding + y + 15))

        self.screen.blit(self.initial_tree_image, (x_padding, y_padding))
        self.screen.blit(self.final_tree_image, (self.X - x, y_padding))

        self.showing = True

    def process_event(self, event):
        for item in self.gui_items:
            result = item.process_event(event)
            if result is False:
                return False
        return True

    def process_standby(self):
        # self.show()
        for item in self.gui_items:
            item.process_standby()


class OvercookedPage(GUIPage):
    def __init__(self, screen, env_wrapper, tree_page, layout, text, font_size, bottom_left_button=False,
                 bottom_right_button=False,
                 bottom_left_fn=None, bottom_right_fn=None):
        GUIPage.__init__(self)
        self.screen = screen
        self.text = text
        self.env_wrapper = env_wrapper
        self.tree_page = tree_page
        self.layout_name = layout
        self.main_font = pygame.font.Font('freesansbold.ttf', font_size)
        self.text_render = self.main_font.render(text, True, (255, 255, 255))
        self.bottom_left_button = bottom_left_button
        self.bottom_right_button = bottom_right_button
        self.bottom_left_fn = bottom_left_fn
        self.bottom_right_fn = bottom_right_fn

    def show(self):
        robot_policy = AgentWrapper(self.tree_page.current_policy)
        traj_folder = os.path.join(self.env_wrapper.data_folder, 'trajectories')
        if not os.path.exists(traj_folder):
            os.makedirs(traj_folder)
        # TODO: ego_idx fixed here? should be the same throughout the experiment though
        demo = OvercookedPlayWithAgent(agent=robot_policy,
                                       behavioral_model=self.env_wrapper.intent_model,
                                       traj_directory=traj_folder,
                                       layout_name=self.layout_name,
                                       n_episodes=1,
                                       ego_idx=0,
                                       screen=self.screen)
        final_rew = demo.play()
        self.env_wrapper.rewards.append(final_rew)

    def process_event(self, event):
        self.bottom_right_fn()
        return True

    def process_standby(self):
        pass


class TreeCreationPage:
    def __init__(self, tree, env_name='overcooked', settings_wrapper=None, screen=None, X=None, Y=None,
                 is_continuous_actions: bool = True,
                 bottom_left_button=False, bottom_right_button=False, bottom_left_fn=None, bottom_right_fn=None):
        self.tree = tree
        self.is_continuous_actions = is_continuous_actions
        self.env_name = env_name
        self.settings = settings_wrapper

        if X is None:
            self.X = 1600
        else:
            self.X = X
        if Y is None:
            self.Y = 900
        else:
            self.Y = Y
        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA)
        else:
            self.screen = screen

        if self.env_name == 'lunar':
            self.env_feat_names = ['x coordinate', 'y coordinate', 'horizontal velocity', 'vertical velocity',
                                   'orientation', 'angular velocity', 'left leg touching', 'right leg touching']
            self.action_names = ['Main Engine Thrust', 'Side Engines Net Thrust']
            self.n_actions = len(self.action_names)
            self.is_continuous_actions = True
        elif self.env_name == 'ip':
            self.env_feat_names = ['position', 'vertical angle', 'linear velocity', 'angular velocity']
            self.action_names = ['Force applied']
            self.n_actions = len(self.action_names)
            self.is_continuous_actions = True
        elif self.env_name == 'cartpole':
            self.env_feat_names = ['Position', 'Linear Velocity', 'Vertical Angle', 'Angular Velocity']
            self.action_names = ['Move Left', 'Move Right']
            self.n_actions = 1
            self.is_continuous_actions = False
        elif self.env_name == 'overcooked':
            # self.env_feat_names = ['Feature' + str(i) for i in range(96)]
            # self.env_feat_names = ['Is Facing Up', 'Is Facing Down', 'Is Facing Right',
            #                         'Is Facing Left',
            #                         'Is Holding Onion',
            #                         'Is Holding Soup',
            #                         'Is Holding Dish',
            #                         'Is Holding Tomato',
            #                         'Closest Onion X Location',
            #                         'Closest Onion Y Location',
            #                         'Closest Tomato X Location',
            #                         'Closest Tomato Y Location',
            #                         'Closest Dish X Location',
            #                         'Closest Dish Y Location',
            #                         'Closest Soup X Location',
            #                         'Closest Soup Y Location',
            #                         'Closest Soup # Onions',
            #                         'Closest Soup # Tomatos',
            #                         'Closest Serving X Location',
            #                         'Closest Serving Y Location',
            #                         'Closest Empty Counter X Location',
            #                         'Closest Empty Counter Y Location',
            #                         'Closest Pot Exists',
            #                         'Closest Pot Is Empty',
            #                         'Closest Pot Is Full',
            #                         'Closest Pot Is Cooking',
            #                         'Closest Pot Is Ready',
            #                         'Closest Pot # Onions',
            #                         'Closest Pot # Tomatoes',
            #                         'Closest Pot Cook Time',
            #                         'Closest Pot X Location',
            #                         'Closest Pot Y Location',
            #                         '2nd Closest Pot Exists',
            #                         '2nd Closest Pot Is Empty',
            #                         '2nd Closest Pot Is Full',
            #                         '2nd Closest Pot Is Cooking',
            #                         '2nd Closest Pot Is Ready',
            #                         '2nd Closest Pot # Onions',
            #                         '2nd Closest Pot # Tomatoes',
            #                         '2nd Closest Pot Cook Time',
            #                         '2nd Closest Pot X Location',
            #                         '2nd Closest Pot Y Location',
            #                         'Is Wall North',
            #                         'Is Wall South',
            #                         'Is Wall East',
            #                         'Is Wall West',
            #                         'Other Player Is Facing Up',
            #                         'Other Player Is Facing Down',
            #                         'Other Player Is Facing Right',
            #                         'Other Player Is Facing Left',
            #                         'Other Player Is Holding Onion',
            #                         'Other Player Is Holding Soup',
            #                         'Other Player Is Holding Dish',
            #                         'Other Player Is Holding Tomato',
            #                         'Other Player Closest Onion X Location',
            #                         'Other Player Closest Onion Y Location',
            #                         'Other Player Closest Tomato X Location',
            #                         'Other Player Closest Tomato Y Location',
            #                         'Other Player Closest Dish X Location',
            #                         'Other Player Closest Dish Y Location',
            #                         'Other Player Closest Soup X Location',
            #                         'Other Player Closest Soup Y Location',
            #                         'Other Player Closest Soup # Onions',
            #                         'Other Player Closest Soup # Tomatos',
            #                         'Other Player Closest Serving X Location',
            #                         'Other Player Closest Serving Y Location',
            #                         'Other Player Closest Empty Counter X Location',
            #                         'Other Player Closest Empty Counter Y Location',
            #                         'Other Player Closest Pot Exists',
            #                         'Other Player Closest Pot Is Empty',
            #                         'Other Player Closest Pot Is Full',
            #                         'Other Player Closest Pot Is Cooking',
            #                         'Other Player Closest Pot Is Ready',
            #                         'Other Player Closest Pot # Onions',
            #                         'Other Player Closest Pot # Tomatoes',
            #                         'Other Player Closest Pot Cook Time',
            #                         'Other Player Closest Pot X Location',
            #                         'Other Player Closest Pot Y Location',
            #                         'Other Player 2nd Closest Pot Exists',
            #                         'Other Player 2nd Closest Pot Is Empty',
            #                         'Other Player 2nd Closest Pot Is Full',
            #                         'Other Player 2nd Closest Pot Is Cooking',
            #                         'Other Player 2nd Closest Pot Is Ready',
            #                         'Other Player 2nd Closest Pot # Onions',
            #                         'Other Player 2nd Closest Pot # Tomatoes',
            #                         'Other Player 2nd Closest Pot Cook Time',
            #                         'Other Player 2nd Closest Pot X Location',
            #                         'Other Player 2nd Closest Pot Y Location',
            #                         'Other Player Is Wall North',
            #                         'Other Player Is Wall South',
            #                         'Other Player Is Wall East',
            #                         'Other Player Is Wall West',
            #                         'X Location',
            #                         'Y Location',
            #                         'X Location (Absolute)',
            #                         'Y Location (Absolute)']
            # self.env_feat_names = ['Direction Facing',
            #                         'Which Object Holding',
            #                         'Closest Soup # Onions',
            #                         'Closest Soup # Tomatoes',
            #                         'Closest Pot Is Cooking',
            #                         'Closest Pot Is Ready',
            #                         'Closest Pot # Onions',
            #                         'Closest Pot # Tomatoes',
            #                         'Closest Pot Cook Time',
            #                         '2nd Closest Pot Is Cooking',
            #                         '2nd Closest Pot Is Ready',
            #                         '2nd Closest Pot # Onions',
            #                         '2nd Closest Pot # Tomatoes',
            #                         '2nd Closest Pot Cook Time',
            #                         'Player X Position',
            #                         'Player Y Position']
            # assert len(self.env_feat_names) == 16
            self.env_feat_names = ['P1 Facing Up',
                                   'P1 Facing Down',
                                   'P1 Facing Right',
                                   'P1 Facing Left',

                                   'P1 Holding Onion',
                                   'P1 Holding Soup',
                                   'P1 Holding Dish',
                                   'P1 Holding Tomato',

                                   'Pot 1 Is Cooking',
                                   # 'Pot 1 Is Ready',
                                   'Pot 1 Needs Ingredients',
                                   'Pot 1 (Almost) Ready',

                                   'Pot 2 Is Cooking',
                                   # 'Pot 2 Is Ready',
                                   'Pot 2 Needs Ingredients',
                                   'Pot 2 (Almost) Ready',

                                   # 'Player X Position',
                                   # 'Player Y Position',
                                   # 'Other Agent X Position',
                                   # 'Other Agent Y Position',

                                   'P2 Facing Up',
                                   'P2 Facing Down',
                                   'P2 Facing Right',
                                   'P2 Facing Left',

                                   'P2 Holding Onion',
                                   'P2 Holding Soup',
                                   'P2 Holding Dish',
                                   'P2 Holding Tomato',

                                   'Dish on a Counter',
                                   'Soup on a Counter',
                                   'Onion on a Counter',
                                   'Tomato on a Counter', ]

            self.action_names = ['Move Up', 'Move Down', 'Move Right', 'Move Left', 'Wait', 'Interact',
                                 'Get Closest Onion', 'Get Closest Tomato', 'Get Closest Dish', 'Get Closest Soup',
                                 'Serve Soup', 'Bring to Closest Pot', 'Place on Counter']
            self.n_actions = 1  # we only take 1 action at a time
            self.is_continuous_actions = False
        else:
            raise ValueError('Invalid environment name')

        assert len(self.env_feat_names) == tree.input_dim
        assert len(self.action_names) == tree.output_dim

        self.env_feat_names = [name[:15] + '..' for name in self.env_feat_names]
        self.action_names = [name[:15] + '..' for name in self.action_names]

        self.bottom_left_button = bottom_left_button
        self.bottom_right_button = bottom_right_button
        self.bottom_left_fn = bottom_left_fn
        self.bottom_right_fn = bottom_right_fn

        self.button_size = (100, 50)
        self.button_size_x, self.button_size_y = self.button_size

        self.bottom_left_pos = (5 * self.button_size_x, self.Y - 2 * self.button_size_y)
        self.bottom_right_pos = (self.X - 5 * self.button_size_x, self.Y - 2 * self.button_size_y)

        self.y_spacing = 175

        self.decision_node_color = (137, 207, 240, 128)
        self.decision_node_border_color = (137, 207, 240, 255)
        self.action_leaf_color = (240, 128, 101, 128)
        self.action_leaf_border_color = (240, 128, 101, 255)

        # decision_node_size_x = 370
        self.decision_node_size_x = 370
        self.decision_node_size_y = 100
        self.decision_node_size = (self.decision_node_size_x, self.decision_node_size_y)

        # action_leaf_size_x = 220
        self.action_leaf_size_x = 180
        self.action_leaf_size_y = 100
        self.action_leaf_size = (self.action_leaf_size_x, self.action_leaf_size_y)

    def show_leaf(self, leaf: Node, leaf_x_pos_perc: float, leaf_y_pos: float):
        for i in range(self.n_actions):
            node_position = ((leaf_x_pos_perc * self.X) - (self.action_leaf_size_x // 2),
                             leaf_y_pos + i * (self.action_leaf_size_y + 20))
            if self.is_continuous_actions:
                name = self.action_names[i]
                node = GUIActionNodeICCT(self.tree, self.screen, self.settings, node_position,
                                         size=self.action_leaf_size, font_size=14, name=name,
                                         text=self.action_node_texts[leaf.idx][i],
                                         rect_color=self.action_leaf_color, border_color=self.action_leaf_border_color,
                                         border_width=3)
            else:
                logits = list(self.tree_info.leaves[leaf.idx][2])
                action_idx = logits.index(max(logits))
                node = GUIActionNodeIDCT(self.tree, self.screen, self.settings, node_position,
                                         size=self.action_leaf_size, font_size=14,
                                         leaf_idx=leaf.idx, action_idx=action_idx, actions_list=self.action_names,
                                         rect_color=self.action_leaf_color, border_color=self.action_leaf_border_color,
                                         border_width=3)
            self.gui_items.append(node)

    def construct_page(self):

        self.gui_items = []

        if self.bottom_left_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_left_pos, 'Previous', self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_right_pos, 'Next', self.bottom_right_fn))

        leg = Legend(self.screen, 1400, 50, 130, 40, self.decision_node_color, self.action_leaf_color,
                     self.decision_node_border_color, self.action_leaf_border_color, None, None,
                     [], selected=-1, transparent=True)
        self.gui_items.append(leg)
        self.construct_subtree(self.tree_info.root, node_x_pos_perc=1 / 2)

    def construct_subtree(self, node: Node, node_x_pos_perc: float):

        depth = node.node_depth
        node_y_pos = self.y_spacing * depth + 50
        child_y_pos = self.y_spacing * (depth + 1) + 50

        node_var_idx = self.tree_info.impactful_vars_for_nodes[node.idx]
        compare_sign = '>' if self.tree_info.compare_sign[node.idx][node_var_idx] else '<'
        comparator_value = self.tree_info.comparators.detach().numpy()[node.idx][0]

        node_position = ((node_x_pos_perc * self.X) - (self.decision_node_size_x // 2), node_y_pos)
        gui_node = GUIDecisionNode(self.tree, node.idx, self.env_feat_names, self.screen, self.settings,
                                   node_position, size=self.decision_node_size, font_size=24,
                                   variable_idx=node_var_idx, compare_sign=compare_sign,
                                   comparator_value=str(round(comparator_value, 2)),
                                   rect_color=self.decision_node_color, border_color=self.decision_node_border_color,
                                   border_width=3)
        self.gui_items.append(gui_node)

        left_child_x_pos_perc = node_x_pos_perc - (1 / 2 ** (depth + 2))
        right_child_x_pos_perc = node_x_pos_perc + (1 / 2 ** (depth + 2))

        left_arrow = Arrow(self.screen,
                           pygame.Vector2(node_x_pos_perc * self.X, node_y_pos + self.decision_node_size_y),
                           pygame.Vector2(left_child_x_pos_perc * self.X, child_y_pos))
        right_arrow = Arrow(self.screen,
                            pygame.Vector2(node_x_pos_perc * self.X, node_y_pos + self.decision_node_size_y),
                            pygame.Vector2(right_child_x_pos_perc * self.X, child_y_pos))

        self.gui_items.append(left_arrow)
        self.gui_items.append(right_arrow)

        if not node.left_child.is_leaf:
            self.construct_subtree(node.left_child, left_child_x_pos_perc)
        else:
            self.show_leaf(node.left_child, left_child_x_pos_perc, child_y_pos)
        if not node.right_child.is_leaf:
            self.construct_subtree(node.right_child, right_child_x_pos_perc)
        else:
            self.show_leaf(node.right_child, right_child_x_pos_perc, child_y_pos)

    def get_action_leaf_text(self, leaf_idx: int):
        texts = []
        if self.is_continuous_actions:
            for i in range(self.n_actions):
                variable_text = self.env_feat_names[self.tree_info.action_node_vars[i, leaf_idx]]
                if self.n_actions > 1:
                    scalar = self.tree_info.action_scalars[leaf_idx][i]
                    bias = self.tree_info.action_biases[leaf_idx][i]
                    std = self.tree_info.action_stds[leaf_idx][i]
                else:
                    scalar = self.tree_info.action_scalars[leaf_idx]
                    bias = self.tree_info.action_biases[leaf_idx]
                    std = self.tree_info.action_stds[leaf_idx]
                # text = 'N(' + str(round(scalar, 2)) + ' * ' + variable_text + ' + ' + str(round(bias, 2)) + ', ' + str(
                #     round(std, 2)) + ')'
                text = str(round(scalar, 2)) + ' * ' + variable_text + ' + ' + str(round(bias, 2))
                texts.append(text)
        else:
            logits = list(self.tree_info.leaves[leaf_idx][2])
            action_idx = logits.index(max(logits))
            text = self.action_names[action_idx]
            texts.append(text)
        return texts

    def hide(self):
        self.showing = False

    def show_tree(self):
        self.tree_info = TreeInfo(self.tree, self.is_continuous_actions)
        self.action_node_texts = [self.get_action_leaf_text(leaf_idx) for leaf_idx in range(len(self.tree_info.leaves))]
        self.construct_page()

    def show(self):
        self.showing = True
        self.screen.fill('white')
        self.show_tree()
        for item in self.gui_items:
            item.show()
        for item in self.gui_items:
            item.show_children()

    def process_event(self, event):
        for gui_item in self.gui_items:
            result_signal, result = gui_item.process_event(event)
            if result_signal == 'new_tree':
                self.tree = result
                self.show_tree()
        return True

    def process_standby(self):
        self.screen.fill('white')
        # for item in self.gui_items:
        #     item.show()
        for item in self.gui_items:
            item.process_standby()
        for item in self.gui_items:
            item.show_children()


class EnvPerformancePage(GUIPageCenterText):
    def __init__(self, env_wrapper, tree_page, screen, X, Y, font_size, bottom_left_button=False,
                 bottom_right_button=False, bottom_left_fn=None, bottom_right_fn=None):
        self.screen = screen
        self.X = X
        self.Y = Y
        self.env_wrapper = env_wrapper
        self.tree_page = tree_page
        super().__init__(screen, '', font_size, bottom_left_button, bottom_right_button, bottom_left_fn,
                         bottom_right_fn)

    def get_performance(self, model, num_episodes=1):
        # TODO: Replace SP teammates with BC model or some other model!

        current_episode = 0
        all_rewards = []
        while current_episode < num_episodes:
            done = False
            total_reward = 0
            obs = self.env_wrapper.env.reset()
            while not done:
                action = model.predict(obs)
                obs, reward, done, info = self.env_wrapper.env.step(action)
                total_reward += reward
            all_rewards.append(total_reward)
            current_episode += 1
        return np.mean(all_rewards)

    def get_finetuned_performance(self, initial_model):
        model = finetune_model(initial_model, env_wrapper=self.env_wrapper)
        return self.get_performance(model)

    def show(self):
        initial_perf = round(self.get_performance(self.tree_page.current_policy), 2)
        finetuned_perf = round(self.get_finetuned_performance(self.tree_page.current_policy), 2)
        self.text = 'Your tree\'s performance on ' + 'overcooked' + ': ' + str(initial_perf)
        self.text_render = self.main_font.render(self.text, True, (0, 0, 0))
        self.improved_text = 'Your teammate believe they found an improved policy with performance: ' + str(
            finetuned_perf)
        self.improved_text_render = self.main_font.render(self.improved_text, True, (0, 0, 0))

        self.screen.fill('white')
        center_x, center_y = self.screen.get_rect().center
        spacing = 100
        self.screen.blit(self.text_render, self.text_render.get_rect(center=(center_x, center_y - spacing)))
        self.screen.blit(self.improved_text_render,
                         self.improved_text_render.get_rect(center=(center_x, center_y + spacing)))

        self.gui_items = []

        if self.bottom_left_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_left_pos, 'Previous', self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_right_pos, 'Next', self.bottom_right_fn))

        for item in self.gui_items:
            item.show()

        self.showing = True


class EnvRewardModificationPage(GUIPageCenterText):
    def __init__(self, env_wrapper, screen, settings, X, Y, font_size, bottom_left_button=False,
                 bottom_right_button=False, bottom_left_fn=None, bottom_right_fn=None):
        self.screen = screen
        self.X = X
        self.Y = Y
        self.settings = settings
        self.env_wrapper = env_wrapper
        super().__init__(screen, '', font_size, bottom_left_button, bottom_right_button, bottom_left_fn,
                         bottom_right_fn)

    def show_text(self):
        self.text_item_in_pot = 'Importance for placing an item in a pot: '
        self.text_render_item_in_pot = self.main_font.render(self.text_item_in_pot, True, (0, 0, 0))
        self.text_item_pickup_dish = 'Importance for picking up a dish: '
        self.text_render_item_pickup_dish = self.main_font.render(self.text_item_pickup_dish, True, (0, 0, 0))
        self.text_item_pickup_soup = 'Importance for placing an item in a pot: '
        self.text_render_item_pickup_soup = self.main_font.render(self.text_item_pickup_soup, True, (0, 0, 0))

        self.screen.fill('white')
        center_x, center_y = self.screen.get_rect().center
        spacing = 200
        y_offset = -100
        self.screen.blit(self.text_render_item_in_pot,
                         self.text_render_item_in_pot.get_rect(center=(center_x, center_y - spacing + y_offset)))
        self.screen.blit(self.text_render_item_pickup_dish,
                         self.text_render_item_pickup_dish.get_rect(center=(center_x, center_y + y_offset)))
        self.screen.blit(self.text_render_item_pickup_soup,
                         self.text_render_item_pickup_soup.get_rect(center=(center_x, center_y + spacing + y_offset)))

    def show(self):
        self.screen.fill('white')
        self.show_text()
        center_x, center_y = self.screen.get_rect().center
        spacing = 200
        y_offset = -100

        self.gui_items = []

        if self.bottom_left_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_left_pos, 'Previous', self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_right_pos, 'Next', self.bottom_right_fn))

        self.node_box_item_in_pot = Multiplier(self.env_wrapper, 0, self.screen, self.settings,
                                               (center_x + 50, center_y - spacing + y_offset - 15))
        self.node_box_pickup_dish = Multiplier(self.env_wrapper, 1, self.screen, self.settings,
                                               (center_x + 50, center_y + y_offset - 15))
        self.node_box_pickup_soup = Multiplier(self.env_wrapper, 2, self.screen, self.settings,
                                               (center_x + 50, center_y + spacing + y_offset - 15))
        self.gui_items.append(self.node_box_item_in_pot)
        self.gui_items.append(self.node_box_pickup_dish)
        self.gui_items.append(self.node_box_pickup_soup)

        self.showing = True

    def process_event(self, event):
        for item in self.gui_items:
            result = item.process_event(event)
            if result is False:
                return False
        return True

    def process_standby(self):
        # self.show()
        self.screen.fill('white')
        self.show_text()
        for item in self.gui_items:
            item.show()
        for item in self.gui_items:
            item.process_standby()
        for item in self.gui_items:
            item.show_children()


class EnvPage:
    def __init__(self, env_name, tree_page, screen, X, Y):
        self.N_TOTAL_SAMPLES = 1000
        self.CURRENT_NUM_SAMPLES = 0
        self.screen = screen
        self.X = X
        self.Y = Y
        self.env_name = env_name
        self.tree_page = tree_page

    def show(self):
        if self.env_name == 'cartpole':
            env = gym.make('CartPole-v1')
        else:
            raise NotImplementedError

        actual_env = env.env.env
        actual_env.screen = self.screen
        actual_env.screen_width = self.X
        actual_env.screen_height = self.Y

        num_eps = 3
        curr_ep = 0

        obs = env.reset()
        while curr_ep < num_eps:
            action = self.tree_page.current_policy.predict(obs)
            render_cartpole(env)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                curr_ep += 1


class DecisionTreeCreationPage:
    def __init__(self, env_wrapper, layout_name, domain_idx, settings_wrapper=None, screen=None, X=None, Y=None,
                 is_continuous_actions: bool = True,
                 bottom_left_button=False, bottom_right_button=False, bottom_left_fn=None, bottom_right_fn=None,
                 horizontal_layout=False):
        self.env_wrapper = env_wrapper
        self.domain_idx = domain_idx
        self.reset_initial_policy(env_wrapper.current_policy)
        self.settings = settings_wrapper
        self.horizontal_layout = horizontal_layout

        if X is None:
            self.X = 1600
        else:
            self.X = X
        if Y is None:
            self.Y = 900
        else:
            self.Y = Y
        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA)
        else:
            self.screen = screen

        # TODO: verify that these env_feat_names are correct
        self.env_feat_names = [
                               #  'AI Facing Up',
                               # 'AI Facing Down',
                               # 'AI Facing Right',
                               # 'AI Facing Left',

                               'AI Holding Onion',
                               'AI Holding Soup',
                               'AI Holding Dish']
        if layout_name == 'two_rooms_narrow':
            self.env_feat_names += ['AI Holding Tomato']
        self.env_feat_names += [
            # 'Human Facing Up',
            # 'Human Facing Down',
            # 'Human Facing Right',
            # 'Human Facing Left',

            'Human Holding Onion',
            'Human Holding Soup',
            'Human Holding Dish']
        if layout_name == 'two_rooms_narrow':
            self.env_feat_names += ['Human Holding Tomato']
        self.env_feat_names += [
            'Onion on Shared Counter']
        if layout_name == 'two_rooms_narrow':
            self.env_feat_names += ['Tomato on Shared Counter']
        self.env_feat_names += [
            'Pot 1 Needs Ingredients',
            'Pot 2 Needs Ingredients',
            'Either Pot Needs Ingredients',
            'A Pot is Ready',
            'Dish on Shared Counter',
            'Soup on Shared Counter',
            'Human Picking Up Onion']
        if layout_name == 'two_rooms_narrow':
            self.env_feat_names += ['Human Picking Up Tomato']
        self.env_feat_names += [
            'Human Picking Up Dish',
            'Human Picking Up Soup',
            'Human Serving Dish',
            'Human Placing Item Down',
        ]

        self.action_names = ['Wait',
                             'Get Onion Dispenser', 'Get Onion Counter',
                              'Get Dish from Dispenser', 'Get Dish from Counter',
                              'Get Soup from Pot', 'Get Soup from Counter',
                              'Serve Soup', 'Bring To Pot', 'Place on Counter']

        if layout_name == 'two_rooms_narrow':
            self.action_names += ['Get Tomato from Dispenser', 'Get Tomato from Counter']

        self.n_actions = 1  # we only take 1 action at a time
        self.is_continuous_actions = False

        assert len(self.env_feat_names) == self.current_policy.num_vars
        assert len(self.action_names) == self.current_policy.num_actions

        # self.env_feat_names = [name[:15] + '..' for name in self.env_feat_names]
        # self.action_names = [name[:15] + '..' for name in self.action_names]

        self.bottom_left_button = bottom_left_button
        self.bottom_right_button = bottom_right_button
        self.bottom_left_fn = bottom_left_fn
        self.bottom_right_fn = bottom_right_fn

        self.button_size = (100, 50)
        self.button_size_x, self.button_size_y = self.button_size

        self.bottom_left_pos = (5 * self.button_size_x, self.Y - 2 * self.button_size_y)
        self.bottom_right_pos = (self.X - 5 * self.button_size_x, self.Y - 2 * self.button_size_y)

        self.level_spacing = 175

        self.decision_node_color = (137, 207, 240, 128)
        self.decision_node_border_color = (137, 207, 240, 255)
        self.action_leaf_color = (240, 128, 101, 128)
        self.action_leaf_border_color = (240, 128, 101, 255)

        # decision_node_size_x = 370
        self.decision_node_size_x = 465 // 2
        self.decision_node_size_y = 150 // 2
        self.decision_node_size = (self.decision_node_size_x, self.decision_node_size_y)

        # action_leaf_size_x = 220
        self.action_leaf_size_x = 230 // 2
        self.action_leaf_size_y = 120 // 2
        self.action_leaf_size = (self.action_leaf_size_x, self.action_leaf_size_y)
        self.time_since_last_undo = time.time()

    def reset_initial_policy(self, policy):
        self.current_policy = policy
        # TODO: go back and fix this by making a custom function that copies tensors?
        self.current_tree_copy = self.current_policy
        self.decision_tree_history = [self.current_tree_copy]

    def show_leaf(self, leaf, leaf_pos_perc: float, leaf_level_pos: float, horizontal_layout=False):
        for i in range(1):
            if not horizontal_layout:
                node_pos_x = (leaf_pos_perc * self.X) - (self.action_leaf_size_x // 2)
                node_pos_y = leaf_level_pos + i * (self.action_leaf_size_y + 20)
            else:
                node_pos_x = leaf_level_pos + i * (self.action_leaf_size_x + 20)
                node_pos_y = (leaf_pos_perc * self.Y) - (self.action_leaf_size_y // 2)

            node_position = (node_pos_x, node_pos_y)
            action_idx = leaf.action.indices
            action_prob = leaf.action.values # round(leaf.action.values[i],2)
            if i==0:
                first_one = True
            else:
                first_one = False
            node = GUIActionNodeDT(self.current_policy, leaf, self.screen, self.settings, domain_idx=self.domain_idx,
                                   position=node_position,
                                   size=self.action_leaf_size, font_size=18,
                                   leaf_idx=leaf.idx, action_idx=action_idx, actions_list=self.action_names,
                                   rect_color=self.action_leaf_color, border_color=self.action_leaf_border_color,
                                   border_width=3, action_prob=action_prob, first_one=first_one)
            self.gui_items.append(node)

    def construct_page(self):

        self.gui_items = []

        if self.bottom_left_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_left_pos, 'Previous', self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(
                get_button(self.screen, self.button_size, self.bottom_right_pos, 'Next', self.bottom_right_fn))
        # undo button

        x_size, y_size = self.button_size
        x_size /= 2
        y_size /= 2
        button_size = (x_size, y_size)

        if not self.horizontal_layout:
            undo_pos = (3 * self.X // 5, self.Y // 15 - 5)
        else:
            undo_pos = (self.X // 15, 3 * self.Y // 5)

        self.gui_items.append(get_undo_button(self.screen, button_size, undo_pos))

        if not self.horizontal_layout:
            reset_pos = (3 * self.X // 5 + x_size + 10, self.Y // 15 - 5)
        else:
            reset_pos = (self.X // 15 + x_size + 10, 3 * self.Y // 5 + y_size + 10)

        self.gui_items.append(get_reset_button(self.screen, button_size, reset_pos))

        leg = Legend(self.screen, 1400, 50, 130, 40, self.decision_node_color, self.action_leaf_color,
                     self.decision_node_border_color, self.action_leaf_border_color, None, None,
                     [], selected=-1, transparent=True)
        self.gui_items.append(leg)
        self.construct_subtree(self.current_policy.root, node_pos_perc=1 / 2)

    def construct_subtree(self, node: BranchingNode, node_pos_perc: float):

        depth = node.depth

        if not self.horizontal_layout:
            node_level_pos = self.level_spacing * depth + self.decision_node_size_y
            child_level_pos = self.level_spacing * (depth + 1) + self.decision_node_size_y
        else:
            node_level_pos = self.level_spacing * 2 * depth + self.decision_node_size_x
            child_level_pos = self.level_spacing * 2 * (depth + 1) + self.decision_node_size_x

        node_var_idx = node.var_idx
        compare_sign = '<=' if node.normal_ordering == 0 else '>'

        if not self.horizontal_layout:
            node_x_pos = (node_pos_perc * self.X) - (self.decision_node_size_x // 2)
            node_y_pos = node_level_pos
        else:
            node_x_pos = node_level_pos
            node_y_pos = (node_pos_perc * self.Y) - (self.decision_node_size_y // 2)
        node_position = (node_x_pos, node_y_pos)
        gui_node = GUIDecisionNodeDT(self.current_policy, node, self.env_feat_names, self.screen, self.settings,
                                     domain_idx=self.domain_idx, position=node_position, size=self.decision_node_size,
                                     font_size=18,
                                     variable_idx=node_var_idx, compare_sign=compare_sign,
                                     rect_color=self.decision_node_color, border_color=self.decision_node_border_color,
                                     border_width=3)
        self.gui_items.append(gui_node)

        left_child_pos_perc = node_pos_perc - (1 / 2 ** (depth + 2))
        right_child_pos_perc = node_pos_perc + (1 / 2 ** (depth + 2))

        # if we have a horizontal layout, we need to change some things
        if not self.horizontal_layout:
            left_child_y_pos = node_level_pos + self.decision_node_size_y
            right_child_y_pos = node_level_pos + self.decision_node_size_y
            left_child_x_pos = left_child_pos_perc * self.X
            right_child_x_pos = right_child_pos_perc * self.X
        else:
            left_child_x_pos = node_level_pos + self.decision_node_size_x
            right_child_x_pos = node_level_pos + self.decision_node_size_x
            left_child_y_pos = left_child_pos_perc * self.Y
            right_child_y_pos = right_child_pos_perc * self.Y

        # for the arrows, we need to account for the actual box.
        # for the horizontal layout: the arrows should start from the end of the box
        if not self.horizontal_layout:
            arrow_start_x = node_x_pos + self.decision_node_size_x // 2
            arrow_start_y = node_y_pos + self.decision_node_size_y
            arrow_left_x = left_child_x_pos  # + self.decision_node_size_x // 2
            arrow_left_y = child_level_pos
            arrow_right_x = right_child_x_pos  # + self.decision_node_size_x // 2
            arrow_right_y = child_level_pos
        else:
            arrow_start_x = node_x_pos + self.decision_node_size_x
            arrow_start_y = node_y_pos + self.decision_node_size_y // 2
            arrow_left_x = child_level_pos
            arrow_left_y = left_child_y_pos  # + self.decision_node_size_y // 2
            arrow_right_x = child_level_pos
            arrow_right_y = right_child_y_pos  # + self.decision_node_size_y // 2

        left_arrow = Arrow(self.screen, pygame.Vector2(arrow_start_x, arrow_start_y),
                           pygame.Vector2(arrow_left_x, arrow_left_y), text='True', text_left=True)
        right_arrow = Arrow(self.screen, pygame.Vector2(arrow_start_x, arrow_start_y),
                            pygame.Vector2(arrow_right_x, arrow_right_y), text='False', text_left=False)

        self.gui_items.append(left_arrow)
        self.gui_items.append(right_arrow)

        if not type(node.left) == LeafNode:
            self.construct_subtree(node.left, left_child_pos_perc)
        else:
            self.show_leaf(node.left, left_child_pos_perc, child_level_pos, horizontal_layout=self.horizontal_layout)
        if not type(node.right) == LeafNode:
            self.construct_subtree(node.right, right_child_pos_perc)
        else:
            self.show_leaf(node.right, right_child_pos_perc, child_level_pos, horizontal_layout=self.horizontal_layout)

    def hide(self):
        self.showing = False

    def show_tree(self):
        self.construct_page()

    def show(self):
        self.showing = True
        self.screen.fill('white')
        self.show_tree()
        for item in self.gui_items:
            item.show()
        for item in self.gui_items:
            item.show_children()

    def process_event(self, event):
        for gui_item in self.gui_items:
            undo_key_combo_pressed = False
            if event.type == pygame.KEYDOWN:
                # check if ctrl z was pressed
                keys = pygame.key.get_pressed()
                if keys[pygame.K_z] and (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]):
                    undo_key_combo_pressed = True
            result_signal, _ = gui_item.process_event(event)
            if result_signal == 'new_tree':
                for i in range(len(self.settings.options_menus_per_domain[self.domain_idx])):
                    self.settings.options_menus_per_domain[self.domain_idx][i] = False
                self.env_wrapper.current_policy = self.current_policy
                self.current_tree_copy = copy.deepcopy(self.current_policy)
                self.decision_tree_history += [self.current_tree_copy]
                self.show_tree()
            elif result_signal == 'Undo' or (
                    undo_key_combo_pressed and (time.time() - self.time_since_last_undo > 0.2)):
                if len(self.decision_tree_history) > 1:
                    for i in range(len(self.settings.options_menus_per_domain[self.domain_idx])):
                        self.settings.options_menus_per_domain[self.domain_idx][i] = False
                    self.current_policy = self.decision_tree_history[-2]
                    self.env_wrapper.current_policy = self.current_policy
                    self.decision_tree_history = self.decision_tree_history[:-2]
                    self.current_tree_copy = copy.deepcopy(self.current_policy)
                    self.decision_tree_history += [self.current_tree_copy]
                    self.show_tree()
                    self.time_since_last_undo = time.time()
                    # wait a bit so that the undo key combo is not registered again
            elif result_signal == 'Reset':
                for i in range(len(self.settings.options_menus_per_domain[self.domain_idx])):
                    self.settings.options_menus_per_domain[self.domain_idx][i] = False
                self.current_policy = self.decision_tree_history[0]
                self.env_wrapper.current_policy = self.current_policy
                self.current_tree_copy = copy.deepcopy(self.current_policy)
                self.decision_tree_history = [self.current_tree_copy]
                self.show_tree()
        return True

    def process_standby(self):
        self.screen.fill('white')
        # for item in self.gui_items:
        #     item.show()
        node_showing, node_idx = self.settings.check_if_options_menu_open(domain_idx=self.domain_idx)
        if node_showing:
            for item in self.gui_items:
                if type(item) == GUIDecisionNodeDT or type(item) == GUIActionNodeDT:
                    item.process_standby()
                    item.drop_down_only = True
                    item.show_children()
                    item.drop_down_only = False
        else:
            for item in self.gui_items:
                item.process_standby()
            for item in self.gui_items:
                item.show_children()

