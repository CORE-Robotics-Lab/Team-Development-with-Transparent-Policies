import numpy as np
import pygame
import torch

from ipm.gui.page_components import GUIButton, OptionBox, Multiplier
from ipm.gui.tree_gui_utils import Node, TreeInfo
from ipm.gui.page_components import GUIActionNodeICCT, GUIActionNodeIDCT, GUIDecisionNode, Arrow, Legend
from ipm.gui.env_rendering import render_cartpole
import gym
from abc import ABC, abstractmethod
from ipm.gui.policy_utils import finetune_model
from ipm.gui.overcooked_page import OvercookedGameDemo
import sys
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from ipm.overcooked.overcooked import OvercookedSelfPlayEnv
from overcooked_ai.src.overcooked_ai_py.agents.agent import RandomAgent, AgentPair

def get_button(screen, button_size, pos, button_text, button_fn):
    # surface: pygame.Surface, position: tuple, size: tuple, event_fn: Callable,
    return GUIButton(surface=screen, position=pos, event_fn=button_fn,
                     size=button_size, text=button_text, rect_color=(240, 240, 240),
                     text_color='black',
                     transparent=False,
                     border_color=(0, 0, 0), border_width=3)


class GUIPage(ABC):
    def __init__(self):
        self.X, self.Y = 1800, 800
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
                 bottom_left_fn = None, bottom_right_fn = None):
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

    def show(self):
        self.screen.fill('white')
        self.screen.blit(self.text_render, self.text_render.get_rect(center=self.screen.get_rect().center))

        self.gui_items = []

        if self.bottom_left_button:
            self.gui_items.append(get_button(self.screen, self.button_size, self.bottom_left_pos, 'Previous', self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(get_button(self.screen, self.button_size, self.bottom_right_pos, 'Next', self.bottom_right_fn))

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


class OvercookedPage(GUIPage):
    def __init__(self, screen, tree_page, text, font_size, bottom_left_button=False, bottom_right_button=False,
                 bottom_left_fn = None, bottom_right_fn = None):
        GUIPage.__init__(self)
        self.screen = screen
        self.text = text
        self.tree_page = tree_page
        self.main_font = pygame.font.Font('freesansbold.ttf', font_size)
        self.text_render = self.main_font.render(text, True, (255, 255, 255))
        self.bottom_left_button = bottom_left_button
        self.bottom_right_button = bottom_right_button
        self.bottom_left_fn = bottom_left_fn
        self.bottom_right_fn = bottom_right_fn

    def show(self):
        demo = OvercookedGameDemo(self.screen, self.tree_page.tree)
        demo.play_game_with_human()

    def process_event(self, event):
        self.bottom_right_fn()
        return True

    def process_standby(self):
        pass


class TreeCreationPage:
    def __init__(self, tree, env_name='overcooked', screen=None, X=None, Y=None, is_continuous_actions: bool = True,
                 bottom_left_button = False, bottom_right_button = False, bottom_left_fn = None, bottom_right_fn = None):
        self.tree = tree
        self.is_continuous_actions = is_continuous_actions
        self.env_name = env_name

        if X is None:
            self.X = 1800
        else:
            self.X = X
        if Y is None:
            self.Y = 800
        else:
            self.Y = Y
        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA)
        else:
            self.screen = screen

        if self.env_name == 'lunar':
            self.env_feat_names = ['x coordinate', 'y coordinate', 'horizontal velocity','vertical velocity',
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
            self.env_feat_names = ['Facing Up',
                                   'Facing Down',
                                    'Facing Right',
                                    'Facing Left',
                                    'Holding Onion',
                                    'Holding Soup',
                                    'Holding Dish',
                                    'Holding Tomato',
                                    'Closest Soup # Onions',
                                    'Closest Soup # Tomatoes',
                                    'Closest Pot Is Cooking',
                                    'Closest Pot Is Ready',
                                    'Closest Pot # Onions',
                                    'Closest Pot # Tomatoes',
                                    'Closest Pot Cook Time',
                                    '2nd Closest Pot Is Cooking',
                                    '2nd Closest Pot Is Ready',
                                    '2nd Closest Pot # Onions',
                                    '2nd Closest Pot # Tomatoes',
                                    '2nd Closest Pot Cook Time',
                                    'Player X Position',
                                    'Player Y Position']
            assert len(self.env_feat_names) == 22

            self.action_names = ['Move Up', 'Move Down', 'Move Right', 'Move Left', 'Stay', 'Interact',
                                 'Get Onion', 'Get Tomato', 'Get Dish', 'Serve Dish', 'Bring to Pot', 'Place on Counter']
            self.n_actions = 1 # we only take 1 action at a time
            self.is_continuous_actions = False

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
            node_position = ((leaf_x_pos_perc * self.X) - (self.action_leaf_size_x // 2), leaf_y_pos + i * (self.action_leaf_size_y + 20))
            if self.is_continuous_actions:
                name = self.action_names[i]
                node = GUIActionNodeICCT(self.tree, self.screen, node_position, size = self.action_leaf_size, font_size=14, name=name,
                                         text=self.action_node_texts[leaf.idx][i],
                                         rect_color = self.action_leaf_color, border_color = self.action_leaf_border_color, border_width = 3)
            else:
                logits = list(self.tree_info.leaves[leaf.idx][2])
                action_idx = logits.index(max(logits))
                node = GUIActionNodeIDCT(self.tree, self.screen, node_position, size = self.action_leaf_size, font_size=14,
                                         leaf_idx=leaf.idx, action_idx=action_idx, actions_list=self.action_names,
                                         rect_color = self.action_leaf_color, border_color = self.action_leaf_border_color,
                                         border_width = 3)
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
        self.construct_subtree(self.tree_info.root, node_x_pos_perc =1 / 2)

    def construct_subtree(self, node: Node, node_x_pos_perc: float):

        depth = node.node_depth
        node_y_pos = self.y_spacing * depth + 50
        child_y_pos = self.y_spacing * (depth + 1) + 50


        node_var_idx = self.tree_info.impactful_vars_for_nodes[node.idx]
        compare_sign = '>' if self.tree_info.compare_sign[node.idx][node_var_idx] else '<'
        comparator_value = self.tree_info.comparators.detach().numpy()[node.idx][0]

        node_position = ((node_x_pos_perc * self.X) - (self.decision_node_size_x // 2), node_y_pos)
        gui_node = GUIDecisionNode(self.tree, node.idx, self.env_feat_names, self.screen,
                                   node_position, size = self.decision_node_size, font_size=24,
                                   variable_idx=node_var_idx, compare_sign=compare_sign,
                                   comparator_value=str(round(comparator_value, 2)),
                                   rect_color = self.decision_node_color, border_color = self.decision_node_border_color,
                                   border_width = 3)
        self.gui_items.append(gui_node)

        left_child_x_pos_perc = node_x_pos_perc - (1 / 2**(depth + 2))
        right_child_x_pos_perc = node_x_pos_perc + (1 / 2**(depth + 2))

        left_arrow = Arrow(self.screen, pygame.Vector2(node_x_pos_perc * self.X, node_y_pos + self.decision_node_size_y),
                                pygame.Vector2(left_child_x_pos_perc * self.X, child_y_pos))
        right_arrow = Arrow(self.screen, pygame.Vector2(node_x_pos_perc * self.X, node_y_pos + self.decision_node_size_y),
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

    def get_performance(self, model):
        if self.env_name == 'cartpole':
            env = gym.make('CartPole-v1')
            current_episode = 0
            NUM_EPISODES = 1
            all_rewards = []
            total_reward = 0
            obs = env.reset()
            while current_episode < NUM_EPISODES:
                action = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    obs = env.reset()
                    current_episode += 1
                    all_rewards.append(total_reward)
                    total_reward = 0
            return np.mean(all_rewards)
        elif self.env_name == 'overcooked':
            env = OvercookedSelfPlayEnv(layout_name='forced_coordination')
            return 10.0

            egocentric_agent = model
            # TODO: Replace with BC model or some other model
            other_agent = RandomAgent()

            current_episode = 0
            NUM_EPISODES = 1
            all_rewards = []
            total_reward = 0
            obs = env.reset()
            all_actions = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), 'interact']
            while current_episode < NUM_EPISODES:
                ego_action = egocentric_agent.predict(obs[0])
                other_action = other_agent.action(obs[1])[0]
                joint_action = (all_actions[ego_action], other_action)
                obs, reward, done, info = env.step(joint_action)
                total_reward += reward
                if done:
                    obs = env.reset()
                    current_episode += 1
                    all_rewards.append(total_reward)
                    total_reward = 0
            return np.mean(all_rewards)
        else:
            raise NotImplementedError

    def get_finetuned_performance(self, initial_model):
        model = finetune_model(initial_model, env_wrapper=self.env_wrapper)
        return self.get_performance(model)

    def show(self):
        initial_perf = round(self.get_performance(self.tree_page.tree), 2)
        finetuned_perf = round(self.get_finetuned_performance(self.tree_page.tree), 2)
        self.text = 'Your tree\'s performance on ' + 'overcooked' + ': ' + str(initial_perf)
        self.text_render = self.main_font.render(self.text, True, (0, 0, 0))
        self.improved_text = 'Your teammate believe they found an improved policy with performance: ' + str(finetuned_perf)
        self.improved_text_render = self.main_font.render(self.improved_text, True, (0, 0, 0))

        self.screen.fill('white')
        center_x, center_y = self.screen.get_rect().center
        spacing = 100
        self.screen.blit(self.text_render, self.text_render.get_rect(center=(center_x, center_y - spacing)))
        self.screen.blit(self.improved_text_render, self.improved_text_render.get_rect(center=(center_x, center_y + spacing)))

        self.gui_items = []

        if self.bottom_left_button:
            self.gui_items.append(get_button(self.screen, self.button_size, self.bottom_left_pos, 'Previous', self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(get_button(self.screen, self.button_size, self.bottom_right_pos, 'Next', self.bottom_right_fn))

        for item in self.gui_items:
            item.show()

        self.showing = True

class EnvRewardModificationPage(GUIPageCenterText):
    def __init__(self, env_wrapper, screen, X, Y, font_size, bottom_left_button=False,
                 bottom_right_button=False, bottom_left_fn=None, bottom_right_fn=None):
        self.screen = screen
        self.X = X
        self.Y = Y
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
        self.screen.blit(self.text_render_item_in_pot, self.text_render_item_in_pot.get_rect(center=(center_x, center_y - spacing + y_offset)))
        self.screen.blit(self.text_render_item_pickup_dish, self.text_render_item_pickup_dish.get_rect(center=(center_x, center_y + y_offset)))
        self.screen.blit(self.text_render_item_pickup_soup, self.text_render_item_pickup_soup.get_rect(center=(center_x, center_y + spacing + y_offset)))

    def show(self):
        self.screen.fill('white')
        self.show_text()
        center_x, center_y = self.screen.get_rect().center
        spacing = 200
        y_offset = -100

        self.gui_items = []

        if self.bottom_left_button:
            self.gui_items.append(get_button(self.screen, self.button_size, self.bottom_left_pos, 'Previous', self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(get_button(self.screen, self.button_size, self.bottom_right_pos, 'Next', self.bottom_right_fn))

        self.node_box_item_in_pot = Multiplier(self.env_wrapper, 0, self.screen,
                                                (center_x + 50, center_y - spacing + y_offset - 15))
        self.node_box_pickup_dish = Multiplier(self.env_wrapper, 1, self.screen,
                                                (center_x + 50, center_y + y_offset - 15))
        self.node_box_pickup_soup = Multiplier(self.env_wrapper, 2, self.screen,
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
            action = self.tree_page.tree.predict(obs)
            render_cartpole(env)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                curr_ep += 1
