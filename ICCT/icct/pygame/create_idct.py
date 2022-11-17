import math
import os
import pygame
import cv2
import time
import torch
from ICCT.icct.pygame.pygame_gui_utils import draw_arrow, GUIActionNode, GUIDecisionNode


class IDCTCreator:
    def __init__(self, icct, env_name):
        self.icct = icct
        self.env_name = env_name

        pygame.init()
        self.X, self.Y = 1800, 800
        self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA)

        if self.env_name == 'cartpole':
            self.env_feat_names = ['position', 'linear velocity', 'vertical angle', 'angular velocity']
            self.action_names = ['Move Left', 'Move Right']
            self.n_actions = 1
            self.is_continuous_actions = False

        self.weights = []
        self.alphas = []
        self.comparators = []
        self.leaves = []

    def construct_gui(self, interact: bool = False):

        self.screen.fill('white')
        interactable_gui_items = []

        def get_decision_node_text(node_idx: int, node_var: int, comp):
            variable_text = self.env_feat_names[node_var]
            compare_sign_text = '>' if self.compare_sign[node_idx][node_var] else '<'
            return variable_text + ' ' + compare_sign_text + ' ' + str(round(comp[0], 2))

        def get_action_leaf_text(leaf_idx: int):
            texts = []
            if self.is_continuous_actions:
                for i in range(self.n_actions):
                    variable_text = self.env_feat_names[self.action_node_vars[i, leaf_idx]]
                    if self.n_actions > 1:
                        scalar = self.action_scalars[leaf_idx][i]
                        bias = self.action_biases[leaf_idx][i]
                        std = self.action_stds[leaf_idx][i]
                    else:
                        scalar = self.action_scalars[leaf_idx]
                        bias = self.action_biases[leaf_idx]
                        std = self.action_stds[leaf_idx]
                    # text = 'N(' + str(round(scalar, 2)) + ' * ' + variable_text + ' + ' + str(round(bias, 2)) + ', ' + str(
                    #     round(std, 2)) + ')'
                    text = str(round(scalar, 2)) + ' * ' + variable_text + ' + ' + str(round(bias, 2))
                    texts.append(text)
            else:
                logits = self.leaves[leaf_idx][2]
                action_idx = logits.index(max(logits))
                text = self.action_names[action_idx]
                texts.append(text)
            return texts

        decision_node_texts = [get_decision_node_text(node_idx, node_var, comp) for node_idx, (node_var, comp) in
                               enumerate(zip(self.impactful_vars_for_nodes, self.comparators.detach().numpy()))]
        action_node_texts = [get_action_leaf_text(leaf_idx) for leaf_idx in range(len(self.leaves))]

        y_spacing = 125

        decision_node_color = (137, 207, 240, 128)
        decision_node_border_color = (137, 207, 240, 255)
        action_leaf_color = (240, 128, 101, 128)
        action_leaf_border_color = (240, 128, 101, 255)

        decision_node_size_x = 320
        decision_node_size_y = 60
        decision_node_size = (decision_node_size_x, decision_node_size_y)

        action_leaf_size_x = 180
        action_leaf_size_y = 50
        action_leaf_size = (action_leaf_size_x, action_leaf_size_y)

        def draw_decision_node(node_idx: int):
            level_idx = math.trunc(math.log(node_idx + 1, 2))
            n_nodes_so_far = 0
            for depth_level in range(level_idx):
                n_nodes_so_far += int(math.pow(2, depth_level))
            node_idx_in_level = int(node_idx - n_nodes_so_far)
            n_nodes_in_level = 2 ** level_idx
            node_x_pos_perc = (2 * node_idx_in_level + 1) / (2 * n_nodes_in_level)
            node_position = ((node_x_pos_perc * self.X) - (decision_node_size_x // 2), y_spacing * (level_idx + 1))

            node = GUIDecisionNode(self.icct, node_idx, self.env_feat_names, self.screen,
                                   node_position, size=decision_node_size, font_size=24,
                                   text=decision_node_texts[node_idx],
                                   rect_color=decision_node_color, border_color=decision_node_border_color,
                                   border_width=3)
            interactable_gui_items.append(node)

        def draw_action_leaves(leaf_idx: int):
            n_nodes_in_level = 2 ** self.depth
            node_x_pos_perc = (2 * leaf_idx + 1) / (2 * n_nodes_in_level)
            for i in range(self.n_actions):
                node_position = ((node_x_pos_perc * self.X) - (action_leaf_size_x // 2),
                                 y_spacing * (self.depth + 1) + i * (action_leaf_size_y + 20))
                if self.is_continuous_actions:
                    name = self.action_names[i]
                    node = GUIActionNode(self.icct, self.screen, node_position, size=action_leaf_size, font_size=14,
                                         name=name,
                                         text=action_node_texts[leaf_idx][i],
                                         rect_color=action_leaf_color, border_color=action_leaf_border_color,
                                         border_width=3)
                else:
                    node = GUIActionNode(self.icct, self.screen, node_position, size=action_leaf_size, font_size=14,
                                         name=None,
                                         text=action_node_texts[leaf_idx][i],
                                         rect_color=action_leaf_color, border_color=action_leaf_border_color,
                                         border_width=3)
                interactable_gui_items.append(node)

        for i in range(len(decision_node_texts)):
            draw_decision_node(i)

        for i in range(len(action_node_texts)):
            draw_action_leaves(i)

        for i in range(1, self.depth + 1):
            for j in range(1, 2 ** i, 2):
                node_pos_x_pos = j / 2 ** i
                left_node_x_pos = (j * 2 - 1) / 2 ** (i + 1)
                right_node_x_pos = (j * 2 + 1) / 2 ** (i + 1)
                for child_node_x_pos in [left_node_x_pos, right_node_x_pos]:
                    draw_arrow(self.screen,
                               pygame.Vector2(node_pos_x_pos * self.X, i * y_spacing + decision_node_size_y),
                               pygame.Vector2(child_node_x_pos * self.X, (i + 1) * y_spacing))

        if interact:
            pygame.display.flip()
            clock = pygame.time.Clock()
            is_running = True
            while is_running:
                # time_delta = clock.tick(60) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        is_running = False
                        break
                    for gui_item in interactable_gui_items:
                        is_running = gui_item.process_event(event)
                for gui_item in interactable_gui_items:
                    gui_item.process_standby()

            # pygame.display.flip()
            pygame.display.update()
            clock.tick(60)

    def export_gui(self, filename):
        self.construct_gui(interact=False)
        pygame.image.save(self.screen, filename)

    def start_gui(self):
        self.construct_gui(interact=True)
