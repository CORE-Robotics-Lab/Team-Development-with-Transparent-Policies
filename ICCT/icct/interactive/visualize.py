import math
import os
import pygame
import cv2
import time
import torch
from ICCT.icct.interactive.pygame_gui_utils import draw_arrow, GUIActionNode, GUIDecisionNode


class ICCTVisualizer:
    def __init__(self, icct, env_name, depth: int =3, is_continuous: bool = True):
        self.icct = icct
        self.is_continuous = is_continuous
        self.env_name = env_name
        
        pygame.init()
        self.X, self.Y = 1800, 800
        self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA)
        self.depth = depth

        if self.env_name == 'lunar':
            self.env_feat_names = ['x coordinate', 'y coordinate', 'horizontal velocity','vertical velocity',
                                   'orientation', 'angular velocity', 'left leg touching', 'right leg touching']
            self.action_names = ['Main Engine Thrust', 'Side Engines Net Thrust']

        elif self.env_name == 'cart':
            self.env_feat_names = ['position', 'vertical angle', 'linear velocity', 'angular velocity']
            self.action_names = ['Force applied']



    def extract_decision_nodes_info(self):
        weights = torch.abs(self.icct.layers.cpu())
        onehot_weights = self.icct.diff_argmax(weights)
        divisors = (weights * onehot_weights).sum(-1).unsqueeze(-1)
        divisors_filler = torch.zeros(divisors.size()).to(divisors.device)
        divisors_filler[divisors == 0] = 1
        divisors = divisors + divisors_filler

        self.impactful_vars_for_nodes = (onehot_weights.argmax(axis=1)).numpy()
        self.compare_sign = (self.icct.alpha.cpu() * self.icct.layers.cpu()) > 0
        self.new_weights = self.icct.layers.cpu() * onehot_weights / divisors
        self.comparators = self.icct.comparators.cpu() / divisors

    def extract_action_leaves_info(self):

        if self.is_continuous:
            w = self.icct.sub_weights.cpu()

            # These 4 lines below are not strictly necessary but keep python from thinking
            # there is a possibility for a unassigned variable
            onehot_weights = self.icct.diff_argmax(torch.abs(w))
            new_w = self.icct.diff_argmax(torch.abs(w))
            new_s = (self.icct.sub_scalars.cpu() * onehot_weights).sum(-1).unsqueeze(-1)
            new_b = (self.icct.sub_biases.cpu() * onehot_weights).sum(-1).unsqueeze(-1)

            for i in range(len(self.icct.sub_weights.cpu())):
                if not i == 0:
                    w = w - w * onehot_weights

                # onehot_weights: [num_leaves, output_dim, input_dim]
                onehot_weights = self.icct.diff_argmax(torch.abs(w))

                # new_w: [num_leaves, output_dim, input_dim]
                # new_s: [num_leaves, output_dim, 1]
                # new_b: [num_leaves, output_dim, 1]
                new_w = onehot_weights
                new_s = (self.icct.sub_scalars.cpu() * onehot_weights).sum(-1).unsqueeze(-1)
                new_b = (self.icct.sub_biases.cpu() * onehot_weights).sum(-1).unsqueeze(-1)

            action_log_stds = torch.exp(self.icct.action_stds.detach().cpu())

            self.action_stds = torch.exp(action_log_stds).numpy()
            self.action_node_vars = (new_w.argmax(axis=0)).numpy()
            self.action_scalars = new_s.squeeze().detach().numpy()
            self.action_biases = new_b.squeeze().detach().numpy()

    def construct_gui(self, interact: bool = False):

        self.screen.fill('white')
        interactable_gui_items = []

        def get_decision_node_text(node_idx: int, node_var: int, comp):
            variable_text = self.env_feat_names[node_var]
            compare_sign_text = '>' if self.compare_sign[node_idx][node_var] else '<'
            return variable_text + ' ' + compare_sign_text + ' ' + str(round(comp[0], 2))

        def get_action_leaf_text(leaf_idx: int):
            texts = []
            n_actions = len(self.action_names)
            for i in range(n_actions):
                variable_text = self.env_feat_names[self.action_node_vars[i, leaf_idx]]
                if n_actions > 1:
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
            return texts

        decision_node_texts = [get_decision_node_text(node_idx, node_var, comp) for node_idx, (node_var, comp) in enumerate(zip(self.impactful_vars_for_nodes, self.comparators.detach().numpy()))]
        action_node_texts = [get_action_leaf_text(leaf_idx) for leaf_idx in range(len(self.action_stds))]

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
                                   node_position, size = decision_node_size, font_size=24, text=decision_node_texts[node_idx],
                                rect_color = decision_node_color, border_color = decision_node_border_color, border_width = 3)
            interactable_gui_items.append(node)


        def draw_action_leaves(leaf_idx: int):
            n_nodes_in_level = 2 ** self.depth
            node_x_pos_perc = (2 * leaf_idx + 1) / (2 * n_nodes_in_level)
            n_actions = len(self.action_names)
            for i in range(n_actions):
                node_position = ((node_x_pos_perc * self.X) - (action_leaf_size_x // 2), y_spacing * (3 + 1) + i * (action_leaf_size_y + 20))
                name = self.action_names[i]
                node = GUIActionNode(self.icct, self.screen, node_position, size = action_leaf_size, font_size=14, name=name,
                                     text=action_node_texts[leaf_idx][i],
                                    rect_color = action_leaf_color, border_color = action_leaf_border_color, border_width = 3)
                interactable_gui_items.append(node)

        for i in range(len(decision_node_texts)):
            draw_decision_node(i)

        for i in range(len(action_node_texts) * len(self.action_names)):
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
        self.extract_decision_nodes_info()
        self.extract_action_leaves_info()
        self.construct_gui(interact=False)
        pygame.image.save(self.screen, filename)

    def modifiable_gui(self):
        self.extract_decision_nodes_info()
        self.extract_action_leaves_info()
        self.construct_gui(interact=True)


class IDCTVisualizer:
    def __init__(self, icct, env_name, depth: int = 3, is_continuous: bool = True):
        self.icct = icct
        self.is_continuous = is_continuous
        self.env_name = env_name

        pygame.init()
        self.X, self.Y = 1800, 800
        self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA)
        self.depth = depth

        if self.env_name == 'cartpole':
            self.env_feat_names = ['cart position', 'cart velocity', 'pole angle', 'pole angular velocity']

    def extract_decision_nodes_info(self):
        weights = torch.abs(self.icct.layers.cpu())
        onehot_weights = self.icct.diff_argmax(weights)
        divisors = (weights * onehot_weights).sum(-1).unsqueeze(-1)
        divisors_filler = torch.zeros(divisors.size()).to(divisors.device)
        divisors_filler[divisors == 0] = 1
        divisors = divisors + divisors_filler

        self.impactful_vars_for_nodes = (onehot_weights.argmax(axis=0)).numpy()
        self.compare_sign = (self.icct.alpha.cpu() * self.icct.layers.cpu()) > 0
        self.new_weights = self.icct.layers.cpu() * onehot_weights / divisors
        self.comparators = self.icct.comparators.cpu() / divisors

    def extract_action_leaves_info(self):
        pass


    def construct_gui(self, interact: bool = False):

        self.screen.fill('white')
        interactable_gui_items = []

        def get_decision_node_text(node_idx: int, node_var: int, comp):
            cartpole_feat_names = ['cart position', 'cart velocity', 'pole angle', 'pole angular velocity']

            if self.env_name == 'cartpole':
                variable_text = cartpole_feat_names[node_var]
            else:
                # If we don't know the environment, just call the feats X1...XN
                variable_text = 'X' + str(node_var + 1)

            compare_sign_text = '>' if self.compare_sign[node_idx][0] else '<'
            return variable_text + ' ' + compare_sign_text + ' ' + str(round(comp[0], 2))

        def get_action_leaf_text(leaf_idx):
            pass
            variable_text = 'X' + str(leaf_idx + 1)
            texts = []
            n_actions = len(self.action_stds[leaf_idx])
            for i in range(n_actions):
                scalar = self.action_scalars[leaf_idx][i]
                bias = self.action_biases[leaf_idx][i]
                std = self.action_stds[leaf_idx][i]
                text = 'N(' + str(round(scalar, 2)) + ' * ' + variable_text + ' + ' + str(round(bias, 2)) + ', ' + str(
                    round(std, 2)) + ')'
                texts.append(text)
            return texts

        decision_node_texts = [get_decision_node_text(node_idx, node_var, comp) for node_idx, (node_var, comp) in
                               enumerate(zip(self.impactful_vars_for_nodes, self.comparators.detach().numpy()))]
        # action_node_texts = [get_action_leaf_text(leaf_idx) for leaf_idx in range(len(self.action_stds))]

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
            return

            n_nodes_in_level = 2 ** self.depth
            node_x_pos_perc = (2 * leaf_idx + 1) / (2 * n_nodes_in_level)
            n_actions = 2
            for i in range(n_actions):
                node_position = (
                (node_x_pos_perc * self.X) - (action_leaf_size_x // 2), y_spacing * (3 + 1) + i * action_leaf_size_y)
                node = GUIActionNode(self.icct, self.screen, node_position, size=action_leaf_size, font_size=14,
                                     text=action_node_texts[leaf_idx][i],
                                     rect_color=action_leaf_color, border_color=action_leaf_border_color,
                                     border_width=3)
                interactable_gui_items.append(node)

        for i in range(3):
            draw_decision_node(i)

        # for i in range(8):
        #     draw_action_leaves(i)

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
        self.extract_decision_nodes_info()
        self.extract_action_leaves_info()
        self.construct_gui(interact=False)
        pygame.image.save(self.screen, filename)

    def modifiable_gui(self):
        self.extract_decision_nodes_info()
        self.extract_action_leaves_info()
        self.construct_gui(interact=True)
