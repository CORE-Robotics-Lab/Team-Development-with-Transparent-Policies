import math
import os
import pygame
import cv2
import time
import copy
import torch
from ICCT.icct.interactive.pygame_gui_utils import GUIActionNodeICCT, GUIActionNodeIDCT, GUIDecisionNode, Arrow, Legend

class Node:
    def __init__(self, idx: int, node_depth: int, is_leaf: bool=False, left_child=None, right_child=None):
        self.idx = idx
        self.node_depth = node_depth
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf


class ICCTVisualizer:
    def __init__(self, icct, env_name, is_continuous_actions: bool = True):
        self.tree = icct
        self.is_continuous_actions = is_continuous_actions
        self.env_name = env_name
        
        pygame.init()
        self.X, self.Y = 1800, 800
        self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA)

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
            self.env_feat_names = ['position', 'linear velocity', 'vertical angle', 'angular velocity']
            self.action_names = ['Move Left', 'Move Right']
            self.n_actions = 1
            self.is_continuous_actions = False

    def extract_decision_nodes_info(self):
        weights = torch.abs(self.tree.layers.cpu())
        onehot_weights = self.tree.diff_argmax(weights)
        divisors = (weights * onehot_weights).sum(-1).unsqueeze(-1)
        divisors_filler = torch.zeros(divisors.size()).to(divisors.device)
        divisors_filler[divisors == 0] = 1
        divisors = divisors + divisors_filler

        self.impactful_vars_for_nodes = (onehot_weights.argmax(axis=1)).numpy()
        self.compare_sign = (self.tree.alpha.cpu() * self.tree.layers.cpu()) > 0
        self.new_weights = self.tree.layers.cpu() * onehot_weights / divisors
        self.comparators = self.tree.comparators.cpu() / divisors

    def extract_action_leaves_info(self):

        if self.is_continuous_actions:
            w = self.tree.sub_weights.cpu()

            # These 4 lines below are not strictly necessary but keep python from thinking
            # there is a possibility for a unassigned variable
            onehot_weights = self.tree.diff_argmax(torch.abs(w))
            new_w = self.tree.diff_argmax(torch.abs(w))
            new_s = (self.tree.sub_scalars.cpu() * onehot_weights).sum(-1).unsqueeze(-1)
            new_b = (self.tree.sub_biases.cpu() * onehot_weights).sum(-1).unsqueeze(-1)

            for i in range(len(self.tree.sub_weights.cpu())):
                if not i == 0:
                    w = w - w * onehot_weights

                # onehot_weights: [num_leaves, output_dim, input_dim]
                onehot_weights = self.tree.diff_argmax(torch.abs(w))

                # new_w: [num_leaves, output_dim, input_dim]
                # new_s: [num_leaves, output_dim, 1]
                # new_b: [num_leaves, output_dim, 1]
                new_w = onehot_weights
                new_s = (self.tree.sub_scalars.cpu() * onehot_weights).sum(-1).unsqueeze(-1)
                new_b = (self.tree.sub_biases.cpu() * onehot_weights).sum(-1).unsqueeze(-1)

            action_log_stds = torch.exp(self.tree.action_stds.detach().cpu())

            self.action_stds = torch.exp(action_log_stds).numpy()
            self.action_node_vars = (new_w.argmax(axis=0)).numpy()
            self.action_scalars = new_s.squeeze().detach().numpy()
            self.action_biases = new_b.squeeze().detach().numpy()

        self.leaves = self.tree.leaf_init_information

    def extract_path_info(self):
        def find_root(leaves):
            root_node = 0
            nodes_in_leaf_path = []
            for leaf in leaves:
                nodes_in_leaf_path.append((leaf[1][0] + leaf[1][1]))
            for node in nodes_in_leaf_path[0]:
                found_root = True
                for nodes in nodes_in_leaf_path:
                    if node not in nodes:
                        found_root = False
                if found_root:
                    root_node = node
                    break
            return root_node

        leaves_with_idx = copy.deepcopy([(leaf_idx, self.leaves[leaf_idx]) for leaf_idx in range(len(self.leaves))])
        self.root = Node(find_root(leaves_with_idx), 0)

        def find_children(node, leaves, current_depth):
            # dfs
            left_subtree = [leaf for leaf in leaves if node.idx in leaf[1][0]]
            right_subtree = [leaf for leaf in leaves if node.idx in leaf[1][1]]

            for _, leaf in left_subtree:
                leaf[0].remove(node.idx)
            for _, leaf in right_subtree:
                leaf[1].remove(node.idx)

            left_child_is_leaf = len(left_subtree) == 1
            right_child_is_leaf = len(right_subtree) == 1


            if not left_child_is_leaf:
                left_child = find_root(left_subtree)
            else:
                left_child = left_subtree[0][0]
            if not right_child_is_leaf:
                right_child = find_root(right_subtree)
            else:
                right_child = right_subtree[0][0]

            left_child = Node(left_child, current_depth, left_child_is_leaf)
            right_child = Node(right_child, current_depth, right_child_is_leaf)
            node.left_child = left_child
            node.right_child = right_child

            if not left_child_is_leaf:
                find_children(left_child, left_subtree, current_depth + 1)
            if not right_child_is_leaf:
                find_children(right_child, right_subtree, current_depth + 1)

        find_children(self.root, leaves_with_idx, current_depth=1)


    def construct_gui(self, interact: bool = False):

        self.screen.fill('white')
        interactable_gui_items = []

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

        action_node_texts = [get_action_leaf_text(leaf_idx) for leaf_idx in range(len(self.leaves))]

        y_spacing = 200

        decision_node_color = (137, 207, 240, 128)
        decision_node_border_color = (137, 207, 240, 255)
        action_leaf_color = (240, 128, 101, 128)
        action_leaf_border_color = (240, 128, 101, 255)

        # decision_node_size_x = 370
        decision_node_size_x = 335
        decision_node_size_y = 80
        decision_node_size = (decision_node_size_x, decision_node_size_y)

        # action_leaf_size_x = 220
        action_leaf_size_x = 180
        action_leaf_size_y = 80
        action_leaf_size = (action_leaf_size_x, action_leaf_size_y)

        def draw_leaf(leaf: Node, leaf_x_pos_perc: float, leaf_y_pos: float):
            for i in range(self.n_actions):
                node_position = ((leaf_x_pos_perc * self.X) - (action_leaf_size_x // 2), leaf_y_pos + i * (action_leaf_size_y + 20))
                if self.is_continuous_actions:
                    name = self.action_names[i]
                    node = GUIActionNodeICCT(self.tree, self.screen, node_position, size = action_leaf_size, font_size=14, name=name,
                                             text=action_node_texts[leaf.idx][i],
                                             rect_color = action_leaf_color, border_color = action_leaf_border_color, border_width = 3)
                else:
                    logits = self.leaves[leaf.idx][2]
                    action_idx = logits.index(max(logits))
                    node = GUIActionNodeIDCT(self.tree, self.screen, node_position, size = action_leaf_size, font_size=14,
                                             leaf_idx=leaf.idx, action_idx=action_idx, actions_list=self.action_names,
                                             rect_color = action_leaf_color, border_color = action_leaf_border_color,
                                             border_width = 3)
                interactable_gui_items.append(node)

        def draw_subtree_nodes(node: Node, node_x_pos_perc: float):
            depth = node.node_depth
            node_y_pos = y_spacing * depth + 50
            child_y_pos = y_spacing * (depth + 1) + 50


            node_var_idx = self.impactful_vars_for_nodes[node.idx]
            compare_sign = '>' if self.compare_sign[node.idx][node_var_idx] else '<'
            comparator_value = self.comparators.detach().numpy()[node.idx][0]

            node_position = ((node_x_pos_perc * self.X) - (decision_node_size_x // 2), node_y_pos)
            gui_node = GUIDecisionNode(self.tree, node.idx, self.env_feat_names, self.screen,
                                       node_position, size = decision_node_size, font_size=24,
                                       variable_idx=node_var_idx, compare_sign=compare_sign,
                                       comparator_value=str(round(comparator_value, 2)),
                                       rect_color = decision_node_color, border_color = decision_node_border_color,
                                       border_width = 3)
            interactable_gui_items.append(gui_node)

            left_child_x_pos_perc = node_x_pos_perc - (1 / 2**(depth + 2))
            right_child_x_pos_perc = node_x_pos_perc + (1 / 2**(depth + 2))

            left_arrow = Arrow(self.screen, pygame.Vector2(node_x_pos_perc * self.X, node_y_pos + decision_node_size_y),
                                    pygame.Vector2(left_child_x_pos_perc * self.X, child_y_pos))
            right_arrow = Arrow(self.screen, pygame.Vector2(node_x_pos_perc * self.X, node_y_pos + decision_node_size_y),
                                    pygame.Vector2(right_child_x_pos_perc * self.X, child_y_pos))

            interactable_gui_items.append(left_arrow)
            interactable_gui_items.append(right_arrow)


            if not node.left_child.is_leaf:
                draw_subtree_nodes(node.left_child, left_child_x_pos_perc)
            else:
                draw_leaf(node.left_child, left_child_x_pos_perc, child_y_pos)
            if not node.right_child.is_leaf:
                draw_subtree_nodes(node.right_child, right_child_x_pos_perc)
            else:
                draw_leaf(node.right_child, right_child_x_pos_perc, child_y_pos)

        draw_subtree_nodes(self.root, node_x_pos_perc = 1 / 2)



        leg = Legend(self.screen, 1400, 50, 130, 40, decision_node_color, action_leaf_color,
                     decision_node_border_color,  action_leaf_border_color, None, None,
                 [], selected=-1, transparent=True)
        interactable_gui_items.append(leg)

        restart = False

        if interact:
            pygame.display.flip()
            clock = pygame.time.Clock()
            is_running = True
            while is_running:
                for event in pygame.event.get():
                    if is_running:
                        if event.type == pygame.QUIT:
                            is_running = False
                            break
                        for gui_item in interactable_gui_items:
                            result_signal, result = gui_item.process_event(event)
                            if result_signal == 'new_tree':
                                self.tree = result
                                is_running = False
                                restart = True
                                break
                            elif result_signal == 'end':
                                is_running = False
                                break
                            # otherwise, we can continue!

                if not is_running:
                    break

                self.screen.fill('white')
                for gui_item in interactable_gui_items:
                    gui_item.draw()
                for gui_item in interactable_gui_items:
                    gui_item.draw_children()

                pygame.display.flip()
                clock.tick(30)

        if restart:
            self.modifiable_gui()


    def export_gui(self, filename):
        self.extract_decision_nodes_info()
        self.extract_action_leaves_info()
        self.extract_path_info()
        self.construct_gui(interact=False)
        pygame.image.save(self.screen, filename)

    def modifiable_gui(self):
        self.extract_decision_nodes_info()
        self.extract_action_leaves_info()
        self.extract_path_info()
        self.construct_gui(interact=True)
