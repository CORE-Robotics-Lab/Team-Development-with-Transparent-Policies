import pygame
from ipm.gui.page_components import GUIButton
from ipm.gui.tree_gui_utils import Node, TreeInfo
from ipm.gui.page_components import GUIActionNodeICCT, GUIActionNodeIDCT, GUIDecisionNode, Arrow, Legend
from ipm.gui.env_rendering import render_cartpole
import gym
from abc import ABC, abstractmethod

def get_button(screen, button_size, pos, button_text, button_fn):
    # surface: pygame.Surface, position: tuple, size: tuple, event_fn: Callable,
    return GUIButton(surface=screen, position=pos, event_fn=button_fn,
                     size=button_size, text=button_text, rect_color=(200, 200, 200),
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
    def process_event(self):
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


class GUIOvercookedPage(GUIPage):
    def __init__(self, screen, text, font_size, bottom_left_button=False, bottom_right_button=False,
                 bottom_left_fn = None, bottom_right_fn = None):
        GUIPage.__init__(self)
        self.screen = screen
        self.text = text
        self.main_font = pygame.font.Font('freesansbold.ttf', font_size)
        self.text_render = self.main_font.render(text, True, (255, 255, 255))
        self.bottom_left_button = bottom_left_button
        self.bottom_right_button = bottom_right_button
        self.bottom_left_fn = bottom_left_fn
        self.bottom_right_fn = bottom_right_fn

    def show(self):
        pass


class TreeCreationPage:
    def __init__(self, tree, env_name, screen=None, X=None, Y=None, is_continuous_actions: bool = True,
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
            self.env_feat_names = ['position', 'linear velocity', 'vertical angle', 'angular velocity']
            self.action_names = ['Move Left', 'Move Right']
            self.n_actions = 1
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
        self.decision_node_size_x = 335
        self.decision_node_size_y = 80
        self.decision_node_size = (self.decision_node_size_x, self.decision_node_size_y)

        # action_leaf_size_x = 220
        self.action_leaf_size_x = 180
        self.action_leaf_size_y = 80
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
                logits = self.tree_info.leaves[leaf.idx][2]
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
            logits = self.tree_info.leaves[leaf_idx][2]
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

class EnvPage:
    def __init__(self, model, env_name, screen, X, Y):
        self.N_TOTAL_SAMPLES = 1000
        self.CURRENT_NUM_SAMPLES = 0
        self.screen = screen
        self.X = X
        self.Y = Y
        self.env_name = env_name
        self.model = model

    def show(self):
        if self.env_name == 'cartpole':
            env = gym.make('CartPole-v1')
        else:
            raise NotImplementedError

        actual_env = env.env.env
        actual_env.screen = self.screen
        actual_env.screen_width = self.X
        actual_env.screen_height = self.Y

        CURRENT_NUM_SAMPLES = 0

        obs = env.reset()
        while self.CURRENT_NUM_SAMPLES < self.N_TOTAL_SAMPLES:
            action = self.model.predict(obs)
            render_cartpole(env)
            obs, reward, done, info = env.step(action)
            CURRENT_NUM_SAMPLES += 1
            if done:
                obs = env.reset()

