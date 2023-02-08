from sys import int_info
import pygad
import numpy

import gym
import numpy as np
import random
import tqdm
import time
import torch
from ipm.models.idct import IDCT
from ipm.gui.tree_gui_utils import TreeInfo


def convert_dt_decision_to_leaf(decision_tree, node):
    parent = decision_tree.root
    q = [parent]
    while q:
        left_child = parent.left
        right_child = parent.right
        if left_child is node:
            parent.left = Leaf(action=random.randint(0, decision_tree.num_actions), idx=None)
            break
        else:
            q.append(left_child)
        if right_child is node:
            parent.right = Leaf(action=random.randint(0, decision_tree.num_actions), idx=None)
            break
        else:
            q.append(right_child)
    # value array will be broken
    return decision_tree


def convert_dt_leaf_to_decision(decision_tree, node):
    random_leaf1 = Leaf(action=random.randint(0, decision_tree.num_actions), idx=None)
    random_leaf2 = Leaf(action=random.randint(0, decision_tree.num_actions), idx=None)
    var_idx = random.randint(0, decision_tree.num_vars)

    parent = decision_tree.root
    q = [parent]
    while q:
        left_child = parent.left
        right_child = parent.right
        if left_child is node:
            parent.left = BranchingNode(var_idx=var_idx, comp_val=0.5, left=random_leaf1, right=random_leaf2, idx=None, is_root=False)
            break
        else:
            q.append(left_child)
        if right_child is node:
            parent.right = BranchingNode(var_idx=var_idx, comp_val=0.5, left=random_leaf1, right=random_leaf2, idx=None,
                                        is_root=False)
            break
        else:
            q.append(right_child)
    # value array will be broken
    return decision_tree


def decision_tree_to_sparse_ddt(tree):

    raise NotImplementedError

def sparse_ddt_to_decision_tree(tree: IDCT, env):
    tree_info = TreeInfo(tree)

    lows = env.observation_space.low
    highs = env.observation_space.high

    n_decision_nodes = len(tree_info.impactful_vars_for_nodes)
    n_leaves = len(tree_info.leaves)

    values = []

    for node_idx in range(n_decision_nodes):
        node_var_idx = tree_info.impactful_vars_for_nodes[node_idx]
        # TODO: Check if not is required
        compare_sign = not tree_info.compare_sign.detach().numpy()[node_idx][node_var_idx]
        comparator_value = tree_info.comparators.detach().numpy()[node_idx][0]
        values.append(node_var_idx)
        values.append(compare_sign)

        # Let's scale the comparator value to [-1, 1]
        # if lows[node_var_idx] < -1e8:
        #     low = -1e8
        # else:
        #     low = lows[node_var_idx]
        # if highs[node_var_idx] > 1e8:
        #     high = 1e8
        # else:
        #     high = highs[node_var_idx]
        # scaled_comparator_value = (comparator_value - low) / (high - low) * 2 - 1
        values.append(comparator_value)

    for leaf_idx in range(n_leaves):
        logits = tree_info.leaves[leaf_idx][2]
        action_idx = np.argmax(logits)
        values.append(action_idx)

    node2node = np.zeros((n_decision_nodes, n_decision_nodes))
    node2leaf = np.zeros((n_decision_nodes, n_leaves))

    # (parent_idx, went_left, node)
    q = [(-1, True, tree_info.root)]
    while q:
        parent_idx, went_left, node = q.pop(0)
        if node.left_child:
            q.append((node.idx, True, node.left_child))
        if node.right_child:
            q.append((node.idx, False, node.right_child))

        # skip the root node since it doesn't have any parents
        if parent_idx == -1:
            continue

        if node.is_leaf:
            # we got a leaf so add parent and it to leaf adj matrix
            if went_left:
                node2leaf[parent_idx, node.idx] = 1
            else:
                node2leaf[parent_idx, node.idx] = 2
        else:
            # we got a node so add parent and it to node adj matrix
            if went_left:
                node2node[parent_idx, node.idx] = 1
            else:
                node2node[parent_idx, node.idx] = 2

    return DecisionTree(values, n_decision_nodes, n_leaves, lows, highs, node2node, node2leaf)


class Leaf:
    def __init__(self, action=None, idx=None, depth=0):
        self.action = action
        self.idx = idx
        self.depth = depth

    def predict(self, values):
        return self.action


class BranchingNode:
    def __init__(self, var_idx=None, comp_val=0.0, left=None, right=None, idx=None, is_root=False, depth=0):
        self.left = left
        self.right = right
        self.var_idx = var_idx
        self.comp_val = comp_val
        self.idx = idx
        self.is_root = is_root
        self.depth = depth

    def predict(self, values):
        if values[self.var_idx] <= self.comp_val:
            return self.left.predict(values)
        else:
            return self.right.predict(values)

# function that converts low level observations to high level observations
def higher_level_obs(obs, get_names=False):
    new_obs = []
    names = {} # key is index, value is (name, possible_values)
    # e.g. names[0] = ("direction_facing", ["left", "right", "up", "down"])

    # get argmax of first 4 features
    names[len(new_obs)] = (["Direction Facing"], ["Up", "Down", "Right", "Left"])
    direction_facing = np.argmax(obs[:4])
    new_obs.append(direction_facing)

    names[len(new_obs)] = (["Item Holding"], ["Onion", "Soup", "Dish", "Tomato"])
    item_holding = np.argmax(obs[4:8])
    new_obs.append(item_holding)

    # closest pot is cooking, ready, or needs more ingredients
    names[len(new_obs)] = (["Closest Pot Cooking"], ["Cooking", "Ready", "Needs More"])
    cooking = obs[8] == 1
    ready = obs[9] == 1
    needs_more_ingredients = obs[10] == 1
    if cooking:
        new_obs.append(0)
    elif ready:
        new_obs.append(1)
    elif needs_more_ingredients:
        new_obs.append(2)
    else:
        raise Exception("Closest pot is not cooking, ready, or needs more ingredients")

    names[len(new_obs)] = (["Closest Pot Almost Done"], ["True", "False"])
    new_obs.append(obs[11])

    # 2nd closest pot is cooking, ready, or needs more ingredients
    names[len(new_obs)] = (["2nd Closest Pot Cooking"], ["Cooking", "Ready", "Needs More"])
    cooking = obs[12] == 1
    ready = obs[13] == 1
    needs_more_ingredients = obs[14] == 1
    if cooking:
        new_obs.append(0)
    elif ready:
        new_obs.append(1)
    elif needs_more_ingredients:
        new_obs.append(2)
    else:
        raise Exception("2nd Closest pot is not cooking, ready, or needs more ingredients")

    # 2nd closest pot almost done
    names[len(new_obs)] = (["2nd Closest Pot Cooking"], ["Cooking", "Ready", "Needs More"])
    new_obs.append(obs[15])

    # x and y position
    names[len(new_obs)] = (["Player Position"], ["Up", "Middle", "Down"])
    assert 1 <= obs[17] <= 3
    new_obs.append(obs[17] - 1)

    # other agent (absolute) position
    names[len(new_obs)] = (["Other Player Position"], ["Up", "Middle", "Down"])
    assert 1 <= obs[19] <= 3
    new_obs.append(obs[19] - 1)

    # get argmax for next 4 features (direction facing)
    names[len(new_obs)] = (["Other Player Direction"], ["Up", "Down", "Right", "Left"])
    other_direction_facing = np.argmax(obs[20:24])
    new_obs.append(other_direction_facing)

    # get argmax for next 4 features (object holding)
    names[len(new_obs)] = (["Other Item Holding"], ["Up", "Down", "Right", "Left"])
    object_holding = np.argmax(obs[24:28])
    new_obs.append(object_holding)

    # may have another feature indicating the predicted action for other agent
    if len(obs) == 29:
        names[len(new_obs)] = (["Teammate is Going"], ["Up", "Down", "Right", "Left"])
        new_obs.append(obs[28])

    return np.array(new_obs)


class DecisionTree:
    def __init__(self, num_vars, num_actions, node_values=None, depth=3, var_names=None, action_names=None):
        self.random_tree = node_values is None
        if self.random_tree:
            self.node_values = []
        else:
            self.node_values = node_values
        self.gene_space = []
        self.depth = depth

        self.num_vars = num_vars
        self.num_actions = num_actions

        self.n_decision_nodes = 2 ** (depth + 1) - 1
        self.n_leaves = 2 ** (depth + 1)
        if not self.random_tree:
            assert len(self.node_values) == self.n_decision_nodes + self.n_leaves

        self.root = None
        self.construct_empty_full_tree()
        self.populate_values()
        assert len(self.node_values) == self.n_decision_nodes + self.n_leaves
        assert self.root is not None

    def construct_empty_full_tree(self):
        assert self.depth > 0
        self.root = BranchingNode(is_root=True, depth=0)
        q = [(1, self.root)]
        while q:
            current_depth, node = q.pop(0)
            if current_depth == self.depth + 1:
                node.left = Leaf(depth=current_depth)
                node.right = Leaf(depth=current_depth)
            elif current_depth < self.depth + 1:
                assert type(node) == BranchingNode
                node.left = BranchingNode(depth=current_depth)
                node.right = BranchingNode(depth=current_depth)
                q.append((current_depth + 1, node.left))
                q.append((current_depth + 1, node.right))

    def dfs_inorder(self, node):
        branch_node = type(node) == BranchingNode
        if branch_node and node.left is not None:
            self.dfs_inorder(node.left)
        if branch_node:
            if self.random_tree:
                node.var_idx = random.randint(0, self.num_vars - 1)
                self.node_values.append(node.var_idx)
            else:
                node.var_idx = self.node_values[self.current_node_idx]
            self.gene_space.append(list(range(self.num_vars)))
        else:
            if self.random_tree:
                node.action = random.randint(0, self.num_actions - 1)
                self.node_values.append(node.action)
            else:
                node.action = self.node_values[self.current_node_idx]
            self.gene_space.append(list(range(self.num_actions)))
        node.idx = self.current_node_idx
        self.current_node_idx += 1
        if branch_node and node.right is not None:
            self.dfs_inorder(node.right)

    @staticmethod
    def from_sklearn(sklearn_model, num_vars, num_actions):
        depth = sklearn_model.tree_.max_depth
        dt = DecisionTree(num_vars, num_actions, node_values=None, depth=depth)

        children_left = sklearn_model.tree_.children_left
        children_right = sklearn_model.tree_.children_right
        feature = sklearn_model.tree_.feature
        leaf_values = sklearn_model.tree_.value
        thresholds = sklearn_model.tree_.threshold

        stack = [(0, dt.root)]  # start with the root node id (0) and its depth (0) and the other trees node

        while len(stack) > 0:
            sklearn_node_id, node = stack.pop()

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[sklearn_node_id] != children_right[sklearn_node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                assert feature[sklearn_node_id] >= 0
                assert type(node) == BranchingNode
                node.var_idx = feature[sklearn_node_id]
                node.comp_val = thresholds[sklearn_node_id]
                dt.node_values[node.idx] = node.var_idx
                stack.append((children_left[sklearn_node_id], node.left))
                stack.append((children_right[sklearn_node_id], node.right))
            else:
                action_tree_idx = np.argmax(leaf_values[sklearn_node_id])
                action = sklearn_model.classes_[action_tree_idx]
                if type(node) == Leaf:
                    node.action = action
                    dt.node_values[node.idx] = node.action
                else:
                    # we set all descendants of this node to the same action
                    q = [node]
                    while q:
                        n = q.pop(0)
                        if type(n) == Leaf:
                            n.action = action
                            dt.node_values[n.idx] = n.action
                        else:
                            n.var_idx = random.randint(0, dt.num_vars - 1)
                            n.comp_val = 0.5
                            q.append(n.left)
                            q.append(n.right)
        return dt

    def populate_values(self):
        self.current_node_idx = 0
        self.dfs_inorder(self.root)

    def predict(self, values):
        return self.root.predict(values)

    def visualize(self):
        # TODO: fix code!
        q = [(self.root, 0)]
        while q:
            current_node, current_depth = q.pop(-1)
            output = ''
            for i in range(current_depth):
                output += '\t'
            if current_node is None:
                output += 'Else'
            elif current_depth < self.depth:
                output += 'if ' + self.var_names[current_node.var_idx]
                output += ' < ' if current_node.comparator == 0 else ' > '
                output += str(round(current_node.value, 2))
                left = current_node.left
                right = current_node.right
                q.append((right, current_depth + 1))
                if current_depth == self.depth - 1:
                    q.append((None, current_depth))
                q.append((left, current_depth + 1))
            else:
                action_idx = np.argmax(current_node.leaf_probs)
                output += self.action_names[action_idx]
            print(output)