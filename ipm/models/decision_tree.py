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
    def __init__(self, action=None, idx=None):
        self.action = action
        self.idx = idx

    def predict(self, values):
        return self.action


class BranchingNode:
    def __init__(self, var_idx=None, left=None, right=None, idx=None):
        self.left = left
        self.right = right
        self.var_idx = var_idx
        self.idx = idx

    def predict(self, values):
        if values[self.var_idx] == 0.0:
            return self.left.predict(values)
        else:
            return self.right.predict(values)


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
        self.root = BranchingNode()
        q = [(0, self.root)]
        while q:
            current_depth, node = q.pop(0)
            if current_depth == self.depth:
                node.left = Leaf()
                node.right = Leaf()
            elif current_depth < self.depth:
                assert type(node) == BranchingNode
                node.left = BranchingNode()
                node.right = BranchingNode()
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