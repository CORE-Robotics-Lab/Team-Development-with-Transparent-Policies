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
        action_idx = logits.index(max(logits))
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
    def __init__(self, leaf_probs):
        self.leaf_probs = leaf_probs

    def predict(self, values):
        return self.leaf_probs
        #return 1 if self.leaf_probs < random.random() else 0
        # return
        # return np.random.sample(range(len(self.leaf_probs)), p=self.leaf_probs)

class Node:
    def __init__(self, var_idx, comparator, value, lows, highs, left=None, right=None):
        self.left = left
        self.right = right
        self.var_idx = var_idx
        self.comparator = comparator

        if lows[self.var_idx] < -1e8:
            low = -10
        else:
            low = lows[self.var_idx]
        if highs[self.var_idx] > 1e8:
            high = 10
        else:
            high = highs[self.var_idx]

        self.value = value
        # self.value = (value - low) / (high - low)
        #self.value = (value + 1) / 2 * (high - low) + low

    def predict(self, values):
        if self.comparator == 0:
            if values[self.var_idx] < self.value:
                return self.left.predict(values)
            else:
                return self.right.predict(values)
        else:
            if values[self.var_idx] > self.value:
                return self.left.predict(values)
            else:
                return self.right.predict(values)

class DecisionTree:
    def __init__(self, node_values, n_decision_nodes, n_leaves, lows, highs,
                 node2node=None, node2leaf=None, var_names=None, action_names=None):
        # i = 0
        # self.root = Node(node_values[i], node_values[i+1], node_values[i+2], lows, highs)

        self.node_values = node_values

        decision_node_values = node_values[:len(node_values) - n_leaves]
        leaf_values = node_values[len(node_values) - n_leaves:]

        if node2node is None:
            depth = np.log2(n_decision_nodes + 1)
            node2node = np.zeros(shape=(n_decision_nodes, n_decision_nodes), dtype=np.int32)
            n_terminal_nodes = int(2 ** (depth - 1))
            for i in range(len(node2node) - n_terminal_nodes):
                node2node[i, 2 * i + 1] = 1
                node2node[i, 2 * i + 2] = 2

        if node2leaf is None:
            depth = np.log2(n_decision_nodes + 1)
            node2leaf = np.zeros(shape=(n_decision_nodes, n_leaves), dtype=np.int32)
            n_terminal_nodes = int(2 ** (depth - 1))
            for i in range(n_terminal_nodes):
                node_idx = len(node2node) - n_terminal_nodes + i - 2
                node2leaf[node_idx, 2 * i] = 1
                node2leaf[node_idx, 2 * i + 1] = 2

        nodes = [Node(decision_node_values[i], decision_node_values[i + 1],
                      decision_node_values[i + 2], lows, highs) for i in range(0, len(decision_node_values), 3)]
        for i in range(n_decision_nodes):
            for j in range(n_decision_nodes):
                if node2node[i, j] == 1:
                    nodes[i].left = nodes[j]
                elif node2node[i, j] == 2:
                    nodes[i].right = nodes[j]

        leaves = [Leaf(leaf_values[i]) for i in range(len(leaf_values))]
        for i in range(n_decision_nodes):
            for j in range(n_leaves):
                if node2leaf[i, j] == 1:
                    nodes[i].left = leaves[j]
                elif node2leaf[i, j] == 2:
                    nodes[i].right = leaves[j]

        self.root = nodes[0]

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