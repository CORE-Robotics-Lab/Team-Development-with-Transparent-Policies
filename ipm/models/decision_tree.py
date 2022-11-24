from sys import int_info
import pygad
import numpy

import gym
import numpy as np
import random
import tqdm
import time


class Leaf:
    def __init__(self, leaf_probs):
        self.leaf_probs = leaf_probs

    def predict(self, values):
        return np.random.sample(range(len(self.leaf_probs)), p=self.leaf_probs)

class Node:
    def __init__(self, var_idx, comparator, value, lows, highs, left=None, right=None):
        self.left = left
        self.right = right
        self.var_idx = var_idx
        self.comparator = comparator

        # scale from [-1, 1] to [lows, highs]
        # low, high = self.action_space.low, self.action_space.high
        # return low + (0.5 * (scaled_action + 1.0) * (high - low))

        if lows[self.var_idx] < -1e10:
            low = -1e10
        else:
            low = lows[self.var_idx]
        if highs[self.var_idx] > 1e10:
            high = 1e10
        else:
            high = highs[self.var_idx]
        self.value = (value - low) / (high - low)

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
    def __init__(self, node_values, depth, n_decision_nodes, lows, highs, var_names, action_names, n_leafs):
        i = 0
        self.depth = depth
        self.root = Node(node_values[i], node_values[i+1], node_values[i+2], lows, highs)
        self.var_names = var_names
        self.action_names = action_names
        q = [self.root]
        i += 3
        while i < len(node_values):
            current_node = q.pop(0)
            if i < n_decision_nodes * 3:
                left = Node(node_values[i], node_values[i+1], node_values[i+2], lows, highs)
                i += 3
                right = Node(node_values[i], node_values[i+1], node_values[i+2], lows, high)
                i += 3
                q.append(left)
                q.append(right)
            else:
                left = Leaf(node_values[i])
                right = Leaf(node_values[i + 1])
                i += 2
            current_node.left = left
            current_node.right = right

    def predict(self, values):
        return self.root.predict(values)

    def visualize(self):
        q = [(self.root, 0)]
        while q:
            current_node, current_depth = q.pop(-1)
            output = ''
            for i in range(current_depth):
                output += '\t'
            if current_node == None:
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