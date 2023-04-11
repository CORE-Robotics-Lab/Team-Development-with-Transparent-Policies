import random
from abc import ABC, abstractmethod

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ipm.gui.tree_gui_utils import TreeInfo
from ipm.models.idct import IDCT
import torch
import torch.nn.functional as F
from torch import nn


class Node(ABC):
    @abstractmethod
    def predict(self, values: np.array, debug: bool = False) -> int:
        """
            Predicts the action to take at this node
        :param values: Values of the observation
        :param debug: Whether to print debug statements
        :return: The action taken at this node
        """
        pass


class LeafNode(Node):
    def __init__(self, action=None, idx: int = None, depth: int = 0, bfs_idx=-1):
        """
            Class for
        :param action: Action to take at this leaf
        :param idx: Leaf index, used for internal bookkeeping
        :param depth: Depth of this leaf in the tree
        """
        self.action = action
        self.idx = idx
        self.depth = depth
        self.bfs_idx = bfs_idx

    def predict(self, values: np.array, debug: bool = False) -> int:
        """
            Predicts the action to take at this leaf
        :param values: Values of the observation. These are not needed but are included
            for consistency with the BranchingNode
        :param debug: Whether to print debug statements
        :return: The action taken at this leaf
        """
        if debug:
            print(f"Leaf {self.idx} with action {self.action}")
        return self.action


class BranchingNode(Node):
    def __init__(self, var_idx: int = None, comp_val: float = 0.5,
                 left: Node = None, right: Node = None, idx: int = None,
                 is_root: bool = False, depth: int = 0,
                 normal_ordering: int = 0, bfs_idx=-1):
        """
            Class for a branching node in a decision tree
        :param var_idx: Index of the variable to compare
        :param comp_val: Value to compare the variable to
        :param left: Left child node
        :param right: Right child node
        :param idx: Node index, used for internal bookkeeping
        :param is_root: Whether this node is the root of the tree
        :param depth: Depth of this node in the tree
        :param normal_ordering: Whether to use normal ordering (0) or reverse ordering (1)
        """
        self.left = left
        self.right = right

        self.var_idx = var_idx
        self.comp_val = comp_val
        self.normal_ordering = normal_ordering

        self.idx = idx
        self.is_root = is_root
        self.depth = depth
        self.bfs_idx = bfs_idx

    def predict(self, values: np.array, debug: bool = False) -> int:
        """
            Predicts the action to take at this node
        :param values: Values of the observation
        :param debug: Whether to print debug statements
        :return: Action to take when traversing through the tree from this node
        """
        if values[self.var_idx] <= self.comp_val:
            if self.normal_ordering == 0:
                if debug:
                    print(f"Going left because val at idx {self.var_idx} is {values[self.var_idx]} <= {self.comp_val}")
                return self.left.predict(values, debug=debug)
            else:
                if debug:
                    print(f"Going right because val at idx {self.var_idx} is {values[self.var_idx]} <= {self.comp_val}")
                return self.right.predict(values, debug=debug)
        else:
            if self.normal_ordering == 0:
                if debug:
                    print(f"Going right because val at idx {self.var_idx} is {values[self.var_idx]} > {self.comp_val}")
                return self.right.predict(values, debug=debug)
            else:
                if debug:
                    print(f"Going left because val at idx {self.var_idx} is {values[self.var_idx]} > {self.comp_val}")
                return self.left.predict(values, debug=debug)


class DecisionTree:
    def __init__(self, num_vars: int, num_actions: int, node_values: list = None,
                 depth: int = 3, idct_root = None):
        """
            Class for a decision tree
        :param num_vars: Number of variables in the observation
        :param num_actions: Number of actions
        :param node_values: Values of the nodes in the tree. If None, the tree is randomly generated.
        :param depth: Depth of the tree
        :param root: Optional, root of the tree which we can infer the structure from
        """

        # check if we need to randomly generate the tree
        self.random_tree = node_values is None
        if self.random_tree:
            self.node_values = []
        else:
            self.node_values = node_values

        self.gene_space = []
        self.depth = depth

        self.num_vars = num_vars
        self.num_actions = num_actions

        if idct_root is None:
            # we assume a full tree here
            self.n_decision_nodes = 2 ** (depth + 1) - 1
            self.n_leaves = 2 ** (depth + 1)
            self.root = None
            self.construct_empty_full_tree()
            self.populate_values()
        else:
            self.root = BranchingNode()
            self.n_decision_nodes = 0
            self.n_leaves = 0
            self.copy_over_nodes(self.root, other_node=idct_root, depth=0)
            self.populate_values(use_prior_values=True)

        assert len(self.node_values) == self.n_decision_nodes + self.n_leaves
        assert self.root is not None

    def copy_over_nodes(self, node, other_node, depth):
        if other_node is None:
            return

        # recurisvely copy over nodes
        if type(node) == LeafNode:
            node.action = other_node.value
            node.depth = other_node.node_depth
            self.n_leaves += 1
        else:
            node.var_idx = other_node.var_idx
            node.depth = other_node.node_depth
            node.comp_val = other_node.value
            self.n_decision_nodes += 1
            # check if the node has children that exist
            # if so, create new nodes
            if other_node.left_child is not None:
                if other_node.left_child.is_leaf:
                    node.left = LeafNode(depth=depth)
                else:
                    node.left = BranchingNode(depth=depth)
                self.copy_over_nodes(node.left, other_node.left_child, depth+1)
            if other_node.right_child is not None:
                if other_node.right_child.is_leaf:
                    node.right = LeafNode(depth=depth)
                else:
                    node.right = BranchingNode(depth=depth)
                self.copy_over_nodes(node.right, other_node.right_child, depth+1)

    def construct_empty_full_tree(self) -> None:
        """
            Constructs an empty full tree
        """

        assert self.depth >= 0

        self.root = BranchingNode(is_root=True, depth=0)
        q = [(1, self.root)]
        idx = 0

        # keep populating until we reach depth
        while q:
            current_depth, node = q.pop(0)
            node.bfs_idx = idx
            idx += 1
            if current_depth == self.depth + 1:
                node.left = LeafNode(depth=current_depth)
                node.right = LeafNode(depth=current_depth)
                q.append((current_depth + 1, node.left))
                q.append((current_depth + 1, node.right))
            elif current_depth < self.depth + 1:
                assert type(node) == BranchingNode
                node.left = BranchingNode(depth=current_depth)
                node.right = BranchingNode(depth=current_depth)
                q.append((current_depth + 1, node.left))
                q.append((current_depth + 1, node.right))

    def convert_dt_decision_to_leaf(self, node: BranchingNode) -> None:
        """
            In the tree, converts a decision node to a leaf node
        :param node: The decision node to convert to a leaf node
        """

        parent = self.root
        q = [parent]

        # Keep searching with bfs until we find the decision node
        while q:

            parent = q.pop(0)
            left_child = parent.left
            right_child = parent.right

            if left_child is node:
                action = LeafInfo(action_idx=[random.sample(range(self.num_actions), 3), [1 / 2, 1 / 4, 1 / 4]],
                                  torch_tensor=False)
                parent.left = LeafNode(action=action, idx=None,
                                       depth=parent.depth + 1)
                break
            elif left_child is not None and type(left_child) == BranchingNode:
                q.append(left_child)
            if right_child is node:
                action = LeafInfo(action_idx=[random.sample(range(self.num_actions), 3), [1 / 2, 1 / 4, 1 / 4]],
                                  torch_tensor=False)
                parent.right = LeafNode(action=action, idx=None,
                                        depth=parent.depth + 1)
                break
            elif right_child is not None and type(right_child) == BranchingNode:
                q.append(right_child)

    def convert_dt_leaf_to_decision(self, node: LeafNode):
        """
            In the tree, converts a leaf node to a decision node
        :param node: The leaf node to convert to a decision node
        """

        # Pick a random variable to split on
        var_idx = random.randint(0, self.num_vars - 1)

        parent = self.root
        q = [parent]

        while q:

            parent = q.pop(0)
            assert type(parent) == BranchingNode

            left_child = parent.left
            right_child = parent.right

            if left_child is node:
                random_leaf1 = LeafNode(action=random.randint(0, self.num_actions - 1), idx=None,
                                        depth=parent.depth + 2)
                random_leaf2 = LeafNode(action=random.randint(0, self.num_actions - 1), idx=None,
                                        depth=parent.depth + 2)
                parent.left = BranchingNode(var_idx=var_idx, comp_val=0.5, left=random_leaf1, right=random_leaf2,
                                            idx=None,
                                            is_root=False, depth=parent.depth + 1)
                break
            elif left_child is not None and type(left_child) == BranchingNode:
                q.append(left_child)
            if right_child is node:
                random_leaf1 = LeafNode(action=random.randint(0, self.num_actions - 1), idx=None,
                                        depth=parent.depth + 2)
                random_leaf2 = LeafNode(action=random.randint(0, self.num_actions - 1), idx=None,
                                        depth=parent.depth + 2)
                parent.right = BranchingNode(var_idx=var_idx, comp_val=0.5, left=random_leaf1, right=random_leaf2,
                                             idx=None,
                                             is_root=False, depth=parent.depth + 1)
                break
            elif right_child is not None and type(right_child) == BranchingNode:
                q.append(right_child)

    def populate_dfs_inorder(self, node: Node, use_prior_values=False):
        """
            Performs a dfs inorder traversal on the tree, also populates values and gene space
        :param node: Current node in the traversal
        """
        is_branch_node = type(node) == BranchingNode

        # recurse on left child
        if is_branch_node and node.left is not None:
            self.populate_dfs_inorder(node.left, use_prior_values)

        if is_branch_node:
            if self.random_tree:
                if not use_prior_values:
                    node.var_idx = random.randint(0, self.num_vars - 1)
                self.node_values.append(node.var_idx)
            else:
                node.var_idx = self.node_values[self.current_node_idx]
            self.gene_space.append(list(range(self.num_vars)))
        else:
            if self.random_tree:
                if not use_prior_values:
                    node.action = random.randint(0, self.num_actions - 1)
                self.node_values.append(node.action)
            else:
                node.action = self.node_values[self.current_node_idx]
            self.gene_space.append(list(range(self.num_actions)))

        node.idx = self.current_node_idx
        self.current_node_idx += 1

        # recurse on right child
        if is_branch_node and node.right is not None:
            self.populate_dfs_inorder(node.right, use_prior_values)

    def populate_values(self, use_prior_values=False):
        """
            Populates the values of the tree
        """
        self.current_node_idx = 0
        self.populate_dfs_inorder(node=self.root, use_prior_values=use_prior_values)

    def predict(self, values: np.array, debug: bool = False) -> int:
        """
            Predicts the action for a given set of values
        :param values: Observation
        :param debug: Whether to print debug information
        :return: The action to take
        """
        if values.shape[0] == 1:
            values = values[0]
        return self.root.predict(values, debug=debug)

    @staticmethod
    def from_sklearn(sklearn_model: DecisionTreeClassifier, num_vars: int, num_actions: int):
        """
            Creates a decision tree from a sklearn decision tree
        :param sklearn_model: Decision tree model from sklearn
        :param num_vars: Number of variables in the tree
        :param num_actions: Number of actions in the tree
        :return: DecisionTree object
        """
        # todo: add support for asymmetric trees (symmetric is only needed for genetic algorithm)
        depth = sklearn_model.tree_.max_depth
        dt = DecisionTree(num_vars, num_actions, node_values=None, depth=depth)

        # extract some data from the sklearn model that we will need to lookup values
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
                if type(node) == LeafNode:
                    node.action = action
                    dt.node_values[node.idx] = node.action
                else:
                    # we set all descendants of this node to the same action
                    q = [node]
                    # this is where we can add support for asymmetric trees
                    while q:
                        n = q.pop(0)
                        if type(n) == LeafNode:
                            n.action = action
                            dt.node_values[n.idx] = n.action
                        else:
                            n.var_idx = random.randint(0, dt.num_vars - 1)
                            n.comp_val = 0.5
                            q.append(n.left)
                            q.append(n.right)
        return dt


def decision_tree_to_ddt(tree, input_dim, output_dim, device):
    node_indices = {}
    # first pass, we assign each node to an index
    q = [tree.root]
    while q:
        node = q.pop(0)
        if type(node) == LeafNode:
            continue
        node_indices[node] = len(node_indices)
        if type(node) == BranchingNode:
            q.append(node.left)
            q.append(node.right)

    n_nodes = len(node_indices)
    comparators = torch.Tensor([[0.5] for _ in range(n_nodes)])
    alphas = torch.Tensor([[-1] for _ in range(n_nodes)])
    leaves = [] # of format [left_parents, right_parents, action_probabilities]
    weights = torch.zeros(n_nodes, tree.num_vars)

    # (node, left_parents, right_parents)
    q = [(tree.root, [], [])]
    while q:
        node, left_parents, right_parents = q.pop(0)
        left_parents = left_parents.copy()
        right_parents = right_parents.copy()
        if type(node) == LeafNode:
            # get logits with numerical stability
            action_probabilities = node.action
            action_probabilities = np.clip(action_probabilities, 1e-8, 1 - 1e-8)
            logits = np.log(action_probabilities)
            leaves.append([left_parents, right_parents, logits])
            continue
        if type(node) == BranchingNode:

            assert node.var_idx is not None
            comparators[node_indices[node]] = node.comp_val
            weights[node_indices[node], node.var_idx] = 1
            if not node.normal_ordering:
                alphas[node_indices[node]] *= -1

            q.append((node.left, left_parents + [node_indices[node]], right_parents))
            q.append((node.right, left_parents, right_parents + [node_indices[node]]))

    action_mus = torch.Tensor([x[-1] for x in leaves])

    idct = IDCT(input_dim=input_dim,
                weights=weights,
                comparators=comparators,
                alpha=alphas,
                leaves=leaves,
                output_dim=output_dim,
                use_individual_alpha=False,
                device=device)

    idct.action_mus = nn.Parameter(action_mus, requires_grad=True)
    idct.update_leaf_init_information()
    return idct

class LeafInfo:
    def __init__(self, action_idx, torch_tensor=True):
        if torch_tensor:
            self.values = action_idx.values.tolist()
            self.indices = action_idx.indices.tolist()
        else:
            self.values = action_idx[1]
            self.indices = action_idx[0]


def sparse_ddt_to_decision_tree(tree: IDCT, env):
    tree_info = TreeInfo(tree)

    n_decision_nodes = len(tree_info.impactful_vars_for_nodes)
    n_leaves = len(tree_info.leaves)
    # note this depth below doesn't match the depth of the idct
    depth = np.log2(n_decision_nodes + n_leaves).astype(int) - 1

    dt = DecisionTree(num_vars=env.observation_space.shape[0], num_actions=env.n_actions_ego,
                      depth=depth, idct_root=tree_info.root)

    return dt, tree_info
