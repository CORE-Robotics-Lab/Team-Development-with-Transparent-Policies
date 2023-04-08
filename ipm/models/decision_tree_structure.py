import random
from abc import ABC, abstractmethod

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ipm.gui.tree_gui_utils import TreeInfo
from ipm.models.idct import IDCT


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


class BranchingNode(Node):
    def __init__(self, var_idx: int = None, comp_val: float = 0.5,
                 left: Node = None, right: Node = None, idx: int = None,
                 is_root: bool = False, depth: int = 0,
                 normal_ordering: int = 0):
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


class DecisionTreeStructure:
    def __init__(self, num_vars: int, node_values: list = None, depth: int = 3):
        """
            Class for a decision tree
        :param num_vars: Number of variables in the observation
        :param num_actions: Number of actions
        :param node_values: Values of the nodes in the tree. If None, the tree is randomly generated
        :param depth: Depth of the tree
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

        # we assume a full tree here
        self.n_decision_nodes = 2 ** (depth + 1) - 1

        if not self.random_tree:
            assert len(self.node_values) == self.n_decision_nodes

        self.root = None
        self.construct_empty_full_tree()
        self.populate_values()

        assert len(self.node_values) == self.n_decision_nodes
        assert self.root is not None

    def construct_empty_full_tree(self) -> None:
        """
            Constructs an empty full tree
        """

        assert self.depth > 0

        self.root = BranchingNode(is_root=True, depth=0)
        q = [(1, self.root)]

        # keep populating until we reach depth
        while q:
            current_depth, node = q.pop(0)
            if current_depth < self.depth + 1:
                assert type(node) == BranchingNode
                node.left = BranchingNode(depth=current_depth)
                node.right = BranchingNode(depth=current_depth)
                q.append((current_depth + 1, node.left))
                q.append((current_depth + 1, node.right))

    def populate_dfs_inorder(self, node: Node):
        """
            Performs a dfs inorder traversal on the tree, also populates values and gene space
        :param node: Current node in the traversal
        """
        is_branch_node = type(node) == BranchingNode

        # recurse on left child
        if is_branch_node and node.left is not None:
            self.populate_dfs_inorder(node.left)

        if is_branch_node:
            if self.random_tree:
                node.var_idx = random.randint(0, self.num_vars - 1)
                self.node_values.append(node.var_idx)
            else:
                node.var_idx = self.node_values[self.current_node_idx]
            self.gene_space.append(list(range(self.num_vars)))
        else:
            raise Exception("Leaf nodes should not exist")

        node.idx = self.current_node_idx
        self.current_node_idx += 1

        # recurse on right child
        if is_branch_node and node.right is not None:
            self.populate_dfs_inorder(node.right)

    def populate_values(self):
        """
            Populates the values of the tree
        """
        self.current_node_idx = 0
        self.populate_dfs_inorder(self.root)

    def predict(self, values: np.array, debug: bool = False) -> int:
        """
            Predicts the action for a given set of values
        :param values: Observation
        :param debug: Whether to print debug information
        :return: The action to take
        """
        raise Exception("Should not predict with just a tree structure.")
