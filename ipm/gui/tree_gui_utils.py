import copy
import torch

class Node:
    def __init__(self, idx: int, node_depth: int, is_leaf: bool=False, left_child=None, right_child=None):
        self.idx = idx
        self.node_depth = node_depth
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf

class TreeInfo:
    def __init__(self, tree, is_continuous_actions=False):
        self.tree = tree
        self.is_continuous_actions = is_continuous_actions
        self.extract_decision_nodes_info()
        self.extract_action_leaves_info()
        self.extract_path_info()

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
        self.action_mus = self.tree.action_mus

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
