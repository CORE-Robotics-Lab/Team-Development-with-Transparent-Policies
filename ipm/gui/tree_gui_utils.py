import copy
import torch
from ipm.models.idct import IDCT
from ipm.models.idct_helpers import find_root, find_children, find_ancestors
import numpy as np
import torch
import torch.nn.functional as F

class LeafInfo:
    def __init__(self, action_idx, torch_tensor=True):
        if torch_tensor:
            self.values = action_idx.values.tolist()
            self.indices = action_idx.indices.tolist()
        else:
            self.values = action_idx[1]
            self.indices = action_idx[0]


class Node:
    def __init__(self, idx: int, node_depth: int, is_leaf: bool = False, left_child=None, right_child=None,
                 parent=None, value=None, var_idx=None):
        self.idx = idx
        self.node_depth = node_depth
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.parent = parent
        self.value = value
        self.var_idx = var_idx


class TreeInfo:
    def __init__(self, tree, is_continuous_actions=False):
        self.tree = tree
        self.node_dict = {}
        self.is_continuous_actions = is_continuous_actions
        need_to_prune = self.extract_decision_nodes_info()
        self.extract_action_leaves_info()
        self.extract_path_info()
        self.prune_all()
        self.prune_all_redundant()

    @staticmethod
    def get_tree_with_pruned_node(tree: IDCT,
                                  decision_node_index: int,
                                  prune_left: bool,
                                  use_gpu: bool = False) -> IDCT:
        """

        :param tree: Original idct
        :param decision_node_index: Index of the decision node to prune (which is equivalent to index of the comparator, etc.)
        :param prune_left: Whether to prune left (true) or right (false). If the former, we don't care about
            # the left child and its children and vice-versa
        :param use_gpu: Whether to use GPU acceleration
        :return: The IDCT where the decision node has been pruned and replaced with its left or right child
            and their children (dependent on prune_left)
        """
        leaf_info = tree.leaf_init_information

        leaves_with_idx = copy.deepcopy([(leaf_idx, leaf_info[leaf_idx]) for leaf_idx in range(len(leaf_info))])
        # n_actions = len(leaves_with_idx[0][1][2])

        # Get the root node then recursively find all children
        root = Node(idx=find_root(leaves_with_idx), node_depth=0, value=tree.comparators[0])
        find_children(root, leaves_with_idx, current_depth=1)

        # run BFS to find the node that we would like to prune
        # which also contains pointers to all of its children
        node_to_prune = root
        q = [root]
        while len(q) > 0:
            node_to_prune = q.pop(0)
            # keep traversing until we find the node we want to prune
            if node_to_prune.idx == decision_node_index:
                break
            if node_to_prune.left_child is not None and node_to_prune.left_child.is_leaf is False:
                q.append(node_to_prune.left_child)
            if node_to_prune.right_child is not None and node_to_prune.right_child.is_leaf is False:
                q.append(node_to_prune.right_child)

        # populate the list of descendants of the node we want to prune
        nodes_to_prune_indices = [node_to_prune.idx]
        q = []
        if prune_left:
            if node_to_prune.left_child is not None and node_to_prune.left_child.is_leaf is False:
                q = [node_to_prune.left_child]
        else:
            if node_to_prune.right_child is not None and node_to_prune.right_child.is_leaf is False:
                q = [node_to_prune.right_child]

        while len(q) > 0:
            node = q.pop(0)
            nodes_to_prune_indices.append(node.idx)
            if node.left_child is not None and node.left_child.is_leaf is False:
                q.append(node.left_child)
            if node.right_child is not None and node.right_child.is_leaf is False:
                q.append(node.right_child)

        _, pruned_node_left_ancestors, pruned_node_right_ancestors = find_ancestors(root, decision_node_index)
        pruned_node_left_ancestors = [node.idx for node in pruned_node_left_ancestors]
        pruned_node_right_ancestors = [node.idx for node in pruned_node_right_ancestors]

        # prune the leaves that have the decision node in their ancestors
        new_leaf_info_pruned = []
        for leaf_idx, leaf in enumerate(leaf_info):
            left_ancestors = leaf_info[leaf_idx][0]
            right_ancestors = leaf_info[leaf_idx][1]
            if prune_left:
                if decision_node_index not in left_ancestors:
                    new_leaf_info_pruned.append(leaf)
            else:
                if decision_node_index not in right_ancestors:
                    new_leaf_info_pruned.append(leaf)

        # adjust the indices so that they are correct after pruning
        n_decision_nodes, _ = tree.comparators.shape
        old_idx_to_new_idx = {idx: idx for idx in range(n_decision_nodes)} # map old indices to new ones after pruning
        for idx in range(n_decision_nodes):
            for descendant in nodes_to_prune_indices:
                if idx > descendant:
                    old_idx_to_new_idx[idx] -= 1

        for descendant in nodes_to_prune_indices:
            del old_idx_to_new_idx[descendant]

        # new leaf info with adjusted ancestors
        new_leaf_info_adjusted_ancestors = []
        for leaf in new_leaf_info_pruned:

            # populate left ancestors
            # we want to remove the decision node from the ancestors
            # and also adjust the indices
            left_ancestors = []
            for i in range(len(leaf[0])):
                old_node_idx = leaf[0][i]
                if old_node_idx == decision_node_index:
                    continue # don't add the decision node to the ancestors
                adjusted_node_idx = old_idx_to_new_idx[old_node_idx]
                left_ancestors.append(adjusted_node_idx)

            # populate right ancestors
            # we want to remove the decision node from the ancestors
            # and also adjust the indices
            right_ancestors = []
            for j in range(len(leaf[1])):
                old_node_idx = leaf[1][j]
                if old_node_idx == decision_node_index:
                    continue # don't add the decision node to the ancestors
                adjusted_node_idx = old_idx_to_new_idx[old_node_idx]
                right_ancestors.append(adjusted_node_idx)

            new_leaf_info_adjusted_ancestors.append([left_ancestors, right_ancestors, leaf[2]])

        # we need to filter out the decision nodes that we are pruning
        # from the prior weights, comparators, and alphas
        old_weights = tree.layers
        old_comparators = tree.comparators
        old_alpha = tree.alpha

        new_weights = [old_weights[i].detach().clone().data.cpu().numpy() \
                       for i in range(len(old_weights)) if i not in nodes_to_prune_indices]
        new_comparators = [old_comparators[i].detach().clone().data.cpu().numpy() \
                           for i in range(len(old_comparators)) if i not in nodes_to_prune_indices]

        n_alphas = old_alpha.shape
        is_individual_alpha = not len(n_alphas) == 0 and n_alphas[0] > 1
        if is_individual_alpha:
            new_alpha = [old_alpha[i].detach().clone().data.cpu().numpy() \
                         for i in range(len(old_alpha)) if i not in nodes_to_prune_indices]
        else:
            new_alpha = [old_alpha.data.item()]

        new_weights = torch.Tensor(new_weights)
        new_comparators = torch.Tensor(new_comparators)
        new_alpha = torch.Tensor(new_alpha)

        new_network = IDCT(input_dim=tree.input_dim, weights=new_weights, comparators=new_comparators,
                           leaves=new_leaf_info_adjusted_ancestors, alpha=new_alpha, is_value=tree.is_value,
                           device='cuda' if use_gpu else 'cpu', output_dim=tree.output_dim)
        # TODO: Check whether leaf probabilities are reinitialized within IDCT constructor or used (we need to use these prior ones)

        if use_gpu:
            new_network = new_network.cuda()
        return new_network

    def prune_all(self):
        self.node_dict[0] = self.root

        # run bfs and keep track of which nodes to prune based upon comparator values
        comparator_max = 1
        comparator_min = 0
        prunable_idx = -1

        try_to_prune = True
        while try_to_prune:
            n_decision_nodes = self.tree.comparators.shape[0]
            n_leaves = len(self.tree.leaf_init_information)
            if n_decision_nodes == 1:
                break
            print('pruning tree with {} decision nodes and {} leaves'.format(n_decision_nodes, n_leaves))
            q = [self.root]
            found_prunable_node = False
            prune_left = False
            # perform bfs and try to find a node to prune
            while len(q) > 0:
                current_node = q.pop(0)
                if current_node.left_child is not None and current_node.left_child.is_leaf is False:
                    q.append(current_node.left_child)
                if current_node.right_child is not None and current_node.right_child.is_leaf is False:
                    q.append(current_node.right_child)

                if current_node.value < comparator_min or current_node.value > comparator_max:
                    if current_node.value < comparator_min:
                        print('Current node has value less than comparator min: {}'.format(current_node.value))
                        prune_left = False
                    elif current_node.value > comparator_max:
                        print('Current node has value greater than comparator max: {}'.format(current_node.value))
                        prune_left = True
                    # this node is prunable
                    found_prunable_node = True
                    prunable_idx = current_node.idx
                    break
            if not found_prunable_node:
                try_to_prune = False
            else:
                self.tree = self.get_tree_with_pruned_node(tree=self.tree,
                                                           decision_node_index=prunable_idx,
                                                           prune_left=prune_left,
                                                           use_gpu=False)
                self.node_dict = {}
                need_to_prune = self.extract_decision_nodes_info()
                self.extract_action_leaves_info()
                self.extract_path_info()

        return self.tree

    def prune_all_redundant(self):
        self.node_dict[0] = self.root

        # run bfs and keep track of which nodes to prune based upon comparator values
        comparator_max = 1
        comparator_min = 0
        domains_for_vars = [[0, 1] for _ in range(self.tree.input_dim)]
        prunable_idx = -1

        try_to_prune = True
        while try_to_prune:
            n_decision_nodes = self.tree.comparators.shape[0]
            n_leaves = len(self.tree.leaf_init_information)
            if n_decision_nodes == 1:
                break
            print('pruning tree with {} decision nodes and {} leaves'.format(n_decision_nodes, n_leaves))
            q = [(self.root, domains_for_vars)]  # Add the root node with initial bounds
            found_prunable_node = False
            prune_left = True

            while len(q) > 0:
                current_node, domains = q.pop(0)
                lower_bound, upper_bound = domains[current_node.var_idx]

                if current_node.left_child is not None and current_node.left_child.is_leaf is False:
                    new_upper_bound = min(upper_bound, current_node.value)
                    if lower_bound < new_upper_bound:
                        new_domains = copy.deepcopy(domains)
                        new_domains[current_node.var_idx] = [lower_bound, new_upper_bound]
                        q.append((current_node.left_child, new_domains))
                    else:
                        # this node is prunable
                        prune_left = True
                        found_prunable_node = True
                        prunable_idx = current_node.left_child.idx
                        break

                if current_node.right_child is not None and current_node.right_child.is_leaf is False:
                    new_lower_bound = max(lower_bound, current_node.value)
                    if new_lower_bound < upper_bound:
                        new_domains = copy.deepcopy(domains)
                        new_domains[current_node.var_idx] = [new_lower_bound, upper_bound]
                        q.append((current_node.right_child, new_domains))
                    else:
                        # this node is prunable
                        prune_left = False
                        found_prunable_node = True
                        prunable_idx = current_node.right_child.idx
                        break

            if not found_prunable_node:
                try_to_prune = False
            else:
                self.tree = self.get_tree_with_pruned_node(tree=self.tree,
                                                           decision_node_index=prunable_idx,
                                                           prune_left=prune_left,
                                                           use_gpu=False)
                self.node_dict = {}
                need_to_prune = self.extract_decision_nodes_info()
                self.extract_action_leaves_info()
                self.extract_path_info()

        return self.tree


    def extract_decision_nodes_info(self):
        weights = torch.abs(self.tree.layers.cpu())
        onehot_weights = self.tree.diff_argmax(weights)
        divisors = (weights * onehot_weights).sum(-1).unsqueeze(-1)
        divisors_filler = torch.zeros(divisors.size()).to(divisors.device)
        divisors_filler[divisors == 0] = 1
        divisors = divisors + divisors_filler

        self.impactful_vars_for_nodes = (onehot_weights.argmax(axis=1)).numpy()
        # TODO: this doesn't seem to support individual alpha?
        self.compare_sign = (self.tree.alpha.cpu() * self.tree.layers.cpu()) > 0
        self.new_weights = self.tree.layers.cpu() * onehot_weights / divisors
        self.comparators = self.tree.comparators.cpu() / divisors

        possible_states = ['Alt Holding onion', 'Alt Holding soup', 'Alt Holding dish', 'Ego Holding onion',
                           'Ego Holding soup', 'Ego Holding dish' 'Onion on Counter', 'Either pot needs ingredients',
                           'Pot Ready', 'Dish on Counter', 'Soup on Counter', 'Human Picking Up Onion',
                           'Human Picking up Dish', 'Human Picking up Soup', 'Human Serving', 'Human Putting Item Down']

        self.prunable = {}
        need_to_prune = False

        for i in range(len(self.comparators)):
            b = onehot_weights[i].argmax().item()
            # print(b, possible_states[onehot_weights[i].argmax()], 'true_weight', self.tree.layers[i][b], 'true_comparator',self.tree.comparators[i])
            # print(b, possible_states[onehot_weights[i].argmax()], 'modified_weight', self.new_weights[i][b], 'modified_comparator', self.comparators[i],
            #       self.compare_sign[i][b])
            print('----------------------------')
            # note all this assumes new_weights is +1 and not -1
            if self.comparators[i] > 1:
                self.prunable[i] = 'cut left'
                need_to_prune = True
            elif self.comparators[i] < 0:
                self.prunable[i] = 'cut right'
                need_to_prune = True
            else:
                self.prunable[i] = 'not prunable'

        # restart from top


        # pruned_tree = IDCT(input_dim=16, output_dim=11, leaves=n_leaves, hard_node=False, device='cuda', argmax_tau=1.0,
        #                 alpha=alpha, comparators=comparators, weights=layers)

        print('hello')
        return need_to_prune

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

        leaves_with_idx = copy.deepcopy([(leaf_idx, self.leaves[leaf_idx]) for leaf_idx in range(len(self.leaves))])
        self.root = Node(idx=self.find_root(leaves_with_idx), node_depth=0, value=self.comparators[0],
                         var_idx=self.impactful_vars_for_nodes[0], is_leaf=False)

        self.find_children(self.root, leaves_with_idx, current_depth=1)

    def find_root(self, leaves):
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

    def find_children(self, node, leaves, current_depth):
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
            left_child = self.find_root(left_subtree)
        else:
            left_child = left_subtree[0][0]
        if not right_child_is_leaf:
            right_child = self.find_root(right_subtree)
        else:
            right_child = right_subtree[0][0]


        if left_child_is_leaf:
            logits = self.action_mus[left_child]
            action_idx = torch.topk(F.softmax(logits), 3)  # torch.argmax(logits)
            new_action_idx = LeafInfo(action_idx)
            left_child_value = new_action_idx
        else:
            left_child_value = self.impactful_vars_for_nodes[left_child]

        left_child_var_idx = self.impactful_vars_for_nodes[left_child] if not left_child_is_leaf else None
        left_child = Node(idx=left_child, node_depth=current_depth, is_leaf=left_child_is_leaf, parent=node.idx,
                          value=left_child_value, var_idx=left_child_var_idx)
        self.node_dict[left_child.idx] = left_child

        if right_child_is_leaf:
            logits = self.action_mus[right_child]
            action_idx = torch.topk(F.softmax(logits), 3)
            new_action_idx = LeafInfo(action_idx)
            right_child_values = new_action_idx
        else:
            right_child_values = self.impactful_vars_for_nodes[right_child]

        right_child_var_idx = self.impactful_vars_for_nodes[right_child] if not right_child_is_leaf else None
        right_child = Node(idx=right_child, node_depth=current_depth, is_leaf=right_child_is_leaf, parent=node.idx,
                           value=right_child_values, var_idx=right_child_var_idx)
        self.node_dict[right_child.idx] = right_child
        node.left_child = left_child
        node.right_child = right_child

        # recursively find children
        if not left_child_is_leaf:
            self.find_children(left_child, left_subtree, current_depth + 1)
        if not right_child_is_leaf:
            self.find_children(right_child, right_subtree, current_depth + 1)
