import copy
import torch
from ipm.models.idct import IDCT


class Node:
    def __init__(self, idx: int, node_depth: int, is_leaf: bool = False, left_child=None, right_child=None, parent=None):
        self.idx = idx
        self.node_depth = node_depth
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.parent = parent


class TreeInfo:
    def __init__(self, tree, is_continuous_actions=False):
        self.tree = tree
        self.node_dict = {}
        self.is_continuous_actions = is_continuous_actions
        self.extract_decision_nodes_info()
        self.extract_action_leaves_info()
        self.extract_path_info()
        self.tree = self.prune_and_make_new_tree()

    def prune_and_make_new_tree(self):
        self.node_dict[0] = self.root
        pruning = True
        while pruning:
            ignore_list = []
            in_list = []
            # step through each decision node
            for i in range(len(self.tree.layers)):
                if i in ignore_list:
                    continue
                if self.prunable[i] == 'cut left':
                    if i == 0:

                        # add ignoring of all on left
                        leaves_with_idx = copy.deepcopy(
                            [(leaf_idx, self.leaves[leaf_idx]) for leaf_idx in range(len(self.leaves))])
                        left_subtree = [leaf for leaf in leaves_with_idx if self.root.idx in leaf[1][0]]
                        right_subtree = [leaf for leaf in leaves_with_idx if self.root.idx in leaf[1][1]]
                        for j in left_subtree:
                            k = j[1][0] + j[1][1]
                            for l in k:
                                if l not in ignore_list:
                                    ignore_list.append(l)

                        for j in right_subtree:
                            k = j[1][0] + j[1][1]
                            for l in k:
                                if l not in in_list:
                                    if l in ignore_list:
                                        continue
                                    in_list.append(l)
                        # new root
                        self.root = self.root.right_child
                        self.root.parent = None
                        pruning = False
                        n_leaves = len(right_subtree)
                        new_layers = torch.zeros((n_leaves-1, 16))
                        new_comparators = torch.zeros(n_leaves-1,1)
                        for e,j in enumerate(sorted(in_list)):
                            new_layers[e] = self.tree.layers[j]
                            new_comparators[e] = self.tree.comparators[j]
                        break

                    else:
                        self.node_dict[i].parent.right_child = self.node_dict[i].left_child
                        pruning = False
                        break
                elif self.prunable[i] == 'cut right':
                    self.root = self.root.left_child
                    pruning = False
                    break
                else:
                    pass


        # start from root and make new tree
        # first let's count leaves


        return IDCT(input_dim=16, output_dim=11, leaves=n_leaves, hard_node=True, device='cuda', argmax_tau=1.0,
                        alpha=self.tree.alpha, comparators=new_comparators, weights=new_layers)


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
        for i in range(self.tree.num_leaves - 1):
            b = onehot_weights[i].argmax().item()
            # print(b, possible_states[onehot_weights[i].argmax()], 'true_weight', self.tree.layers[i][b], 'true_comparator',self.tree.comparators[i])
            # print(b, possible_states[onehot_weights[i].argmax()], 'modified_weight', self.new_weights[i][b], 'modified_comparator', self.comparators[i],
            #       self.compare_sign[i][b])
            print('----------------------------')
            # note all this assumes new_weights is +1 and not -1
            if self.comparators[i] > 1:
                self.prunable[i] = 'cut left'
            elif self.comparators[i] < 0:
                self. prunable[i] = 'cut right'
            else:
                self.prunable[i] = 'not prunable'

        # restart from top


        # pruned_tree = IDCT(input_dim=16, output_dim=11, leaves=n_leaves, hard_node=False, device='cuda', argmax_tau=1.0,
        #                 alpha=alpha, comparators=comparators, weights=layers)

        print('hello')

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
        self.root = Node(self.find_root(leaves_with_idx), 0)


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

        left_child = Node(left_child, current_depth, left_child_is_leaf, parent=node.idx)
        self.node_dict[left_child.idx] = left_child
        right_child = Node(right_child, current_depth, right_child_is_leaf, parent=node.idx)
        self.node_dict[right_child.idx] = right_child
        node.left_child = left_child
        node.right_child = right_child

        # recursively find children
        if not left_child_is_leaf:
            self.find_children(left_child, left_subtree, current_depth + 1)
        if not right_child_is_leaf:
            self.find_children(right_child, right_subtree, current_depth + 1)
