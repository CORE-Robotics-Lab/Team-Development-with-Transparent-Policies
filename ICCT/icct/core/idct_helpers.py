import numpy  as np
import torch
import sys
sys.path.insert(0, '../')
from ICCT.icct.core.idct import IDCT


class Node:
    def __init__(self, idx: int, node_depth: int, is_leaf: bool=False, left_child=None, right_child=None):
        self.idx = idx
        self.node_depth = node_depth
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf


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


def convert_decision_to_leaf(network, decision_node_index, use_gpu=False):
    old_weights = network.layers  # Get the weights out
    old_comparators = network.comparators  # get the comparator values out

    leaf_info = network.leaf_init_information

    leaves_with_idx = [(leaf_idx, leaf_info[leaf_idx]) for leaf_idx in range(len(leaf_info))]
    root = Node(find_root(leaves_with_idx), 0)
    find_children(root, leaves_with_idx, current_depth=1)

    node = root
    q = []
    while node.idx != decision_node_index:
        node = q.pop(0)

    descendants = []
    q = [node]
    while len(q) > 0:
        node = q.pop(0)
        descendants.append(node)
        if node.left_child is not None and node.left_child.is_leaf is False:
            q.append(node.left_child)
        if node.right_child is not None and node.right_child.is_leaf is False:
            q.append(node.right_child)

    new_weights = [old_weights[i].detach().clone().data.cpu().numpy() \
                   for i in range(len(old_weights)) if i not in descendants]
    new_comparators = [old_comparators[i].detach().clone().data.cpu().numpy() \
                       for i in range(len(old_comparators)) if i not in descendants]

    for leaf_index in len(leaf_info):
        if decision_node_index in leaf_info[leaf_index][0] or \
                decision_node_index in leaf_info[leaf_index][1]:
            del leaf_info[leaf_index]

    new_network = IDCT(input_dim=network.input_dim, weights=new_weights, comparators=new_comparators,
                       leaves=leaf_info, alpha=network.alpha.item(), is_value=network.is_value,
                       device='cuda' if use_gpu else 'cpu', output_dim=network.output_dim)
    if use_gpu:
        new_network = new_network.cuda()
    return new_network

def convert_leaf_to_decision(network, leaf_index, use_gpu=False):
    """
    Duplicates the network and returns a new one, where the node at leaf_index as been turned into a splitting node
    with two leaves that are slightly noisy copies of the previous node
    :param network: prolonet in
    :param deeper_network: deeper_network to take the new node / leaves from
    :param leaf_index: index of leaf to turn into a split
    :return: new prolonet (value or normal)
    """
    old_weights = network.layers  # Get the weights out
    old_comparators = network.comparators  # get the comparator values out
    leaf_information = network.leaf_init_information[leaf_index]  # get the old leaf init info out
    left_path = leaf_information[0]
    right_path = leaf_information[1]

    new_weight = np.random.normal(scale=0.2,
                                  size=old_weights[0].size()[0])
    new_comparator = np.random.normal(scale=0.2,
                                      size=old_comparators[0].size()[0])
    new_leaf1 = np.random.normal(scale=0.2,
                                 size=network.action_mus[leaf_index].size()[0])
    new_leaf2 = np.random.normal(scale=0.2,
                                 size=network.action_mus[leaf_index].size()[0])

    new_weights = [weight.detach().clone().data.cpu().numpy() for weight in old_weights]
    new_weights.append(new_weight)  # Add it to the list of nodes
    new_comparators = [comp.detach().clone().data.cpu().numpy() for comp in old_comparators]
    new_comparators.append(new_comparator)  # Add it to the list of nodes

    new_node_ind = len(new_weights) - 1  # Remember where we put it

    # Create the paths, which are copies of the old path but now with a left / right at the new node
    new_leaf1_left = left_path.copy()
    new_leaf1_right = right_path.copy()
    new_leaf2_left = left_path.copy()
    new_leaf2_right = right_path.copy()
    # Leaf 1 goes left at the new node, leaf 2 goes right
    new_leaf1_left.append(new_node_ind)
    new_leaf2_right.append(new_node_ind)

    new_leaf_information = network.leaf_init_information
    for index, leaf_prob_vec in enumerate(network.action_mus):  # Copy over the learned leaf weight
        new_leaf_information[index][-1] = leaf_prob_vec.detach().clone().data.cpu().numpy()
    new_leaf_information.append([new_leaf1_left, new_leaf1_right, new_leaf1])
    new_leaf_information.append([new_leaf2_left, new_leaf2_right, new_leaf2])
    # Remove the old leaf
    del new_leaf_information[leaf_index]
    new_network = IDCT(input_dim=network.input_dim, weights=new_weights, comparators=new_comparators,
                       leaves=new_leaf_information, alpha=network.alpha.item(), is_value=network.is_value,
                       device='cuda' if use_gpu else 'cpu', output_dim=network.output_dim)
    if use_gpu:
        new_network = new_network.cuda()
    return new_network