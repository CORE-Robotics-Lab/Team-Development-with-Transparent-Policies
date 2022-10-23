import numpy  as np
import torch
import sys
sys.path.insert(0, '../')
from ICCT.icct.core.idct import IDCT


def swap_in_node(network, leaf_index, use_gpu=False):
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