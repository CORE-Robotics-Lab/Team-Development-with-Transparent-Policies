from ipm.models.decision_tree import DecisionTree
import pickle
import os
import numpy as np
from collections import Counter

import math
def calculate_entropy(probabilities):
    """
    Args:
        values: list of values
    Returns:
        entropy: entropy of the list of values
    """


    entropy = 0.0

    for probability in probabilities:
        if probability > 0:
            entropy -= probability * math.log2(probability)

    return entropy

def find_indices_of_repeated_value(lst):
    repeated_value = None
    repeated_indices = []

    for index, value in enumerate(lst):
        if lst.count(value) > 1:
            repeated_value = value
            repeated_indices.append(index)

    return repeated_value, repeated_indices

def compute_entropy_with_double_indices(indices, values):
    """ takes in multiple nodes"""
    avg_entropy = None
    from collections import Counter
    entropies = []
    for e, i in enumerate(indices):
        if len(set(i)) == len(i):
            repeated_values = values[e]
        else:
            # find indices of repeated elements
            repeated_value, repeated_indices = find_indices_of_repeated_value(i)
            repeated_values = [0]
            for k in repeated_indices:
                repeated_values[0] += values[e][k]

            for m in range(3):
                if m not in repeated_indices:
                    repeated_values.append(values[e][m])

        # calculate entropy
        entropies.append(calculate_entropy(repeated_values))

    print('mean entropy: ', np.mean(entropies))

    return np.mean(entropies)



nodes = {}
def get_leaf_nodes(tree, path=[], path2=[]):
    # Base case: If the node is a leaf node, add the path to the leaf node
    try:
        if tree.left is None and tree.right is None:
            path.append(tree)
            return path
    except:
        # print(tree.action.values)
        path.append(tree.action.values)
        path2.append(tree.action.indices)
        return path

    # Recursive case: Continue exploring the left and right subtrees
    if tree.left:
        get_leaf_nodes(tree.left, path)
    if tree.right:
        get_leaf_nodes(tree.right, path)

    return path, path2

class Data:
    def __init__(self, domain_name, user_id, entropy_diff):
        self.domain_name = domain_name
        self.user_id = user_id
        self.entropy_diff = entropy_diff

if __name__ == '__main__':
    data = []
    # let's load the decision tree then compute the entropy of the leaves
    experiment_folder = 'data/experiments/human_modifies_tree'
    # get subfolders
    entropies_domain_1 = []
    entropies_domain_2 = []
    user_folders = [os.path.join(experiment_folder, f) for f in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, f))]
    for user_f in user_folders:
        if user_f[-2:] == "41":
            continue
        if user_f[-2:] == "43":
            continue
        domain_names = [f for f in os.listdir(user_f) if os.path.isdir(os.path.join(user_f, f))]
        for domain_name in domain_names:
            if domain_name == 'tutorial':
                continue
            elif domain_name == 'two_rooms_narrow':
                og_entropy = (calculate_entropy([0.79, .13, 0.08]) + calculate_entropy([0.68, .31, 0.01]))/2
            elif domain_name == 'forced_coordination':
                og_entropy = (calculate_entropy([0.53, .46, 0.01]) + calculate_entropy([0.54, .45, 0.01])) / 2
            domain_folder = os.path.join(user_f, domain_name)
            # check for files that end with '.pkl
            files = os.listdir(domain_folder)
            pkl_files = [os.path.join(user_f, f) for f in files if f.endswith('.pkl')]
            if len(pkl_files) == 0:
                continue
            else:
                # let's get first and last decision tree and compare (first one is decision_tree_0.pkl, last one is decision_tree_len(pkl_files).pkl)
                first_pickle_file = 'decision_tree_0.pkl'
                second_pickle_file = 'decision_tree_1.pkl'
                last_pickle_file = 'decision_tree_{}.pkl'.format(len(pkl_files) - 1)
                first_pickle_path = os.path.join(domain_folder, first_pickle_file)
                second_pickle_path = os.path.join(domain_folder, second_pickle_file)
                last_pickle_path = os.path.join(domain_folder, last_pickle_file)

                first_tree = pickle.load(open(first_pickle_path, 'rb'))
                second_tree = pickle.load(open(second_pickle_path, 'rb'))
                last_tree = pickle.load(open(last_pickle_path, 'rb'))


                # let's get the minimum number of leaves across both trees
                min_num_leaves = min(first_tree.n_leaves, last_tree.n_leaves)
                initial_leaves = [node for node in first_tree.node_values if 'Leaf' in type(node).__name__]
                final_leaves = [node for node in last_tree.node_values if 'Leaf' in type(node).__name__]
                # let's get the entropy of the leaves
                old_nodes = get_leaf_nodes(first_tree.root)
                second_nodes = get_leaf_nodes(second_tree.root)
                all_nodes = get_leaf_nodes(last_tree.root)
                # print('first tree')
                # old_entropy = compute_entropy_with_double_indices(old_nodes[1], old_nodes[0])
                # print('second tree')
                # second_entropy = compute_entropy_with_double_indices(second_nodes[1], second_nodes[0])
                print('last tree')
                last_entropy = compute_entropy_with_double_indices(all_nodes[1], all_nodes[0])

                # final_entropies = [calculate_entropy(leaf.values) for leaf in final_leaves]
                # let's get the average entropy of the leaves
                # initial_avg_entropy = np.mean(old_entropy)
                final_avg_entropy = last_entropy # np.mean([old_entropy, second_entropy, last_entropy])
                # calculate the difference in entropy
                entropy_diff = final_avg_entropy - og_entropy
                if domain_name == 'two_rooms_narrow':
                    entropies_domain_2.append(entropy_diff)
                elif domain_name == 'forced_coordination':
                    entropies_domain_1.append(entropy_diff)
                user = user_f.split('/')[-1]
                data.append(Data(domain_name, user, entropy_diff))

    print('mean entropy diff domain 1: ', np.mean(entropies_domain_1))
    print('mean entropy diff domain 2: ', np.mean(entropies_domain_2))
    with open('data/experiments/human_modifies_tree/entropy_diffs.csv', 'w') as f:
        f.write('domain_name,user_id,entropy_diff\n')
        for d in data:
            f.write('{},{},{}\n'.format(d.domain_name, d.user_id, d.entropy_diff))