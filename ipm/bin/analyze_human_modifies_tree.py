from ipm.models.decision_tree import DecisionTree
import pickle
import os
import numpy as np
from collections import Counter

def calculate_entropy(values: list):
    """
    Args:
        values: list of values
    Returns:
        entropy: entropy of the list of values
    """
    # get counts of each value
    value_counts = Counter(values)
    # get total number of values
    total_num_values = len(values)
    # calculate entropy
    entropy = 0.0
    for value in value_counts:
        prob = value_counts[value] / total_num_values
        entropy += prob * np.log2(prob)
    return -entropy


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
    user_folders = [os.path.join(experiment_folder, f) for f in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, f))]
    for user_f in user_folders:
        domain_names = [f for f in os.listdir(user_f) if os.path.isdir(os.path.join(user_f, f))]
        for domain_name in domain_names:
            if domain_name == 'tutorial':
                continue
            domain_folder = os.path.join(user_f, domain_name)
            # check for files that end with '.pkl
            files = os.listdir(domain_folder)
            pkl_files = [os.path.join(user_f, f) for f in files if f.endswith('.pkl')]
            if len(pkl_files) == 0:
                continue
            else:
                # let's get first and last decision tree and compare (first one is decision_tree_0.pkl, last one is decision_tree_len(pkl_files).pkl)
                first_pickle_file = 'decision_tree_0.pkl'
                last_pickle_file = 'decision_tree_{}.pkl'.format(len(pkl_files) - 1)
                first_pickle_path = os.path.join(domain_folder, first_pickle_file)
                last_pickle_path = os.path.join(domain_folder, last_pickle_file)
                first_tree = pickle.load(open(first_pickle_path, 'rb'))
                last_tree = pickle.load(open(last_pickle_path, 'rb'))
                # let's get the entropy of the leaves
                first_entropy = 0.0
                last_entropy = 0.0
                # let's get the minimum number of leaves across both trees
                min_num_leaves = min(first_tree.n_leaves, last_tree.n_leaves)
                initial_leaves = [node for node in first_tree.node_values if 'Leaf' in type(node).__name__]
                final_leaves = [node for node in last_tree.node_values if 'Leaf' in type(node).__name__]
                # let's get the entropy of the leaves
                initial_entropies = [calculate_entropy(leaf.values) for leaf in initial_leaves]
                final_entropies = [calculate_entropy(leaf.values) for leaf in final_leaves]
                # let's get the average entropy of the leaves
                initial_avg_entropy = np.mean(initial_entropies)
                final_avg_entropy = np.mean(final_entropies)
                # calculate the difference in entropy
                entropy_diff = final_avg_entropy - initial_avg_entropy
                user = user_f.split('/')[-1]
                data.append(Data(domain_name, user, entropy_diff))
    with open('data/experiments/human_modifies_tree/entropy_diffs.csv', 'w') as f:
        f.write('domain_name,user_id,entropy_diff\n')
        for d in data:
            f.write('{},{},{}\n'.format(d.domain_name, d.user_id, d.entropy_diff))