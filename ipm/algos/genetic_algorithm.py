import argparse
import os

import pygad
import numpy as np
import random
import sys

import torch
from stable_baselines3 import PPO
from sklearn.tree import DecisionTreeClassifier
from ipm.models.decision_tree import DecisionTree, sparse_ddt_to_decision_tree, LeafNode
import gym

from ipm.models.decision_tree_structure import DecisionTreeStructure


class GA_DT_Structure_Optimizer:
    def __init__(self, trajectories_file, initial_depth, max_depth, n_vars, n_actions, num_gens=20, seed: int = 1, initial_population=None):
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

        trajectories = torch.load(trajectories_file)
        self.X, self.Y = trajectories['alt_inputs_npy'], trajectories['alt_actions']
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        self.num_generations = num_gens # Number of generations.
        self.num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
        self.sol_per_pop = 10 # Number of solutions in the population.

        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.current_depth = initial_depth

        self.n_vars = n_vars
        self.var_space = list(range(self.n_vars))
        self.best_solution = None

        self.action_space = n_actions
        self.value_space = [0, 1]
        self.set_gene_types()

        self.initial_population = []

    def set_gene_types(self):
        dt = DecisionTreeStructure(num_vars=self.n_vars, depth=self.current_depth)
        self.gene_space = dt.gene_space
        self.num_genes = len(self.gene_space)
        self.gene_types = [int for _ in range(self.num_genes)]

    def compute_entropy(self, y_at_the_leaf):
        """
        Compute the entropy of the leaf
        :param y_at_the_leaf: (np.array) array of labels at the leaf
        :return: (float) entropy of the leaf
        """
        total = len(y_at_the_leaf)
        if total == 0:
            return 0
        entropy = 0
        for action in range(self.action_space):
            count = np.sum(y_at_the_leaf == action)
            if count == 0:
                continue
            prob = count / total
            entropy += -prob * np.log2(prob)
        return entropy

    def evaluate_model(self, model):
        # traverse the tree and split up the data, compute the entropies at each leaf and add them up
        total_entropies = 0.0

        # each item is a tuple of the current node, the X data at that node, and the Y data at that node
        q = [(model.root, self.X, self.Y)]
        while len(q) > 0:
            node, X, Y = q.pop(0)
            if node is None: # if no child, we know it will be a leaf
                total_entropies += self.compute_entropy(Y)
            else:
                # split the data according to the value to compare at the current node
                feature_idx = node.var_idx
                X_mask = X[:, feature_idx] <= node.comp_val
                X_left, X_right = X[X_mask], X[~X_mask]
                Y_left, Y_right = Y[X_mask], Y[~X_mask]
                # add children to the q
                q.append((node.left, X_left, Y_left))
                q.append((node.right, X_right, Y_right))
        return total_entropies

    def populate_leaves(self, model):
        # traverse the tree and split up the data,
        # where there are no children, populate the leaf with the action probabilities

        dt = DecisionTree(num_vars=self.n_vars, num_actions=self.action_space, depth=self.current_depth)
        # let's now copy over our values to this new tree WITH leaves

        q = [(model.root, dt.root)]
        while len(q) > 0:
            node, new_node = q.pop(0)
            if node is not None:
                new_node.var_idx = node.var_idx
                new_node.comp_val = node.comp_val
                new_node.normal_ordering = node.normal_ordering
                q.append((node.left, new_node.left))
                q.append((node.right, new_node.right))

        # each item is a tuple of the current node, the X data at that node, and the Y data at that node
        q = [(dt.root, self.X, self.Y)]
        while len(q) > 0:
            node, X, Y = q.pop(0)
            if type(node) == LeafNode:
                # compute the probabilities of each action
                total = len(Y)
                if total == 0:
                    node.action = np.zeros(self.action_space)
                else:
                    node.action = np.zeros(self.action_space)
                    for action in range(self.action_space):
                        count = np.sum(Y == action)
                        node.action[action] = count / total
            else:
                # split the data according to the value to compare at the current node
                feature_idx = node.var_idx
                X_mask = X[:, feature_idx] <= node.comp_val
                X_left, X_right = X[X_mask], X[~X_mask]
                Y_left, Y_right = Y[X_mask], Y[~X_mask]
                # add children to the q
                q.append((node.left, X_left, Y_left))
                q.append((node.right, X_right, Y_right))
        return dt

    def get_random_genes(self):
        dt = DecisionTreeStructure(num_vars=self.n_vars, depth=self.current_depth)
        return dt.node_values

    def run(self):
        # alternative: use partial funcs
        def on_generation(ga_instance):
            generation = ga_instance.generations_completed
            self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
            if generation % 1 == 0:
                print("Seed = {seed}, Generation = {generation}, Fitness  = {fitness}".format(seed=self.seed,
                                                                                              generation=generation,
                                                                                              fitness=self.last_fitness))
            # fitness_across_all[seed, generation - 1] = self.last_fitness
            # print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))

        def fitness_func(solution, solution_idx):
            """
            Evaluate a RL agent
            :param model: (BaseRLModel object) the RL Agent
            :param num_episodes: (int) number of episodes to evaluate it
            :return: (float) Mean reward for the last num_episodes
            """
            # This function will only work for a single Environment
            model = DecisionTreeStructure(node_values=solution, depth=self.current_depth, num_vars=self.n_vars)
            return self.evaluate_model(model)

        initial_population = None

        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=self.num_parents_mating,
                               sol_per_pop=self.sol_per_pop,
                               num_genes=self.num_genes,
                               fitness_func=fitness_func,
                               on_generation=on_generation,
                               gene_space=self.gene_space,
                               initial_population=initial_population,
                               parent_selection_type="rank",
                               #    crossover_type='two_points',
                               #    crossover_probability=0.5,
                               random_seed=self.seed,
                               gene_type=self.gene_types)

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()

        # Returning the details of the best solution.
        self.best_solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

        self.final_trees = []
        # for the final population, let's create the trees and populate the leaves
        for solution in ga_instance.population:
            dt = DecisionTreeStructure(node_values=solution, depth=self.current_depth, num_vars=self.n_vars)
            dt_with_leaves = self.populate_leaves(dt)
            self.final_trees.append(dt_with_leaves)

class GA_DT_Optimizer:
    def __init__(self, initial_depth, max_depth, env,
                 num_gens=20,
                 seed: int = 1, initial_population=None):
        random.seed(seed)
        np.random.seed(seed)
        # env.seed(seed)
        self.seed = seed

        self.N_EPISODES_EVAL = 3
        self.num_generations = num_gens # Number of generations.
        self.num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
        self.sol_per_pop = 10 # Number of solutions in the population.

        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.current_depth = initial_depth

        self.env = env
        self.n_vars = env.observation_space.shape[0]
        self.var_space = list(range(self.n_vars))
        self.best_solution = None

        self.action_space = list(range(env.action_space.n))
        self.value_space = [0, 1]
        self.set_gene_types()

        if type(initial_population) == str:
            # then we use behavioral cloning to populate the initial population
            # in this case, initial population is a path to the models
            # we only want the best ones
            self.distill_self_play_policies(initial_population)
        elif type(initial_population) == list:
            self.initial_population = initial_population
        elif initial_population is None:
            self.initial_population = []

    def distill_self_play_policies(self, filepath):
        # then we use behavioral cloning to populate the initial population
        # in this case, initial population is a path to the models
        # we only want the best ones
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
        custom_objects = {}
        if newer_python_version:
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }

        threshold = 30.0
        self.initial_population = []
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if file.endswith('final_model.zip'):
                    agent = PPO.load(os.path.join(root, file), custom_objects=custom_objects)
                    avg_rew, distilled_policy = self.distill_policy(agent, evaluate_bc=True)
                    if avg_rew > threshold:
                        self.initial_population.append(distilled_policy.node_values)

    def distill_policy(self, policy, evaluate_bc=False):
        rew, (obs, acts) = self.evaluate_model(policy, num_episodes=5, include_obs_acts=True)
        print('Raw model performance:', rew)
        obs = np.array(obs)
        acts = np.array(acts)
        # create full tree in sklearn
        decision_tree = DecisionTreeClassifier(max_depth=self.current_depth)
        decision_tree.fit(obs, acts)

        # extract the node values from the decision tree
        distilled_model = DecisionTree.from_sklearn(decision_tree, self.n_vars, self.env.action_space.n)
        if evaluate_bc:
            rew_distilled = self.evaluate_model(distilled_model, num_episodes=5,
                                                include_obs_acts=False, identical_model=decision_tree)
            print('BC model performance:', rew_distilled)
        return rew, distilled_model

    def set_gene_types(self):
        dt = DecisionTree(self.env.observation_space.shape[0], self.env.action_space.n, depth=self.current_depth)
        self.gene_space = dt.gene_space
        self.num_genes = len(self.gene_space)
        self.gene_types = [int for _ in range(self.num_genes)]

    def evaluate_model(self, model, num_episodes=None, include_obs_acts=False, identical_model=None):
        if num_episodes is None:
            num_episodes = self.N_EPISODES_EVAL
        all_episode_rewards = []
        all_episode_obs = []
        all_episode_acts = []
        same_actions = True
        for i in range(num_episodes):
            done = False
            obs = self.env.reset()
            total_reward = 0.0
            while not done:
                # _states are only useful when using LSTM policies
                all_episode_obs.append(obs)
                if type(model) == PPO:
                    action, _states = model.predict(self.env.ego_raw_obs, deterministic=True)
                else:
                    action = model.predict(obs)
                    if identical_model is not None:
                        obs = obs.reshape(1, -1)
                        identical_action = identical_model.predict(obs)
                        if identical_action[0] != action:
                            same_actions = False
                all_episode_acts.append(action)
                # here, action, rewards and dones are arrays
                # because we are using vectorized env
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                # all_rewards_per_timestep[seed].append(last_fitness)
            all_episode_rewards.append(total_reward)
        if identical_model is not None:
            print('Same actions?', same_actions)
        if include_obs_acts:
            return np.mean(all_episode_rewards), (all_episode_obs, all_episode_acts)
        else:
            return np.mean(all_episode_rewards)

    def get_random_genes(self):
        dt = DecisionTree(self.env.observation_space.shape[0], self.env.action_space.n, depth=self.current_depth)
        return dt.node_values

    def run(self, idct=None):
        # alternative: use partial funcs
        def on_generation(ga_instance):
            generation = ga_instance.generations_completed
            self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
            if generation % 1 == 0:
                print("Seed = {seed}, Generation = {generation}, Fitness  = {fitness}".format(seed=self.seed,
                                                                                              generation=generation,
                                                                                              fitness=self.last_fitness))
            # fitness_across_all[seed, generation - 1] = self.last_fitness
            # print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))

        def fitness_func(solution, solution_idx):
            """
            Evaluate a RL agent
            :param model: (BaseRLModel object) the RL Agent
            :param num_episodes: (int) number of episodes to evaluate it
            :return: (float) Mean reward for the last num_episodes
            """
            # This function will only work for a single Environment
            model = DecisionTree(node_values=solution, depth=self.current_depth, num_vars=self.n_vars, num_actions=self.env.action_space.n)
            return self.evaluate_model(model, num_episodes=self.N_EPISODES_EVAL)

        if idct is not None:
            initial_population = self.initial_population
            # initial_population = [sparse_ddt_to_decision_tree(idct, self.env).node_values for _ in range(self.sol_per_pop)]
            # initial_population = [sparse_ddt_to_decision_tree(idct, self.env).node_values]
            for i in range(self.sol_per_pop - 1):
                initial_population.append(self.get_random_genes())
            initial_population.append(self.get_random_genes())
        else:
            initial_population = None

        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=self.num_parents_mating,
                               sol_per_pop=self.sol_per_pop,
                               num_genes=self.num_genes,
                               fitness_func=fitness_func,
                               on_generation=on_generation,
                               gene_space=self.gene_space,
                               initial_population=initial_population,
                               parent_selection_type="rank",
                               #    crossover_type='two_points',
                               #    crossover_probability=0.5,
                               random_seed=self.seed,
                               gene_type=self.gene_types)

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()

        # Returning the details of the best solution.
        self.best_solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

    if __name__ == '__main__':
        # this will execute the genetic algorithms depending on initial policy (if none, random)
        # and the collection of expert trajectories
        # we also optimize the idct later
        parser = argparse.ArgumentParser
        parser.add_argument('--env', type=str, default='CartPole-v1')
