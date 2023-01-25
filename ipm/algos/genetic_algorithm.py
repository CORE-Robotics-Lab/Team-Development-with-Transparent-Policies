import os

import pygad
import numpy as np
import random

from stable_baselines3 import PPO
from sklearn.tree import DecisionTreeClassifier
from ipm.models.decision_tree import DecisionTree, sparse_ddt_to_decision_tree
import gym

class GA_DT_Optimizer:
    def __init__(self, initial_depth, max_depth, env,
                 num_gens=20,
                 seed: int = 1, initial_population=None):
        random.seed(seed)
        np.random.seed(seed)
        # env.seed(seed)
        self.seed = seed

        self.N_EPISODES_EVAL = 2
        self.num_generations = num_gens # Number of generations.
        self.num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.
        self.sol_per_pop = 20 # Number of solutions in the population.

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
            self.initial_population = []
            for root, dirs, files in os.walk(initial_population):
                for file in files:
                    if file.endswith('final_model.zip'):
                        agent = PPO.load(os.path.join(root, file))
                        self.initial_population.append(self.distill_policy(agent))

    def distill_policy(self, policy):
        rew, (obs, acts) = self.evaluate_model(policy)
        obs = np.array(obs)
        acts = np.array(acts)
        # create full tree in sklearn
        decision_tree = DecisionTreeClassifier(max_depth=self.current_depth)
        decision_tree.fit(obs, acts)
        # extract the node values from the decision tree
        node_values = decision_tree.tree_.value


    def set_gene_types(self):
        dt = DecisionTree(self.env.observation_space.shape[0], self.env.action_space.n, depth=self.current_depth)
        self.gene_space = dt.gene_space
        self.gene_types = [int for _ in range(len(self.gene_space))]
        self.num_genes = len(self.gene_space)

    def evaluate_model(self, model, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.N_EPISODES_EVAL
        all_episode_rewards = []
        all_episode_obs = []
        all_episode_acts = []
        for i in range(num_episodes):
            done = False
            obs = self.env.reset()
            total_reward = 0.0
            while not done:
                # _states are only useful when using LSTM policies
                all_episode_obs.append(obs)
                action = model.predict(obs)
                all_episode_acts.append(action)
                # here, action, rewards and dones are arrays
                # because we are using vectorized env
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                # all_rewards_per_timestep[seed].append(last_fitness)
            all_episode_rewards.append(total_reward)
        return np.mean(all_episode_rewards), (np.array(all_episode_obs), np.array(all_episode_acts))

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
            rew, (_, _) = self.evaluate_model(model, num_episodes=self.N_EPISODES_EVAL)

        if idct is not None:
            initial_population = []
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