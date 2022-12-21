import pygad
import numpy as np
import random
from ipm.models.decision_tree import DecisionTree, sparse_ddt_to_decision_tree
import gym

class GA_DT_Optimizer:
    def __init__(self, n_decision_nodes, n_leaves, env, seed: int = 1, initial_population=None):
        random.seed(seed)
        np.random.seed(seed)
        # env.seed(seed)
        self.seed = seed

        self.N_EPISODES_EVAL = 10
        self.num_generations = 100 # Number of generations.
        self.num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.
        self.sol_per_pop = 30 # Number of solutions in the population.

        self.n_decision_nodes = n_decision_nodes
        self.n_leaves = n_leaves

        self.env = env
        self.n_vars = env.observation_space.shape[0]
        self.var_space = list(range(self.n_vars))
        self.compare_sign_space = [0, 1]

        if isinstance(env.action_space, gym.spaces.Box):
            self.action_space = {'low': -1, 'high': 1}
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space = list(range(env.action_space.n))
        else:
            raise "Unknown action space"

        if isinstance(env.observation_space, gym.spaces.Box):
            self.value_space = [{'low': -1, 'high': 1} for _ in range(self.n_vars)]
            self.lows = env.observation_space.low
            self.highs = env.observation_space.high
        elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            self.value_space = [list(range(env.observation_space[i].n)) for i in range(self.n_vars)]
            self.lows = np.array([env.observation_space[i].low for i in range(self.n_vars)])
            self.highs = np.array([env.observation_space[i].high for i in range(self.n_vars)])
        else:
            raise "Unknown observation space"

        self.set_gene_types()

    def set_gene_types(self):
        self.gene_space = []
        self.gene_types = []
        for i in range(self.n_decision_nodes):
            self.gene_space.append(self.var_space)
            self.gene_types.append(int)
            self.gene_space.append(self.compare_sign_space)
            self.gene_types.append(int)
            self.gene_space.append(self.value_space[0])
            # for multi discrete, we need to be careful
            # probably need to use max value from it

            # check if value_space is a list
            if isinstance(self.value_space[0], list):
                self.gene_types.append(int)
            else:
                self.gene_types.append(float)

        for i in range(self.n_leaves):
            self.gene_space.append(self.action_space)
            self.gene_types.append(int)

        self.num_genes = len(self.gene_space)

    def evaluate_model(self, model, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.N_EPISODES_EVAL
        all_episode_rewards = []
        for i in range(num_episodes):
            done = False
            obs = self.env.reset()
            total_reward = 0.0
            while not done:
                # _states are only useful when using LSTM policies
                action = model.predict(obs)
                # here, action, rewards and dones are arrays
                # because we are using vectorized env
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                # all_rewards_per_timestep[seed].append(last_fitness)
            all_episode_rewards.append(total_reward)
        return np.mean(all_episode_rewards)

    def get_random_genes(self):
        genes = []
        for i in range(self.num_genes):
            if self.gene_types[i] == int:
                genes.append(random.choice(self.gene_space[i]))
            else:
                genes.append(random.uniform(self.gene_space[i]['low'], self.gene_space[i]['high']))
        return genes

    def run(self, idct=None):
        # alternative: use partial funcs
        def on_generation(ga_instance):
            generation = ga_instance.generations_completed
            self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
            if generation % 10 == 0:
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
            model = DecisionTree(solution, n_decision_nodes=self.n_decision_nodes, n_leaves=self.n_leaves,
                                 lows=self.lows, highs=self.highs)
            return self.evaluate_model(model)

        if idct is not None:
            # initial_population = [sparse_ddt_to_decision_tree(idct, self.env).node_values for _ in range(self.sol_per_pop)]
            initial_population = [sparse_ddt_to_decision_tree(idct, self.env).node_values]
            for i in range(self.sol_per_pop - 1):
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
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)