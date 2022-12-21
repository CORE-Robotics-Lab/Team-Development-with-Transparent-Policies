from ipm.algos.genetic_algorithm import GA_DT_Optimizer
import gym

if __name__ == '__main__':
    cartpole = gym.make('CartPole-v1')
    ga = GA_DT_Optimizer(n_decision_nodes=7, n_leaves=8, env=cartpole)
    ga.run()