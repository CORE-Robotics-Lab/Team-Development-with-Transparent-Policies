import torch
import gym
from stable_baselines3.common.torch_layers import FlattenExtractor
from ipm.algos import ddt_sac_policy
from ipm.algos import ddt_ppo_policy
from stable_baselines3 import PPO
from ipm.models.idct import IDCT
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.preprocessing import get_action_dim
import numpy as np
from ipm.algos.genetic_algorithm import GA_DT_Optimizer
from ipm.models.decision_tree import decision_tree_to_sparse_ddt
import sys
sys.path.insert(0, '../../overcooked_ai/src/overcooked_ai_py')
sys.path.insert(0, '../../overcooked_ai/src')
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action



def estimate_performance(model, env_name: str):
    if env_name == 'cartpole':
        env = gym.make('CartPole-v1')
    elif env_name == 'overcooked':
        mdp = OvercookedGridworld.from_layout_name(layout_name='cramped_room')
        env = OvercookedEnv.from_mdp(mdp)
    else:
        raise "Unknown environment"

    num_eps = 100
    curr_ep = 0
    total_reward = 0.0
    avg_reward = []

    obs = env.reset()
    while curr_ep < num_eps:
        action = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()
            curr_ep += 1
            avg_reward.append(total_reward)
            total_reward = 0.0

    print('Average reward: ', np.mean(avg_reward))


def finetune_model(initial_model: IDCT, algo:str='ppo', env_name: str = 'cartpole'):
    if env_name == 'cartpole':
        env = gym.make('CartPole-v1')
    elif env_name == 'overcooked':
        mdp = OvercookedGridworld.from_layout_name(layout_name='cramped_room')
        env = OvercookedEnv.from_mdp(mdp)
        # TODO: Fix :). This is just temporary, need to wrap over cooked env into something
        # stable baselines can use
        return initial_model
    else:
        raise "Unknown environment"

    if algo == 'ga':
        # TODO: Bug in IDCT -> DT. Performs way worse with oracle model.
        # TODO: Create code from DT -> array
        # TODO: Create code from array -> DT
        optimizer = GA_DT_Optimizer(n_decision_nodes=5, n_leaves=6, env=env)
        optimizer.run(initial_model)
        return decision_tree_to_sparse_ddt(optimizer.best_tree)
    elif algo == 'ppo':
        ppo_lr = 0.0003
        ppo_batch_size = 64
        ppo_n_steps = 2000

        ddt_kwargs = {
            'num_leaves': len(initial_model.leaf_init_information),
            'hard_node': False,
            'weights': initial_model.layers,
            'alpha': initial_model.alpha,
            'comparators': initial_model.comparators,
            'leaves': initial_model.leaf_init_information,
            'fixed_idct': False,
            'device': 'cuda',
            'argmax_tau': 1.0,
            'ddt_lr': 0.001, # this param is irrelevant for the IDCT
            'use_individual_alpha': True,
            'l1_reg_coeff': 1.0,
            'l1_reg_bias': 1.0,
            'l1_hard_attn': 1.0,
            'use_gumbel_softmax': False,
            'alg_type': 'ppo'
        }

        features_extractor = FlattenExtractor
        policy_kwargs = dict(features_extractor_class=features_extractor, ddt_kwargs=ddt_kwargs)

        model = PPO("DDT_PPOPolicy", env,
                    n_steps=ppo_n_steps,
                    # batch_size=args.batch_size,
                    # buffer_size=args.buffer_size,
                    learning_rate=ppo_lr,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log='log',
                    gamma=0.99,
                    verbose=1,
                    seed=1
                    )

        model.learn(total_timesteps=2000)
        return model.policy.action_net

def get_idct(env_name: str):
    if env_name == 'cartpole':
        env = gym.make('CartPole-v1')

        alpha = torch.Tensor([[-1], [1], [-1], [-1], [-1]])

        leaves = [[[2], [0], [2, -2]], [[], [0, 2], [-2, 2]], [[0, 1, 3], [], [2, -2]], [[0, 1], [3], [-2, 2]],
                  [[0, 4], [1], [2, -2]], [[0], [1, 4], [-2, 2]]]

        weights = torch.Tensor([
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])

        comparators = torch.Tensor([[0.03], [-0.03], [0], [0], [0]])
        input_dim = get_obs_shape(env.observation_space)[0]
        output_dim = get_action_dim(env.action_space)

        return IDCT(input_dim=input_dim,
                    output_dim=output_dim,
                    hard_node=False,
                    device='cuda',
                    argmax_tau=1.0,
                    use_individual_alpha=True,
                    use_gumbel_softmax=False,
                    alg_type='ppo',
                    weights=weights,
                    comparators=comparators,
                    alpha=alpha,
                    fixed_idct=True,
                    leaves=leaves)

    elif env_name == 'overcooked':
        mdp = OvercookedGridworld.from_layout_name(layout_name='cramped_room')
        env = OvercookedEnv.from_mdp(mdp)

        # TODO: Dynamically determine input and output dims
        input_dim = 96
        output_dim = len(Action.ALL_ACTIONS)

        return IDCT(input_dim=input_dim,
                    output_dim=output_dim,
                    hard_node=False,
                    device='cuda',
                    argmax_tau=1.0,
                    use_individual_alpha=True,
                    use_gumbel_softmax=False,
                    alg_type='ppo',
                    weights=None,
                    comparators=None,
                    alpha=None,
                    fixed_idct=False,
                    leaves=8)

    else:
        raise "Unknown environment"
