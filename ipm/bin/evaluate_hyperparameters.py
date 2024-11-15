import argparse
import configparser
import os
import pickle
import time
from datetime import datetime
import sys

sys.path.insert(0, '/home/rohanpaleja/PycharmProjects/ipm/')
sys.path.insert(0, '/home/rohanpaleja/PycharmProjects/ipm/ipm/')
sys.path.insert(0, '/home/rohanpaleja/PycharmProjects/ipm/overcooked_ai/')
sys.path.insert(0, '/home/rohanpaleja/PycharmProjects/ipm/overcooked_ai/src/')

import numpy as np
import pygame
import torch
from ipm.bin.utils import play_episode_together, play_episode_together_get_states
from ipm.gui.experiment_gui_utils import SettingsWrapper
from ipm.models.bc_agent import StayAgent
from ipm.models.decision_tree import sparse_ddt_to_decision_tree
from ipm.models.human_model import HumanModel
from ipm.models.robot_model import RobotModel
from ipm.overcooked.overcooked_envs import OvercookedPlayWithFixedPartner, OvercookedJointRecorderEnvironment
from stable_baselines3 import PPO
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune various hyperaparameters')
    parser.add_argument('--config_file', help='Config file', type=str, default='data/test_hyperparams.ini')
    parser.add_argument('--layout', help='layout', type=str, default='two_rooms_narrow')
    args = parser.parse_args()

    # load in config file
    config = configparser.ConfigParser()
    config.read(args.config_file)

    layout = args.layout
    prior_iteration_models = config.get('main', 'prior_iteration_models')
    n_episode_samples = 50 # config.getint('main', 'n_episode_samples')
    n_random_seeds = config.getint('main', 'n_random_seeds')

    rpo = config.getboolean('main', 'rpo')
    rpo_ga = config.getboolean('main', 'rpo_ga')
    rpo_rl = config.getboolean('main', 'rpo_rl')
    rpo_random_initial_idct = config.getboolean('main', 'rpo_random_initial_idct')

    rpo_rl_n_steps = config.getint('main', 'rpo_rl_n_steps')
    rpo_rl_lr = config.getfloat('main', 'rpo_rl_lr')
    rpo_rl_only_optimize_leaves = config.getboolean('main', 'rpo_rl_only_optimize_leaves')
    random_seed = config.getint('main', 'rpo_rl_random_seed')

    data_folder = os.path.join('data',
                               'tuning')

    folder = os.path.join(data_folder, layout)
    if not os.path.exists(folder):
        os.makedirs(folder)

    pygame.init()

    all_rewards_initial = [[] for _ in range(n_random_seeds)]
    all_rewards_finetuned_human_policy = [[] for _ in range(n_random_seeds)]
    all_initial_ce_finetuned_human_policy = [[] for _ in range(n_random_seeds)]
    all_final_ce_finetuned_human_policy = [[] for _ in range(n_random_seeds)]
    all_rewards_finetuned_intent = [[] for _ in range(n_random_seeds)]
    all_initial_ce_finetuned_intent = [[] for _ in range(n_random_seeds)]
    all_final_ce_finetuned_intent = [[] for _ in range(n_random_seeds)]
    all_rewards_finetuned_robot_policy = [[] for _ in range(n_random_seeds)]
    all_training_times_robot_policy = []

    np.random.seed(0)
    torch.manual_seed(0)

    print('-----------------------')
    print('STARTING RANDOM SEED: ', random_seed)
    print('-----------------------')

    settings = SettingsWrapper()

    dummy_env = OvercookedPlayWithFixedPartner(partner=StayAgent(), layout_name=layout,
                                               reduced_state_space_ego=True, reduced_state_space_alt=True,
                                               use_skills_ego=True, use_skills_alt=True)

    initial_policy_path = '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/data/warm_start.tar' # os.path.join('data', 'prior_tree_policies', layout + '.tar')
    initial_policy_path3 = os.path.join('data', 'fcp', layout + '.tar')
    intent_model_path = os.path.join('data', 'intent_models', layout + '.pt')
    # data_file = prior_iteration_data
    initial_policy_path2 = '/home/rohanpaleja/Desktop/ego_alt_idct_2.tar'
    model = PPO("MlpPolicy", dummy_env)
    weights = torch.load(initial_policy_path2)
    # weights = torch.load(initial_policy_path3)
    model.policy.load_state_dict(weights['alt_state_dict'])
    human_ppo_policy = model.policy
    human_policy = HumanModel(layout, human_ppo_policy)
    weights2 = torch.load(initial_policy_path3)

    # for fcp
    robot_policy_nn = PPO("MlpPolicy", dummy_env)

    robot_policy_nn.policy.load_state_dict(weights2['ego_state_dict'])
    robot_policy_nn2 = robot_policy_nn.policy
    robot_policy_nn_2 = HumanModel(layout, robot_policy_nn2)

    input_dim = dummy_env.observation_space.shape[0]
    output_dim = dummy_env.n_actions_alt


    # why is this robot policy
    ddt_kwargs = {
        'num_leaves': 128,
        'hard_node': False,
        'weights': None,
        'alpha': 1.0,
        'comparators': None,
        'leaves': None,
        'fixed_idct': False,
        'device': 'cuda',
        'argmax_tau': 1.0,
        'ddt_lr': 0.001,  # this param is irrelevant for the IDCT
        'use_individual_alpha': True,
        'l1_reg_coeff': 1.0,
        'l1_reg_bias': 1.0,
        'l1_hard_attn': 1.0,
        'use_gumbel_softmax': False,
        'alg_type': 'ppo'
    }
    from stable_baselines3.common.torch_layers import FlattenExtractor

    features_extractor = FlattenExtractor
    policy_kwargs = dict(features_extractor_class=features_extractor, ddt_kwargs=ddt_kwargs)

    # model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
    # model.learn(1000)

    # model = PPO("DDT_PPOPolicy", dummy_env,
    #             # n_steps=25000,
    #             batch_size=10000,
    #             # buffer_size=args.buffer_size,
    #             learning_rate=0.0003,
    #             policy_kwargs=policy_kwargs,
    #             tensorboard_log='log',
    #             gamma=0.99,
    #             verbose=1,
    #             seed=1
    #             )
    # robot_policy = PPO("DDT_PPOPolicy", dummy_env)
    weights = torch.load(initial_policy_path)
    # robot_policy.policy.load_state_dict(weights['alt_state_dict'])
    robot_policy = RobotModel(layout=layout,
                               idct_policy_filepath=initial_policy_path,
                               human_policy=human_policy,
                               input_dim=input_dim,
                               output_dim=output_dim,
                               randomize_initial_idct=False,
                               only_optimize_leaves=rpo_rl_only_optimize_leaves,
                               with_key=False)

    joint_environment = OvercookedJointRecorderEnvironment(layout_name=layout, seed_num=0,
                                                           ego_idx=0,
                                                           failed_skill_rew=0,
                                                           reduced_state_space_ego=True,
                                                           reduced_state_space_alt=True,
                                                           use_skills_ego=True,
                                                           use_skills_alt=True,
                                                           use_true_intent_ego=False,
                                                           use_true_intent_alt=False,
                                                           double_cook_times=False)

    print('-----------------------')
    print('PLAYING INITIAL EPISODES')
    print('-----------------------')

    current_policy, tree_info = sparse_ddt_to_decision_tree(robot_policy.robot_idct_policy,
                                                                 robot_policy.env)

    for i in tqdm(range(n_episode_samples)):
        reward = play_episode_together(joint_environment, human_policy, current_policy, render=True)
        all_rewards_initial[0].append(reward)
    #
    # data_file = 'data/iteration_0.tar'
    # human_policy.translate_recent_data_to_labels(recent_data_loc=data_file)
    #
    # if hpo:
    #     print('-----------------------')
    #     print('FINETUNING HUMAN POLICY')
    #     print('-----------------------')
    #     initial_ce, final_ce = human_policy.finetune_human_ppo_policy(learning_rate=hpo_lr, n_epochs=hpo_n_epochs)
    #     all_initial_ce_finetuned_human_policy[random_seed].append(initial_ce)
    #     all_final_ce_finetuned_human_policy[random_seed].append(final_ce)
    #
    #     print('-----------------------')
    #     print('PLAYING EPISODES AFTER FINETUNING HUMAN POLICY')
    #     print('-----------------------')
    #
    #     for i in tqdm(range(n_episode_samples)):
    #         reward = play_episode_together(joint_environment, human_policy, robot_policy, render=False)
    #         all_rewards_finetuned_human_policy[random_seed].append(reward)
    #
    # current_policy, tree_info = sparse_ddt_to_decision_tree(robot_policy.robot_idct_policy,
    #                                                         robot_policy.env)
    # intent_model = robot_policy.intent_model
    #
    # robot_policy.translate_recent_data_to_labels(recent_data_loc=data_file)
    # if ipo:
    #     print('-----------------------')
    #     print('FINETUNING INTENT MODEL')
    #     print('-----------------------')
    #
    #     initial_ce, final_ce = robot_policy.finetune_intent_model(learning_rate=ipo_lr, n_epochs=ipo_n_epochs)
    #     all_initial_ce_finetuned_intent[random_seed].append(initial_ce)
    #     all_final_ce_finetuned_intent[random_seed].append(final_ce)
    #
    #     print('-----------------------')
    #     print('PLAYING EPISODES AFTER FINETUNING INTENT MODEL')
    #     print('-----------------------')
    #
    #     for i in tqdm(range(n_episode_samples)):
    #         reward = play_episode_together(joint_environment, human_policy, robot_policy, render=False)
    #         all_rewards_finetuned_intent[random_seed].append(reward)

    # load in human model after its updated and robot model after its intent is updated
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if rpo:
        if rpo_ga and rpo_rl:
            algorithm_choice = 'ga+rl'
        elif rpo_ga:
            algorithm_choice = 'ga'
        elif rpo_rl:
            algorithm_choice = 'rl'
        else:
            raise ValueError('Invalid rpo algorithm choice')

        print('-----------------------')
        print('FINETUNING ROBOT MODEL')
        print('-----------------------')

        start_time = time.time()
        robot_policy.finetune_robot_idct_policy(rl_n_steps=rpo_rl_n_steps,
                                                rl_learning_rate=rpo_rl_lr,
                                                algorithm_choice=algorithm_choice,
                                                unique_id=args.config_file[-5]
                                                )
        end_time = time.time()
        all_training_times_robot_policy.append(end_time - start_time)

        print('-----------------------')
        print('PLAYING EPISODES AFTER FINETUNING ROBOT MODEL')
        print('-----------------------')

        np.random.seed(0)
        torch.manual_seed(0)
        # want to test with more episodes than train
        for i in tqdm(range(n_episode_samples * 2)):
            reward = play_episode_together(joint_environment, human_policy, robot_policy, render=False)
            all_rewards_finetuned_robot_policy[0].append(reward)


    def get_avg_and_std(all_rewards):
        arr = np.array(all_rewards).flatten()
        return np.mean(arr), np.std(arr)


    avg_rewards_initial, std_rewards_initial = get_avg_and_std(all_rewards_initial)
    avg_rewards_rpo, std_rewards_rpo = get_avg_and_std(all_rewards_finetuned_robot_policy)

    initial_performance_str = 'Average reward for initial policy: ' + \
                              str(round(avg_rewards_initial, 2)) + ' +/- ' + str(round(std_rewards_initial, 2)) + '\n\n'

    rpo_performance_str = 'Average reward after fine-tuning robot policy: ' + \
                          str(round(avg_rewards_rpo, 2)) + ' +/- ' + str(round(std_rewards_rpo, 2)) + '\n\n'

    rpo_avg_training_time = np.mean(all_training_times_robot_policy)
    rpo_training_str = 'Average training time for robot policy: ' + str(round(rpo_avg_training_time, 2)) + 's\n'

    print(initial_performance_str)

    if rpo:
        print(rpo_performance_str)
        print(rpo_training_str)
        print(args.config_file)

    torch.save({'robot_idct_policy': robot_policy.robot_idct_policy.state_dict(),
                'init_reward': (avg_rewards_initial, std_rewards_initial),
                'end_reward': (avg_rewards_rpo, std_rewards_rpo)},
               'data' + args.config_file[-5] + '.tar')
