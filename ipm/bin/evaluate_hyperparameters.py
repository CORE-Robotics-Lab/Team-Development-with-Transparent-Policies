import argparse
import configparser
import os
import time
from datetime import datetime

import numpy as np
import pygame
import torch
from ipm.bin.utils import play_episode_together
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
    args = parser.parse_args()

    # load in config file
    config = configparser.ConfigParser()
    config.read(args.config_file)

    layout = config.get('main', 'layout')
    prior_iteration_data = config.get('main', 'prior_iteration_data')
    n_episode_samples = config.getint('main', 'n_episode_samples')
    n_random_seeds = config.getint('main', 'n_random_seeds')

    hpo = config.getboolean('main', 'hpo')
    hpo_lr = config.getfloat('main', 'hpo_lr')
    hpo_n_epochs = config.getint('main', 'hpo_n_epochs')

    ipo = config.getboolean('main', 'ipo')
    ipo_lr = config.getfloat('main', 'ipo_lr')
    ipo_n_epochs = config.getint('main', 'ipo_n_epochs')

    rpo = config.getboolean('main', 'rpo')
    rpo_ga = config.getboolean('main', 'rpo_ga')
    rpo_rl = config.getboolean('main', 'rpo_rl')
    rpo_random_initial_idct = config.getboolean('main', 'rpo_random_initial_idct')

    rpo_ga_data_file = config.get('main', 'rpo_ga_data_file')
    rpo_ga_depth = config.getint('main', 'rpo_ga_depth')
    rpo_ga_n_gens = config.getint('main', 'rpo_ga_n_gens')
    rpo_ga_n_pop = config.getint('main', 'rpo_ga_n_pop')
    rpo_ga_n_parents_mating = config.getint('main', 'rpo_ga_n_parents_mating')
    rpo_ga_crossover_prob = config.getfloat('main', 'rpo_ga_crossover_prob')
    rpo_ga_crossover_type = config.get('main', 'rpo_ga_crossover_type')
    rpo_ga_mutation_prob = config.getfloat('main', 'rpo_ga_mutation_prob')
    rpo_ga_mutation_type = config.get('main', 'rpo_ga_mutation_type')

    rpo_rl_n_steps = config.getint('main', 'rpo_rl_n_steps')
    rpo_rl_lr = config.getfloat('main', 'rpo_rl_lr')
    rpo_rl_only_optimize_leaves = config.getboolean('main', 'rpo_rl_only_optimize_leaves')

    data_folder = os.path.join('data',
                               'tuning')

    folder = os.path.join(data_folder, layout)
    if not os.path.exists(folder):
        os.makedirs(folder)

    pygame.init()

    all_rewards_initial = [[] for _ in range(n_random_seeds)]
    all_rewards_finetuned_human_policy = [[] for _ in range(n_random_seeds)]
    all_rewards_finetuned_intent = [[] for _ in range(n_random_seeds)]
    all_rewards_finetuned_robot_policy = [[] for _ in range(n_random_seeds)]
    all_training_times_robot_policy = []

    for random_seed in range(n_random_seeds):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        print('-----------------------')
        print('STARTING RANDOM SEED: ', random_seed)
        print('-----------------------')

        settings = SettingsWrapper()

        dummy_env = OvercookedPlayWithFixedPartner(partner=StayAgent(), layout_name=layout,
                                                   behavioral_model='dummy',
                                                   reduced_state_space_ego=True, reduced_state_space_alt=True,
                                                   use_skills_ego=True, use_skills_alt=True)

        initial_policy_path = os.path.join('data', 'prior_tree_policies', layout + '.tar')
        intent_model_path = os.path.join('data', 'intent_models', layout + '.pt')
        data_file = prior_iteration_data

        model = PPO("MlpPolicy", dummy_env)
        weights = torch.load(initial_policy_path)
        model.policy.load_state_dict(weights['ego_state_dict'])
        human_ppo_policy = model.policy
        human_policy = HumanModel(layout, human_ppo_policy)

        input_dim = dummy_env.observation_space.shape[0]
        output_dim = dummy_env.n_actions_alt

        robot_policy = RobotModel(layout=layout,
                                  idct_policy_filepath=initial_policy_path,
                                  human_policy=human_policy,
                                  intent_model_filepath=intent_model_path,
                                  input_dim=input_dim,
                                  output_dim=output_dim,
                                  randomize_initial_idct=rpo_random_initial_idct,
                                  only_optimize_leaves=rpo_rl_only_optimize_leaves)

        joint_environment = OvercookedJointRecorderEnvironment(behavioral_model=robot_policy.intent_model,
                                                               layout_name=layout, seed_num=0,
                                                               ego_idx=0,
                                                               failed_skill_rew=0,
                                                               reduced_state_space_ego=True,
                                                               reduced_state_space_alt=True,
                                                               use_skills_ego=True,
                                                               use_skills_alt=True,
                                                               use_true_intent_ego=True,
                                                               use_true_intent_alt=False,
                                                               double_cook_times=False)

        print('-----------------------')
        print('PLAYING INITIAL EPISODES')
        print('-----------------------')
        for i in tqdm(range(n_episode_samples)):
            reward = play_episode_together(joint_environment, human_policy, robot_policy, render=False)
            all_rewards_initial[random_seed].append(reward)

        data_file = 'data/iteration_0.tar'
        human_policy.translate_recent_data_to_labels(recent_data_loc=data_file)

        if hpo:
            print('-----------------------')
            print('FINETUNING HUMAN POLICY')
            print('-----------------------')
            human_policy.finetune_human_ppo_policy()

            print('-----------------------')
            print('PLAYING EPISODES AFTER FINETUNING HUMAN POLICY')
            print('-----------------------')

            for i in tqdm(range(n_episode_samples)):
                reward = play_episode_together(joint_environment, human_policy, robot_policy, render=False)
                all_rewards_finetuned_human_policy[random_seed].append(reward)

        current_policy, tree_info = sparse_ddt_to_decision_tree(robot_policy.robot_idct_policy,
                                                                robot_policy.env)
        intent_model = robot_policy.intent_model

        robot_policy.translate_recent_data_to_labels(recent_data_loc=data_file)
        if ipo:
            print('-----------------------')
            print('FINETUNING INTENT MODEL')
            print('-----------------------')

            robot_policy.finetune_intent_model()

            print('-----------------------')
            print('PLAYING EPISODES AFTER FINETUNING INTENT MODEL')
            print('-----------------------')

            for i in tqdm(range(n_episode_samples)):
                reward = play_episode_together(joint_environment, human_policy, robot_policy, render=False)
                all_rewards_finetuned_intent[random_seed].append(reward)

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
                                                    ga_depth=rpo_ga_depth,
                                                    ga_n_gens=rpo_ga_n_gens,
                                                    ga_n_pop=rpo_ga_n_pop,
                                                    ga_n_parents_mating=rpo_ga_n_parents_mating,
                                                    ga_crossover_prob=rpo_ga_crossover_prob,
                                                    ga_crossover_type=rpo_ga_crossover_type,
                                                    ga_mutation_prob=rpo_ga_mutation_prob,
                                                    ga_mutation_type=rpo_ga_mutation_type,
                                                    recent_data_file=rpo_ga_data_file)
            end_time = time.time()
            all_training_times_robot_policy.append(end_time - start_time)

            print('-----------------------')
            print('PLAYING EPISODES AFTER FINETUNING ROBOT MODEL')
            print('-----------------------')

            for i in tqdm(range(n_episode_samples)):
                reward = play_episode_together(joint_environment, human_policy, robot_policy, render=False)
                all_rewards_finetuned_robot_policy[random_seed].append(reward)

    def get_avg_and_std(all_rewards):
        arr = np.array(all_rewards).flatten()
        return np.mean(arr), np.std(arr)

    avg_rewards_initial, std_rewards_initial = get_avg_and_std(all_rewards_initial)
    avg_rewards_hpo, std_rewards_hpo = get_avg_and_std(all_rewards_finetuned_human_policy)
    avg_rewards_ipo, std_rewards_ipo = get_avg_and_std(all_rewards_finetuned_intent)
    avg_rewards_rpo, std_rewards_rpo = get_avg_and_std(all_rewards_finetuned_robot_policy)

    initial_performance_str = 'Average reward for initial policy: ' + \
                              str(round(avg_rewards_initial, 2)) + ' +/- ' + str(round(std_rewards_initial, 2)) + '\n'
    hpo_performance_str = 'Average reward after fine-tuning human policy: ' + \
                          str(round(avg_rewards_hpo, 2)) + ' +/- ' + str(round(std_rewards_hpo, 2)) + '\n'
    ipo_performance_str = 'Average reward after fine-tuning intent model: ' + \
                          str(round(avg_rewards_ipo, 2)) + ' +/- ' + str(round(std_rewards_ipo, 2)) + '\n'
    rpo_performance_str = 'Average reward after fine-tuning robot policy: ' + \
                          str(round(avg_rewards_rpo, 2)) + ' +/- ' + str(round(std_rewards_rpo, 2)) + '\n'

    rpo_avg_training_time = np.mean(all_training_times_robot_policy)
    rpo_training_str = 'Average training time for robot policy: ' + str(round(rpo_avg_training_time, 2)) + 's\n'

    print(initial_performance_str)

    if hpo:
        print(hpo_performance_str)
    if ipo:
        print(ipo_performance_str)
    if rpo:
        print(rpo_performance_str)
        print(rpo_training_str)

    # save the hyperparameters and results to a file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = 'hyperparameter_tuning_' + current_time + '.txt'
    file_path = os.path.join(data_folder, file_name)

    with open(file_path, 'w') as f:
        # let's first write out the results
        f.write(initial_performance_str)

        if hpo:
            f.write(hpo_performance_str)
        if ipo:
            f.write(ipo_performance_str)
        if rpo:
            f.write(rpo_performance_str)
            f.write(rpo_training_str)

        # then let's copy over the hyperparameters from our ini file
        # basically append the entire config file contents
        f.write('\nHyperparameters:\n')
        with open(args.config_file, 'r') as config_file:
            f.write(config_file.read())

    print('Results and hyperparameters saved to ', file_path)
