import argparse
import os
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
    parser.add_argument('--hpo', help='Include human policy optimization', type=bool, default=False)
    parser.add_argument('--hpo_lr', help='learning rate for human policy optimization', type=float, default=5e-4)
    parser.add_argument('--hpo_n_epochs', help='number of epochs for human policy optimization', type=int, default=50)

    parser.add_argument('--ipo', help='Include intent policy optimization', type=bool, default=False)
    parser.add_argument('--ipo_lr', help='learning rate for intent policy optimization', type=float, default=5e-3)
    parser.add_argument('--ipo_n_epochs', help='number of epochs for intent policy optimization', type=int, default=50)


    parser.add_argument('--rpo', help='Include robot policy optimization', type=bool, default=False)

    # in order to use BOTH, just use both flags
    parser.add_argument('--rpo_ga', help='Include ga in rpo', type=bool, default=False)
    parser.add_argument('--rpo_rl', help='Include rl in ppo', type=bool, default=False)

    parser.add_argument('--rpo_ga_depth', help='number of individuals for robot policy optimization', type=int, default=3)
    parser.add_argument('--rpo_ga_n_gens', help='number of generations for robot policy optimization', type=int, default=100)

    parser.add_argument('--rpo_rl_lr', help='learning rate for robot policy optimization', type=float, default=0.0003)
    parser.add_argument('--rpo_rl_n_steps', help='number of steps for robot policy optimization', type=int, default=70000)

    args = parser.parse_args()

    data_folder = os.path.join('data',
                               'tuning')

    layout = 'forced_coordination'
    folder = os.path.join(data_folder, layout)
    if not os.path.exists(folder):
        os.makedirs(folder)

    pygame.init()
    settings = SettingsWrapper()

    dummy_env = OvercookedPlayWithFixedPartner(partner=StayAgent(), layout_name=layout,
                                               behavioral_model='dummy',
                                               reduced_state_space_ego=True, reduced_state_space_alt=True,
                                               use_skills_ego=True, use_skills_alt=True)

    initial_policy_path = os.path.join('data', 'prior_tree_policies', layout + '.tar')
    intent_model_path = os.path.join('data', 'intent_models', layout + '.pt')

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
                              output_dim=output_dim)

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

    n_episode_samples = 30

    all_rewards_initial = []
    for i in tqdm(range(n_episode_samples)):
        reward = play_episode_together(joint_environment, human_policy, robot_policy, render=False)
        all_rewards_initial.append(reward)

    data_file = 'data/iteration_0.tar'
    human_policy.translate_recent_data_to_labels(recent_data_loc=data_file)
    human_policy.finetune_human_ppo_policy()

    all_rewards_finetuned_human = []
    for i in tqdm(range(n_episode_samples)):
        reward = play_episode_together(joint_environment, human_policy, robot_policy, render=False)
        all_rewards_finetuned_human.append(reward)

    current_policy, tree_info = sparse_ddt_to_decision_tree(robot_policy.robot_idct_policy,
                                                            robot_policy.env)
    intent_model = robot_policy.intent_model

    data_file = 'data/iteration_0.tar'
    robot_policy.translate_recent_data_to_labels(recent_data_loc=data_file)
    robot_policy.finetune_intent_model()

    # all_rewards_finetuned_intent = []
    # for i in range(n_samples):
    #     reward = play_episode_together(joint_environment, human_policy, robot_policy, render=False)
    #     all_rewards_finetuned_intent.append(reward)
    #
    robot_policy.finetune_robot_idct_policy()

    all_rewards_finetuned_ga_rl = []
    for i in range(n_episode_samples):
        reward = play_episode_together(joint_environment, human_policy, robot_policy, render=False)
        all_rewards_finetuned_ga_rl.append(reward)

    print('Average reward for initial policy: ', round(np.mean(all_rewards_initial), 2))
    print('Average reward for fine-tuned human policy: ', round(np.mean(all_rewards_finetuned_human), 2))
    # print('Average reward for fine-tuned intent model: ', np.mean(all_rewards_finetuned_intent))
    print('Average reward for fine-tuned GA: ', round(np.mean(all_rewards_finetuned_ga_rl), 2))

    # save the hyperparameters and results to a file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = 'hyperparameter_tuning_' + current_time + '.txt'
    file_path = os.path.join(data_folder, file_name)
    with open(file_path, 'w') as f:
        # write the hyperparameters
        f.write('Hyperparameters:\n')
        f.write('layout: ' + layout + '\n')
        f.write('n_episode_samples: ' + str(n_episode_samples) + '\n')
        f.write('include robot policy optimization: ' + str(args.rpo) + '\n')
        f.write('include intent model optimization: ' + str(args.imo) + '\n')
        f.write('include human policy optimization: ' + str(args.hpo) + '\n')

        f.write('rpo_rl_lr: ' + str(args.rpo_rl_lr) + '\n')
        f.write('rpo_rl_n_steps: ' + str(args.rpo_rl_n_steps) + '\n')
        f.write('rpo_rl_n_epochs: ' + str(args.rpo_rl_n_epochs) + '\n')
        f.write('rpo_rl_batch_size: ' + str(args.rpo_rl_batch_size) + '\n')
        f.write('rpo_rl_gamma: ' + str(args.rpo_rl_gamma) + '\n')

        f.write('Average reward for initial policy: ' + str(round(np.mean(all_rewards_initial), 2)) + '\n')
        f.write(
            'Average reward for fine-tuned human policy: ' + str(round(np.mean(all_rewards_finetuned_human), 2)) + '\n')
        # f.write('Average reward for fine-tuned intent model: ' + str(np.mean(all_rewards_finetuned_intent)) + '\n')
        f.write('Average reward for fine-tuned GA: ' + str(round(np.mean(all_rewards_finetuned_ga_rl), 2)) + '\n')
