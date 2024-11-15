import argparse
import os
import pickle
import sys

import pandas as pd
import pygame
import torch

sys.path.insert(0, '../../overcooked_ai/src/')
sys.path.insert(0, '../../overcooked_ai/src/overcooked_ai_py')
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from ipm.overcooked.overcooked_envs import OvercookedJointRecorderEnvironment
from datetime import datetime


class OvercookedPlayWithAgent:
    def __init__(self, agent, base_save_dir, layout_name='forced_coordination', n_episodes=1,
                 SCREEN_WIDTH=1920, SCREEN_HEIGHT=1080, screen=None, ego_idx=0, current_iteration=0):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.layout_name = layout_name
        self.n_episodes = n_episodes
        self.agent = agent
        self.save_dir = os.path.join(base_save_dir, layout_name)
        self.current_iteration = current_iteration
        self.save_file = os.path.join(self.save_dir, 'iteration_{}.tar'.format(self.current_iteration))

        self.ego_idx = ego_idx
        self.alt_idx = (self.ego_idx + 1) % 2

        self.n_timesteps = 200

        self.set_env()
        self.visualizer = StateVisualizer()
        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = screen

    def set_env(self):
        self.env = OvercookedJointRecorderEnvironment(layout_name=self.layout_name, seed_num=0,
                                                      ego_idx=0, n_timesteps=self.n_timesteps,
                                                      failed_skill_rew=0,
                                                      reduced_state_space_ego=True,
                                                      reduced_state_space_alt=True,
                                                      use_skills_ego=False,
                                                      use_skills_alt=True,
                                                      use_true_intent_ego=False,
                                                      use_true_intent_alt=False,
                                                      double_cook_times=False)

    def get_human_action(self, agent_idx):
        # force the user to make a move
        self.visualize_state(self.env.state)

        agent_str = 'first' if agent_idx == 0 else 'second'
        color = 'BLUE' if agent_idx == 0 else 'GREEN'
        print(f'\nPlease enter the action to take for {agent_str} agent (hat color: {color})')

        command = None
        while command is None:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        command = 0  # UP -> UP
                    elif event.key == pygame.K_DOWN:
                        command = 1  # DOWN -> DOWN
                    elif event.key == pygame.K_RIGHT:
                        command = 2  # RIGHT -> RIGHT
                    elif event.key == pygame.K_LEFT:
                        command = 3  # LEFT -> LEFT
                    elif event.key == pygame.K_w:  # press w
                        command = 4  # 0 -> STAND STILL
                    elif event.key == pygame.K_SPACE:
                        command = 5  # SPACE -> INTERACT
                    elif event.key == pygame.K_s:
                        command = -1  # s -> STOP GAME
                    else:
                        print("Please enter a valid action")
        return command

    def visualize_state(self, state):
        self.screen.fill((0, 0, 0))
        state_visualized_surf = self.visualizer.render_state(state=state, grid=self.env.base_env.mdp.terrain_mtx)
        self.screen.blit(pygame.transform.scale(state_visualized_surf, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)), (0, 0))

        # let's also print the current timer on the top left
        font = pygame.font.SysFont('Arial', 36)
        # make it bold
        font.set_bold(True)
        # for 2 domains, put timer in top left
        # for two_rooms_narrow domain, put timer in top right
        text = font.render('Timesteps Left: {}'.format(self.n_timesteps - self.timestep), True, (0, 0, 0))
        reward_text = font.render('Reward: {}'.format(self.total_reward), True, (0, 0, 0))
        iteration_text = font.render('Iteration: {}'.format(self.across_iteration + 1), True, (0, 0, 0))
        if self.layout_name == 'two_rooms_narrow':
            self.screen.blit(text, (self.SCREEN_WIDTH - 400, 0))
            self.screen.blit(reward_text, (0, 0))
            self.screen.blit(iteration_text, (0, self.SCREEN_HEIGHT - 50))
        else:
            self.screen.blit(text, (0, 0))
            self.screen.blit(reward_text, (self.SCREEN_WIDTH - 400, 0))
            self.screen.blit(iteration_text, (0, self.SCREEN_HEIGHT - 50))
        pygame.display.flip()

    def play(self):

        # data format:
        # [state, action, episode, agent_idx]

        done = False
        self.total_reward = 0.0
        p0_obs, p1_obs = self.env.reset(use_reduced=True)
        clock = pygame.time.Clock()
        # self.visualize_state(self.env.state)
        clock.tick(60)

        self.human_observations = []
        self.AI_observations = []
        self.states = []
        self.human_actions = []
        self.AI_actions = []
        self.rewards = []
        self.joint_rewards = []
        self.agent_idxs = []
        p0_rew, p1_rew = 0, 0

        print('----------------------')
        print('\n\nBEGINNING EPISODE ', 0)
        self.timestep = 0

        while not done:
            p0_action = self.get_human_action(agent_idx=0)
            if p0_action == -1:
                print('User stopped the game.')
                break
            p1_action, _ = self.agent.predict(p1_obs)

            idx_to_skill_strings = [
                ['stand_still'],
                ['get_onion_from_dispenser'], ['pickup_onion_from_counter'],
                ['get_dish_from_dispenser'], ['pickup_dish_from_counter'],
                ['get_soup_from_pot'], ['pickup_soup_from_counter'],
                ['serve_at_dispensary'],
                ['bring_to_closest_pot'], ['place_on_closest_counter'],
            ['get_tomato_from_dispenser'], ['pickup_tomato_from_counter']]
            print('robot actions', idx_to_skill_strings[p1_action])
            joint_action = (p0_action, p1_action)

            self.states.append(self.env.state)

            self.human_observations.append(p0_obs)
            self.human_actions.append(p0_action)
            self.rewards.append(p0_rew)
            self.joint_rewards.append(p0_rew + p1_rew)
            self.agent_idxs.append(0)

            self.AI_observations.append(p1_obs)
            self.AI_actions.append(p1_action)
            self.rewards.append(p1_rew)
            self.joint_rewards.append(p0_rew + p1_rew)
            self.agent_idxs.append(1)

            (p0_obs, p1_obs), (p0_rew, p1_rew), done, info = self.env.step(joint_action, use_reduced=True)

            self.total_reward += p0_rew + p1_rew
            print(f'Timestep: {self.timestep} / {self.n_timesteps}, reward so far in ep {0}: {self.total_reward}.')
            self.timestep += 1
            clock.tick(60)

        data_dict = {'human_obs': self.human_observations,
                     'human_action': self.human_actions,
                     'AI_obs': self.AI_observations,
                     'AI_action': self.AI_actions,
                     'reward': self.rewards,
                     'joint_reward': self.joint_rewards,
                     'agent_idx': self.agent_idxs,
                     'states': self.states}

        torch.save(data_dict, self.save_file)
        return self.total_reward


class OvercookedRecorder:
    def __init__(self, traj_directory, layout_name='coordination_ring',
                 SCREEN_WIDTH=1280, SCREEN_HEIGHT=720):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.layout_name = layout_name
        self.traj_directory = traj_directory

        num_primitives = 6

        assert 'demonstrations' not in layout_name, 'Not backwards compatible'

        self.n_timesteps = 199
        self.set_env()

        self.visualizer = StateVisualizer()
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

    def set_env(self):
        self.env = OvercookedJointRecorderEnvironment(layout_name=self.layout_name, seed_num=0,
                                                      ego_idx=0, n_timesteps=self.n_timesteps,
                                                      failed_skill_rew=0,
                                                      reduced_state_space_ego=False,
                                                      reduced_state_space_alt=False,
                                                      use_skills_ego=False,
                                                      use_skills_alt=False,
                                                      double_cook_times=False)

    def get_human_action(self, agent_idx):
        # force the user to make a move

        # KEY -> ACTION

        # LEFT -> LEFT
        # UP -> UP
        # DOWN -> DOWN
        # RIGHT -> RIGHT
        # SPACE -> INTERACT

        # 0 -> STAND STILL

        self.visualize_state(self.env.state)

        agent_str = 'first' if agent_idx == 0 else 'second'
        color = 'BLUE' if agent_idx == 0 else 'GREEN'
        print(f'\nPlease enter the action to take for {agent_str} agent (hat color: {color})')

        onion_only_layouts = ['forced_coordination', 'two_rooms', 'tutorial']

        command = None
        while command is None:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        command = 0  # UP -> UP
                    elif event.key == pygame.K_DOWN:
                        command = 1  # DOWN -> DOWN
                    elif event.key == pygame.K_RIGHT:
                        command = 2  # RIGHT -> RIGHT
                    elif event.key == pygame.K_LEFT:
                        command = 3  # LEFT -> LEFT
                    elif event.key == pygame.K_w:  # press s
                        command = 4  # 0 -> WAIT
                    elif event.key == pygame.K_SPACE:
                        command = 5  # SPACE -> INTERACT
                    elif event.key == pygame.K_s:
                        command = -1  # s -> STOP GAME
                    else:
                        print("Please enter a valid action")
        return command

    def visualize_state(self, state):
        self.screen.fill((0, 0, 0))
        state_visualized_surf = self.visualizer.render_state(state=state, grid=self.env.base_env.mdp.terrain_mtx)
        self.screen.blit(pygame.transform.scale(state_visualized_surf, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)), (0, 0))
        pygame.display.flip()

    def record_trajectories(self):
        # data format:
        # [state, action, episode, agent_idx]

        done = False
        total_reward = 0
        p0_obs, p1_obs = self.env.reset(use_reduced=True)
        clock = pygame.time.Clock()
        # self.visualize_state(self.env.state)
        clock.tick(60)

        self.human_observations = []
        self.AI_observations = []
        self.states = []
        self.human_actions = []
        self.AI_actions = []
        self.rewards = []
        self.joint_rewards = []
        self.agent_idxs = []
        p0_rew, p1_rew = 0, 0

        timestep = 0

        while not done:
            p0_action = self.get_human_action(agent_idx=0)
            p1_action = self.get_human_action(agent_idx=1)
            joint_action = (p0_action, p1_action)

            self.states.append(self.env.state)

            self.human_observations.append(p0_obs)
            self.human_actions.append(p0_action)
            self.rewards.append(p0_rew)
            self.joint_rewards.append(p0_rew + p1_rew)
            self.agent_idxs.append(0)

            self.AI_observations.append(p1_obs)
            self.AI_actions.append(p1_action)
            self.rewards.append(p1_rew)
            self.joint_rewards.append(p0_rew + p1_rew)
            self.agent_idxs.append(1)

            (p0_obs, p1_obs), (p0_rew, p1_rew), done, info = self.env.step(joint_action, use_reduced=True)

            total_reward += p0_rew + p1_rew
            print(f'Timestep: {timestep} / {self.n_timesteps}, reward so far in ep {0}: {total_reward}.')
            timestep += 1
            clock.tick(60)

        # why is w saved as action 4?

        data_dict = {'human_obs': self.human_observations,
                     'human_action': self.human_actions,
                     'AI_obs': self.AI_observations,
                     'AI_action': self.AI_actions,
                     'reward': self.rewards,
                     'joint_reward': self.joint_rewards,
                     'agent_idx': self.agent_idxs,
                     'states': self.states}

        df = pd.DataFrame({'obs': self.observations,
                           'action': self.actions,
                           'reward': self.rewards,
                           'joint_reward': self.joint_rewards,
                           'agent_idx': self.agent_idxs})
        torch.save(data_dict,
                   '/home/rohanpaleja/PycharmProjects/PantheonRL/overcookedgym/rohan_models/recorder_data.tar')

        if len(df) > 0:
            timestamp = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
            filename = self.layout_name + '_' + timestamp + '.csv'
            output_path = os.path.join(self.traj_directory, filename)
            df.to_csv(output_path, index=False)
            print('Trajectories saved to ', output_path)
            # save states array as pickle
            with open(output_path.replace('.csv', '_states.pkl'), 'wb') as f:
                pickle.dump(self.states, f)
            print('States saved to ', output_path.replace('.csv', '_states.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Records trajectories of human playing overcooked')
    parser.add_argument('--traj_directory', help='the output directory to save the data to', type=str)
    parser.add_argument('--layout_name', help='the layout to use', type=str, default='coordination_ring')
    parser.add_argument('--use_bc_teammate', help='whether to use a bc teammate', type=bool, default=False)
    parser.add_argument('--alternate_agent_idx', help='whether to alternate the agent index', type=bool, default=False)
    parser.add_argument('--n_episodes', help='the number of episodes to record', type=int, default=1)
    args = parser.parse_args()

    demo = OvercookedRecorder(traj_directory=args.traj_directory, layout_name=args.layout_name)
    demo.record_trajectories()
