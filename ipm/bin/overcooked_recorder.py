import argparse
import os
import pickle

import pandas as pd
import pygame
import sys
sys.path.insert(0, '../../overcooked_ai/src/')
sys.path.insert(0, '../../overcooked_ai/src/overcooked_ai_py')
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from ipm.overcooked.overcooked_envs import OvercookedPlayWithFixedPartner, OvercookedSelfPlayEnv
from ipm.overcooked.overcooked_envs import OvercookedJointRecorderEnvironment
from ipm.models.bc_agent import get_pretrained_teammate_finetuned_with_bc
from datetime import datetime

class OvercookedPlayWithAgent:
    def __init__(self, agent, behavioral_model, traj_directory, layout_name='forced_coordination', n_episodes=1,
                 SCREEN_WIDTH=1920, SCREEN_HEIGHT=1080, screen=None,
                 ego_idx=0):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.layout_name = layout_name
        self.n_episodes = n_episodes
        self.agent = agent
        self.behavioral_model = behavioral_model
        self.traj_directory = traj_directory

        self.ego_idx = ego_idx
        self.alt_idx = (self.ego_idx + 1) % 2

        self.n_timesteps = 200

        self.set_env()
        self.env.base_env.mdp.behavioral_model = self.behavioral_model
        self.visualizer = StateVisualizer()
        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = screen

    def set_env(self):
        self.env = OvercookedPlayWithFixedPartner(partner=self.agent, behavioral_model=self.behavioral_model,
                                                  layout_name=self.layout_name, seed_num=0,
                                                  ego_idx=self.ego_idx, n_timesteps=self.n_timesteps, failed_skill_rew=0,
                                                  reduced_state_space_ego=True,
                                                  reduced_state_space_alt=True,
                                                  use_skills_ego=False,
                                                  use_skills_alt=True)

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
                    elif event.key == pygame.K_w: # press w
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
        if self.layout_name == 'two_rooms_narrow':
            self.screen.blit(text, (self.SCREEN_WIDTH - 400, 0))
            self.screen.blit(reward_text, (0, 0))
        else:
            self.screen.blit(text, (0, 0))
            self.screen.blit(reward_text, (self.SCREEN_WIDTH - 400, 0))
        pygame.display.flip()

    def play(self):

        # data format:
        # [state, action, episode, agent_idx]

        done = False
        self.total_reward = 0.0
        obs = self.env.reset()
        clock = pygame.time.Clock()
        # self.visualize_state(self.env.state)
        clock.tick(60)

        self.observations = []
        self.raw_observations = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.episode_idxs = []
        self.agent_idxs = []
        self.current_episode_num = 0

        print('----------------------')
        print('\n\nBEGINNING EPISODE ', 0)
        self.timestep = 0

        while not done:
            action = self.get_human_action(agent_idx=self.ego_idx)
            if action == -1:
                print('User stopped the game.')
                break

            self.observations.append(obs)
            self.raw_observations.append(self.env.ego_raw_obs)
            self.states.append(self.env.state)
            self.actions.append(action)
            self.episode_idxs.append(self.current_episode_num)
            self.agent_idxs.append(self.ego_idx)

            print(action)
            obs, reward, done, info = self.env.step(action)
            reward = self.env.joint_reward

            self.rewards.append(reward)

            self.total_reward += reward
            print(f'Timestep: {self.timestep} / {self.n_timesteps}, reward so far in ep {0}: {self.total_reward}.')
            self.timestep += 1
            clock.tick(60)

        df = pd.DataFrame(
            {'obs': self.observations, 'raw_obs': self.raw_observations,
             'action': self.actions, 'reward': self.rewards, 'episode': self.episode_idxs,
             'agent_idx': self.agent_idxs})
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

        self.current_episode_num += 1
        return self.total_reward


class OvercookedRecorder:
    def __init__(self, traj_directory, layout_name='forced_coordination',
                 SCREEN_WIDTH=1280, SCREEN_HEIGHT=720):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.layout_name = layout_name
        self.traj_directory = traj_directory

        num_primitives = 6

        assert 'demonstrations' not in layout_name, 'Not backwards compatible'

        self.n_timesteps = 100
        self.set_env()

        self.visualizer = StateVisualizer()
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

    def set_env(self):
        self.env = OvercookedJointRecorderEnvironment(layout_name=self.layout_name,
                                                      n_timesteps=self.n_timesteps)

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
        p0_obs, p1_obs = self.env.reset()
        clock = pygame.time.Clock()
        # self.visualize_state(self.env.state)
        clock.tick(60)

        self.observations = []
        self.states = []
        self.actions = []
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

            self.observations.append(p0_obs)
            self.actions.append(p0_action)
            self.rewards.append(p0_rew)
            self.joint_rewards.append(p0_rew + p1_rew)
            self.agent_idxs.append(0)

            self.observations.append(p1_obs)
            self.actions.append(p1_action)
            self.rewards.append(p1_rew)
            self.joint_rewards.append(p0_rew + p1_rew)
            self.agent_idxs.append(1)

            (p0_obs, p1_obs), (p0_rew, p1_rew), done, info = self.env.step(joint_action)

            total_reward += p0_rew + p1_rew
            print(f'Timestep: {timestep} / {self.n_timesteps}, reward so far in ep {0}: {total_reward}.')
            timestep += 1
            clock.tick(60)

        df = pd.DataFrame(
            {'obs': self.observations,
             'action': self.actions,
             'reward': self.rewards,
             'joint_reward': self.joint_rewards,
             'agent_idx': self.agent_idxs})

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
    parser.add_argument('--layout_name', help='the layout to use', type=str, default='forced_coordination')
    parser.add_argument('--use_bc_teammate', help='whether to use a bc teammate', type=bool, default=False)
    parser.add_argument('--alternate_agent_idx', help='whether to alternate the agent index', type=bool, default=False)
    parser.add_argument('--n_episodes', help='the number of episodes to record', type=int, default=1)
    args = parser.parse_args()

    demo = OvercookedRecorder(traj_directory=args.traj_directory, layout_name=args.layout_name)
    demo.record_trajectories()
