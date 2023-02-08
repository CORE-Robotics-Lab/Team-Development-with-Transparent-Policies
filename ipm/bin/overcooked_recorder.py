import argparse
import os

import pandas as pd
import pygame
import sys
sys.path.insert(0, '../../overcooked_ai/src/')
sys.path.insert(0, '../../overcooked_ai/src/overcooked_ai_py')
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from ipm.overcooked.overcooked import OvercookedPlayWithFixedPartner, OvercookedSelfPlayEnv
from ipm.models.bc_agent import get_human_bc_partner
from datetime import datetime

class OvercookedPlayWithAgent:
    def __init__(self, agent, traj_directory, layout_name='forced_coordination', n_episodes=1,
                 SCREEN_WIDTH=1920, SCREEN_HEIGHT=1080, screen=None,
                 ego_idx=0):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.layout_name = layout_name
        self.n_episodes = n_episodes
        self.agent = agent
        self.traj_directory = traj_directory

        self.ego_idx = ego_idx
        self.alt_idx = (self.ego_idx + 1) % 2

        self.n_timesteps = 100

        self.set_env()

        self.visualizer = StateVisualizer()
        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = screen

    def set_env(self):
        self.env = OvercookedPlayWithFixedPartner(partner=self.agent, layout_name=self.layout_name, seed_num=0,
                                                  ego_idx=self.ego_idx, n_timesteps=self.n_timesteps,
                                                  reduced_state_space_ego=True,
                                                  reduced_state_space_alt=True,
                                                  use_skills_ego=False,
                                                  use_skills_alt=False)

    def get_human_action(self, agent_idx):
        # force the user to make a move

        # KEY -> ACTION

        # LEFT -> LEFT
        # UP -> UP
        # DOWN -> DOWN
        # RIGHT -> RIGHT
        # SPACE -> INTERACT

        # 1 -> GET CLOSEST ONION
        # 2 -> GET CLOSEST TOMATO
        # 3 -> GET CLOSEST DISH
        # 4 -> GET CLOSEST SOUP
        # 5 -> SERVE SOUP
        # 6 -> BRING TO CLOSEST POT
        # 7 -> PLACE ON CLOSEST COUNTER

        # 0 -> STAND STILL

        self.visualize_state(self.env.state)

        agent_str = 'first' if agent_idx == 0 else 'second'
        color = 'BLUE' if agent_idx == 0 else 'GREEN'
        print(f'\nPlease enter the action to take for {agent_str} agent (hat color: {color})')

        command = None
        while command is None:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        command = 3  # LEFT -> LEFT
                    elif event.key == pygame.K_UP:
                        command = 0  # UP -> UP
                    elif event.key == pygame.K_DOWN:
                        command = 1  # DOWN -> DOWN
                    elif event.key == pygame.K_RIGHT:
                        command = 2  # RIGHT -> RIGHT
                    elif event.key == pygame.K_SPACE:
                        command = 5  # SPACE -> INTERACT
                    elif event.key == pygame.K_0:
                        command = 4  # 0 -> STAND STILL
                    # elif event.key == pygame.K_1:
                    #     command = 6  # 1 -> GET CLOSEST ONION
                    # elif event.key == pygame.K_2:
                    #     command = 7  # 2 -> GET CLOSEST TOMATO
                    # elif event.key == pygame.K_3:
                    #     command = 8  # 3 -> GET CLOSEST DISH
                    # elif event.key == pygame.K_4:
                    #     command = 9  # 4 -> GET CLOSEST SOUP
                    # elif event.key == pygame.K_5:
                    #     command = 10  # 5 -> SERVE SOUP
                    # elif event.key == pygame.K_6:
                    #     command = 11  # 6 -> BRING TO CLOSEST POT
                    # elif event.key == pygame.K_7:
                    #     command = 12  # 7 -> PLACE ON CLOSEST COUNTER
                    # elif event.key == pygame.K_ESCAPE:
                    #     command = 13  # ESC -> QUIT
                    else:
                        print("Please enter a valid action")
        return command

    def visualize_state(self, state):
        self.screen.fill((0, 0, 0))
        state_visualized_surf = self.visualizer.render_state(state=state, grid=self.env.base_env.mdp.terrain_mtx)
        self.screen.blit(pygame.transform.scale(state_visualized_surf, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)), (0, 0))
        pygame.display.flip()

    def play(self):

        # data format:
        # [state, action, episode, agent_idx]

        for i in range(self.n_episodes):
            done = False
            total_reward = 0
            obs = self.env.reset()
            clock = pygame.time.Clock()
            # self.visualize_state(self.env.state)
            clock.tick(60)

            self.observations = []
            self.raw_observations = []
            self.states = []
            self.actions = []
            self.episode_idxs = []
            self.agent_idxs = []
            self.current_episode_num = 0

            print('----------------------')
            print('\n\nBEGINNING EPISODE ', i)
            timestep = 0

            while not done:
                action = self.get_human_action(agent_idx=self.ego_idx)

                self.observations.append(obs)
                self.raw_observations.append(self.env.ego_raw_obs)
                self.states.append(self.env.state)
                self.actions.append(action)
                self.episode_idxs.append(self.current_episode_num)
                self.agent_idxs.append(self.ego_idx)

                obs, reward, done, info = self.env.step(action)

                total_reward += reward
                print(f'Timestep: {timestep} / {self.n_timesteps}, reward so far in ep {i}: {total_reward}.')
                timestep += 1
                clock.tick(60)

            df = pd.DataFrame(
                {'state': self.states, 'obs': self.observations, 'raw_obs': self.raw_observations,
                 'action': self.actions, 'episode': self.episode_idxs,
                 'agent_idx': self.agent_idxs})
            if len(df) > 0:
                timestamp = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
                output_path = os.path.join(self.traj_directory, f'{timestamp}.csv')
                df.to_csv(output_path, index=False)
                print('Trajectories saved to ', output_path)

            self.current_episode_num += 1


class OvercookedGameRecorder:
    def __init__(self, traj_directory, layout_name='forced_coordination_demonstrations', n_episodes=1,
                 SCREEN_WIDTH=1920, SCREEN_HEIGHT=1080, double_cook_times=True,
                 use_bc_teammate=False, alternate_agent_idx=False, screen=None):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.layout_name = layout_name
        self.traj_directory = traj_directory
        self.n_episodes = n_episodes
        self.use_bc_teammate = use_bc_teammate
        self.alternate_agent_idx = alternate_agent_idx

        self.n_actions = 13 # hardcoded for now
        self.actions = list(range(self.n_actions))

        num_primitives = 6
        num_skills = self.n_actions - num_primitives
        # skills
        self.skills_to_idx = {str(i):num_primitives + i for i in range(num_skills)}
        # 1 -> GET CLOSEST ONION
        # 2 -> GET CLOSEST TOMATO
        # 3 -> GET CLOSEST DISH
        # 4 -> GET CLOSEST SOUP
        # 5 -> SERVE SOUP
        # 6 -> BRING TO CLOSEST POT
        # 7 -> PLACE ON CLOSEST COUNTER

        self.ego_idx = 1
        self.alt_idx = (self.ego_idx + 1) % 2

        # other_agent = OtherAgentWrapper(possible_commands=self.actions,
        #                                 skills_to_idx=self.skills_to_idx,
        #                                 get_action_fn=self.get_human_action,
        #                                 alt_idx=self.alt_idx)

        # self.env = OvercookedPlayWithFixedPartner(partner=other_agent, layout_name=layout_name, ego_idx=self.ego_idx,
        #                                          reduced_state_space_ego=False,
        #                                          reduced_state_space_alt=False)

        self.n_timesteps = 400

        self.set_env()

        # assert self.n_actions == self.env.n_actions_ego
        # assert self.env.n_actions_ego == self.env.n_actions_alt

        self.visualizer = StateVisualizer()
        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = screen

    def set_env(self):
        if not self.use_bc_teammate:
            self.env = OvercookedSelfPlayEnv(layout_name=self.layout_name, ego_idx=self.ego_idx,
                                             reduced_state_space_ego=True,
                                             reduced_state_space_alt=True,
                                             n_timesteps=self.n_timesteps)
        else:
            word = 'demonstrations'
            if word in self.layout_name:
                layout_name = self.layout_name[:-len(word)-1]
            else:
                layout_name = self.layout_name
            self.bc_partner = get_human_bc_partner(self.traj_directory, layout_name, self.alt_idx)
            self.env = OvercookedPlayWithFixedPartner(partner=self.bc_partner, layout_name=layout_name, seed_num=0,
                                                      ego_idx=self.ego_idx, n_timesteps=self.n_timesteps,
                                                     reduced_state_space_ego=True,
                                                     reduced_state_space_alt=True,
                                                     use_skills_ego=False,
                                                     use_skills_alt=False)

    def get_human_action(self, agent_idx):
        # force the user to make a move

        # KEY -> ACTION

        # LEFT -> LEFT
        # UP -> UP
        # DOWN -> DOWN
        # RIGHT -> RIGHT
        # SPACE -> INTERACT

        # 1 -> GET CLOSEST ONION
        # 2 -> GET CLOSEST TOMATO
        # 3 -> GET CLOSEST DISH
        # 4 -> GET CLOSEST SOUP
        # 5 -> SERVE SOUP
        # 6 -> BRING TO CLOSEST POT
        # 7 -> PLACE ON CLOSEST COUNTER

        # 0 -> STAND STILL

        self.visualize_state(self.env.state)

        agent_str = 'first' if agent_idx == 0 else 'second'
        color = 'BLUE' if agent_idx == 0 else 'GREEN'
        print(f'\nPlease enter the action to take for {agent_str} agent (hat color: {color})')

        command = None
        while command is None:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        command = 3 # LEFT -> LEFT
                    elif event.key == pygame.K_UP:
                        command = 0 # UP -> UP
                    elif event.key == pygame.K_DOWN:
                        command = 1 # DOWN -> DOWN
                    elif event.key == pygame.K_RIGHT:
                        command = 2 # RIGHT -> RIGHT
                    elif event.key == pygame.K_SPACE:
                        command = 5 # SPACE -> INTERACT
                    elif event.key == pygame.K_0:
                        command = 4 # 0 -> STAND STILL
                    elif event.key == pygame.K_1:
                        command = 6 # 1 -> GET CLOSEST ONION
                    elif event.key == pygame.K_2:
                        command = 7 # 2 -> GET CLOSEST TOMATO
                    elif event.key == pygame.K_3:
                        command = 8 # 3 -> GET CLOSEST DISH
                    elif event.key == pygame.K_4:
                        command = 9 # 4 -> GET CLOSEST SOUP
                    elif event.key == pygame.K_5:
                        command = 10 # 5 -> SERVE SOUP
                    elif event.key == pygame.K_6:
                        command = 11 # 6 -> BRING TO CLOSEST POT
                    elif event.key == pygame.K_7:
                        command = 12 # 7 -> PLACE ON CLOSEST COUNTER
                    elif event.key == pygame.K_ESCAPE:
                        command = 13 # ESC -> QUIT
                    else:
                        print("Please enter a valid action")
        return command


    def visualize_state(self, state):
        self.screen.fill((0, 0, 0))
        state_visualized_surf = self.visualizer.render_state(state=state, grid=self.env.base_env.mdp.terrain_mtx)
        self.screen.blit(pygame.transform.scale(state_visualized_surf, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)), (0, 0))
        pygame.display.flip()

    def record_trajectories(self):
        debug = False

        # data format:
        # [state, action, episode, agent_idx]

        for i in range(self.n_episodes):
            done = False
            total_reward = 0
            obs = self.env.reset()
            clock = pygame.time.Clock()
            # self.visualize_state(self.env.state)
            clock.tick(60)

            self.observations = []
            self.raw_observations = []
            self.states = []
            self.actions = []
            self.episode_idxs = []
            self.agent_idxs = []
            self.current_episode_num = 0

            print('----------------------')
            print('\n\nBEGINNING EPISODE ', i)
            timestep = 0

            while not done:
                if debug:
                    if self.ego_idx == 0:
                        action = self.get_human_action(agent_idx=self.ego_idx)
                    else:
                        action = 4
                else:
                    action = self.get_human_action(agent_idx=self.ego_idx)

                if action == 13:
                    print('Ending game early...')
                    done = True
                else:
                    self.observations.append(obs)
                    self.raw_observations.append(self.env.ego_raw_obs)
                    self.states.append(self.env.state)
                    self.actions.append(action)
                    self.episode_idxs.append(self.current_episode_num)
                    self.agent_idxs.append(self.ego_idx)

                    obs, reward, done, info = self.env.step(action)
                    if not self.use_bc_teammate:
                        self.ego_idx = (self.ego_idx + 1) % 2
                        self.alt_idx = (self.alt_idx + 1) % 2
                    total_reward += reward
                    print(f'Timestep: {timestep} / {self.n_timesteps}, reward so far in ep {i}: {total_reward}.')
                    timestep += 1
                    clock.tick(60)

            df = pd.DataFrame(
                {'state': self.states, 'obs': self.observations, 'raw_obs': self.raw_observations,
                 'action': self.actions, 'episode': self.episode_idxs,
                 'agent_idx': self.agent_idxs})
            if len(df) > 0:
                timestamp = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
                output_path = os.path.join(self.traj_directory, f'{timestamp}.csv')
                df.to_csv(output_path, index=False)
                print('Trajectories saved to ', output_path)

            self.current_episode_num += 1

            if self.use_bc_teammate and self.alternate_agent_idx:
                self.ego_idx = (self.ego_idx + 1) % 2
                self.alt_idx = (self.alt_idx + 1) % 2
                print('Retraining bc agent (if using one)...')
                self.set_env()
            else:
                self.ego_idx = 0 # or default ego value
                self.alt_idx = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Records trajectories of human playing overcooked')
    parser.add_argument('--traj_directory', help='the output directory to save the data to', type=str)
    parser.add_argument('--layout_name', help='the layout to use', type=str, default='ipm2')
    parser.add_argument('--use_bc_teammate', help='whether to use a bc teammate', type=bool, default=False)
    parser.add_argument('--alternate_agent_idx', help='whether to alternate the agent index', type=bool, default=False)
    parser.add_argument('--n_episodes', help='the number of episodes to record', type=int, default=1)
    args = parser.parse_args()

    demo = OvercookedGameRecorder(traj_directory=args.traj_directory, layout_name=args.layout_name, n_episodes=args.n_episodes,
                                  use_bc_teammate=args.use_bc_teammate, alternate_agent_idx=args.alternate_agent_idx)
    demo.record_trajectories()
