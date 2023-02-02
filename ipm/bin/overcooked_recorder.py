import argparse
import pandas as pd
import pygame
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from ipm.overcooked.overcooked import OvercookedPlayWithFixedPartner, OvercookedSelfPlayEnv


class OvercookedGameRecorder:
    def __init__(self, traj_filepath, layout_name='forced_coordination_tomato', n_episodes=1, SCREEN_WIDTH=1600, SCREEN_HEIGHT=900):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.layout_name = layout_name
        self.traj_filepath = traj_filepath
        self.n_episodes = n_episodes

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

        self.ego_idx = 0
        self.alt_idx = (self.ego_idx + 1) % 2

        # other_agent = OtherAgentWrapper(possible_commands=self.actions,
        #                                 skills_to_idx=self.skills_to_idx,
        #                                 get_action_fn=self.get_human_action,
        #                                 alt_idx=self.alt_idx)

        # self.env = OvercookedPlayWithFixedPartner(partner=other_agent, layout_name=layout_name, ego_idx=self.ego_idx,
        #                                          reduced_state_space_ego=False,
        #                                          reduced_state_space_alt=False)

        self.n_timesteps = 10

        self.env = OvercookedSelfPlayEnv(layout_name=layout_name, ego_idx=self.ego_idx,
                                         reduced_state_space_ego=False,
                                         reduced_state_space_alt=False,
                                         n_timesteps=self.n_timesteps)

        assert self.n_actions == self.env.n_actions_ego
        assert self.env.n_actions_ego == self.env.n_actions_alt

        self.observations = []
        self.actions = []
        self.episode_idxs = []
        self.agent_idxs = []

        self.current_episode_num = 0

        self.visualizer = StateVisualizer()
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

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
            agent_idx = 0

            print('----------------------')
            print('\n\nBEGINNING EPISODE ', i)
            timestep = 0

            while not done:
                if debug:
                    if agent_idx == 0:
                        action = self.get_human_action(agent_idx=agent_idx)
                    else:
                        action = 4
                else:
                    action = self.get_human_action(agent_idx=agent_idx)

                self.observations.append(obs)
                self.actions.append(action)
                self.episode_idxs.append(self.current_episode_num)
                self.agent_idxs.append(agent_idx)

                obs, reward, done, info = self.env.step(action)
                agent_idx = (agent_idx + 1) % 2
                total_reward += reward
                print(f'Timestep: {timestep} / {self.n_timesteps}, reward so far in ep {i}: {total_reward}.')
                timestep += 1
                clock.tick(60)

            self.current_episode_num += 1

        df = pd.DataFrame({'obs': self.observations, 'action': self.actions, 'episode': self.episode_idxs, 'agent_idx': self.agent_idxs})
        df.to_csv(self.traj_filepath, index=False)
        print('Trajectories saved to ', self.traj_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Records trajectories of human playing overcooked')
    parser.add_argument('--traj_filepath', help='the output file to save the data to', type=str)
    parser.add_argument('--layout_name', help='the layout to use', type=str, default='forced_coordination_tomato')
    parser.add_argument('--n_episodes', help='the number of episodes to record', type=int, default=1)
    args = parser.parse_args()

    demo = OvercookedGameRecorder(traj_filepath=args.traj_filepath, layout_name=args.layout_name, n_episodes=args.n_episodes)
    demo.record_trajectories()
