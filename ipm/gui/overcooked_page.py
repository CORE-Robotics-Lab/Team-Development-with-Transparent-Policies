import os

from stable_baselines3 import PPO

import pygame
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai.src.overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
import pygame
from stable_baselines3 import PPO
from ipm.overcooked.skills import Skills

from overcooked_ai.src.overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from ipm.overcooked.overcooked_envs import OvercookedPlayWithFixedPartner


class OvercookedGameDemo:
    def __init__(self, screen=None, other_agent=None,
                 layout_name='forced_coordination_tomato', horizon_length=15,
                 SCREEN_WIDTH=1920, SCREEN_HEIGHT=1080,
                 use_custom_visualizer=False):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.layout_name = layout_name

        default_agent_filepath = os.path.join('data', layout_name, 'self_play_training_models', 'seed_0',
                                              'final_model.zip')
        self.other_agent = PPO.load(default_agent_filepath)
        (observation_len,) = self.other_agent.observation_space.shape
        if observation_len < 96:
            reduce_teammate_state_space = True
        else:
            reduce_teammate_state_space = False

        self.env = OvercookedPlayWithFixedPartner(partner=self.other_agent, layout_name=layout_name, ego_idx=0,
                                                 reduced_state_space_ego=True,
                                                 reduced_state_space_alt=reduce_teammate_state_space)

        if use_custom_visualizer is False:
            self.visualizer = StateVisualizer()

        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = screen

        # if other_agent is not None:
        #     self.other_agent = other_agent
        #     self.env.partner = self.other_agent

    def get_human_action(self, time_ticks=False):
        # force the user to make a move
        # if we want to run the game continuously:
        # we would need to use a timer to keep track of seconds elapsed

        if time_ticks:
            clock = pygame.time.Clock()
            ms = 0
            command = 4 # by default, stay
            while ms < 500:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            command = 3
                        elif event.key == pygame.K_UP:
                            command = 0
                        elif event.key == pygame.K_DOWN:
                            command = 1
                        elif event.key == pygame.K_RIGHT:
                            command = 2
                        elif event.key == pygame.K_SPACE:
                            command = 5
                    return command
                ms += clock.tick(60)
                self.visualize_state(self.env.state)
            command = self.action_sequence[self.current_action_idx]
            return command
        else:
            command = None
            while command is None:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            command = 3
                        elif event.key == pygame.K_UP:
                            command = 0
                        elif event.key == pygame.K_DOWN:
                            command = 1
                        elif event.key == pygame.K_RIGHT:
                            command = 2
                        elif event.key == pygame.K_SPACE:
                            command = 5
                        return command
            return command

    def visualize_state(self, state):
        self.screen.fill((0, 0, 0))
        state_visualized_surf = self.visualizer.render_state(state=state, grid=self.env.base_env.mdp.terrain_mtx)
        self.screen.blit(pygame.transform.scale(state_visualized_surf, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)), (0, 0))
        # center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        # self.screen.blit(state_visualized_surf, (center_x, center_y))
        pygame.display.flip()

    def play_game_with_human(self, time_ticks=True):
        done = False
        total_reward = 0

        (observation_len,) = self.other_agent.observation_space.shape
        self.action_sequence = [8, 11, 8, 11, 12]
        self.current_action_idx = 0

        self.env.reset()
        clock = pygame.time.Clock()
        self.visualize_state(self.env.state)
        clock.tick(60)

        while not done:
            action = self.get_human_action(time_ticks=time_ticks)
            _, reward, done, info = self.env.step(action)
            if self.env.previous_ego_action == 'interact':
                self.current_action_idx += 1
            total_reward += reward
            self.visualize_state(self.env.state)
            clock.tick(60)

if __name__ == "__main__":
    demo = OvercookedGameDemo()
    demo.play_game_with_human()
