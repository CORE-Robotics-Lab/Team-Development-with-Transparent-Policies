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


class OvercookedGameDemo:
    def __init__(self, screen=None, other_agent=None,
                 layout_name='forced_coordination_tomato', horizon_length=15,
                 SCREEN_WIDTH=1920, SCREEN_HEIGHT=1080,
                 use_custom_visualizer=False):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.skills = Skills(robot_index=1)

        if use_custom_visualizer is False:
            self.visualizer = StateVisualizer()

        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = screen

        if other_agent is None:
            default_agent_filepath = '../../../cs-7648/logs/results/1213_01_44/modified_agent/rl_model_500000_steps.zip'
            self.other_agent = PPO.load(default_agent_filepath)
        else:
            self.other_agent = other_agent

        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": layout_name},
            env_params={"horizon": horizon_length},
        )

        self.horizon_env = ae.env.copy()

        self.idx_to_action = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), 'interact']
        # translating commands into actions
        self.action_dict = {'right': (1, 0),
                            'left': (-1, 0),
                            'up': (0, -1),
                            'down': (0, 1),
                            'interact': 'interact',
                            'stay': (0, 0)}

        self.horizon_env.start_state_fn = ae.env.start_state_fn
        self.horizon_env.reset()
        agent1 = RandomAgent()
        agent2 = RandomAgent()
        self.agent_pair = AgentPair(agent1, agent2)
        self.horizon_env.reset()

    def play_game_with_human(self):
        done = False
        while not self.horizon_env.is_done():
            clock = pygame.time.Clock()
            command = None

            s_t = self.horizon_env.state

            # force the user to make a move
            # if we want to run the game continuously:
            # we would need to use a timer to keep track of seconds elapsed
            while command is None and not done:
                for event in pygame.event.get():

                    # no quitting allowed :) must complete experiment fully!
                    # if event.type == pygame.QUIT:
                    #     exit(0)

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            command = 'left'
                        elif event.key == pygame.K_UP:
                            command = 'up'
                        elif event.key == pygame.K_DOWN:
                            command = 'down'
                        elif event.key == pygame.K_RIGHT:
                            command = 'right'
                        elif event.key == pygame.K_SPACE:
                            command = 'interact'

                state_visualized_surf = self.visualizer.render_state(state=s_t, grid=self.horizon_env.mdp.terrain_mtx)
                self.screen.blit(pygame.transform.scale(state_visualized_surf, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)), (0, 0))
                pygame.display.flip()
                clock.tick(60)

            joint_action_and_infos = self.agent_pair.joint_action(s_t)
            # command is integrated by replacing part of the joint actions!

            a_t, a_info_t = zip(*joint_action_and_infos)

            modified_a_t = list(a_t)
            modified_a_t[0] = self.action_dict[command]

            obs_p2 = self.horizon_env.featurize_state_mdp(s_t)[1]
            robot_skill_idx = self.other_agent.predict(obs_p2)
            # for now, let's just get an onion.
            # TODO: add ability to pick up items from counter
            robot_skill_idx = 0
            robot_skill_primitive_moves, _, _ = self.skills.idx_to_skill[robot_skill_idx](self.horizon_env)
            # take the first move in the sequence
            modified_a_t[1] = robot_skill_primitive_moves[0]

            modified_a_t = tuple(modified_a_t)
            assert all(a in Action.ALL_ACTIONS for a in a_t)
            assert all(type(a_info) is dict for a_info in a_info_t)
            display_phi = False
            s_t, r_t, done, info = self.horizon_env.step(modified_a_t, a_info_t, display_phi)

if __name__ == "__main__":
    demo = OvercookedGameDemo()
    demo.play_game_with_human()
