from stable_baselines3 import PPO

import pygame
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai.src.overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
import pygame
from stable_baselines3 import PPO
from ipm.overcooked.high_level_actions import Behaviors

from overcooked_ai.src.overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer


class OvercookedGameDemo:
    def __init__(self, screen=None, other_agent=None, layout_name='forced_coordination'):
        layout_name = 'cramped_room'
        self.SCREEN_WIDTH = 1500
        self.SCREEN_HEIGHT = 800
        self.skills = Behaviors(1)

        if screen is None:
            # initialize some pygame things
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = screen

        if other_agent is None:
            default_agent_filepath = '../../../cs-7648/logs/results/1213_01_44/modified_agent/rl_model_500000_steps.zip'
            self.other_agent = PPO.load(default_agent_filepath)
        else:
            self.other_agent = other_agent

        self.board_dict = {'get_square': {}, 'board_surf': {}, 'draw_pieces': {}, 'draw_selector': {}, 'draw_diag': {}}

        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": layout_name},
            env_params={"horizon": 400},
        )

        self.horizon_env = ae.env.copy()
        self.LAYOUT_X = self.horizon_env.mdp.width
        self.LAYOUT_Y = self.horizon_env.mdp.height

        # colors for pygame
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.SIDE_COLOR = (154, 100, 23)
        self.SERVE_COLOR = (183, 183, 183)
        self.DISH_COLOR = (255, 204, 102)

        self.TILE_SIZE = 150
        self.INGREDIENT_TILESCALE = 0.99
        self.BOARD_WIDTH = self.LAYOUT_X * self.TILE_SIZE
        self.BOARD_HEIGHT = self.LAYOUT_Y * self.TILE_SIZE
        self.BORDER_WIDTH = 1
        self.BORDER_PAD = 60
        self.SCREEN_CENTER_X = self.SCREEN_WIDTH / 2
        self.SCREEN_CENTER_Y = self.SCREEN_HEIGHT / 2

        self.BOARD_POS = (self.SCREEN_CENTER_X - self.BOARD_WIDTH / 2, self.SCREEN_CENTER_Y - self.BOARD_HEIGHT / 2)

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
        # agent.set_mdp(horizon_env.mdp)
        self.agent_pair = AgentPair(agent1, agent2)
        self.horizon_env.reset()
        # env = OvercookedSelfPlayEnv(layout_name='cramped_room', mlam=horizon_env.mlam)

    def draw_tile(self, surface, imgname, tilepos, tilescale, bordercolor=None, **kwargs):
        img = pygame.image.load(imgname)
        img = pygame.transform.scale(img, (tilescale * self.TILE_SIZE, tilescale * self.TILE_SIZE))

        rect = img.get_rect()
        for key, val in kwargs.items():
            setattr(rect, key, val)
        surface.blit(img, tilepos)

        if bordercolor is not None:
            pygame.draw.rect(img, bordercolor, [0, 0, tilepos[0], tilepos[1]], self.BORDER_WIDTH)

    def create_board_surf(self, horizon_env, screen, board_dict):
        board_surf = pygame.Surface((self.TILE_SIZE * self.LAYOUT_X, self.TILE_SIZE * self.LAYOUT_Y))
        # TODO: add this path into __init__.py
        image_folderpath = '../../images'

        for y in range(self.LAYOUT_Y):
            for x in range(self.LAYOUT_X):
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                board_dict['board_surf'][(x, y)] = (x * self.TILE_SIZE, y * self.TILE_SIZE)

                tilepos = (rect.center[0] - self.TILE_SIZE / 2, rect.center[1] - self.TILE_SIZE / 2)

                tilespec = horizon_env.mdp.terrain_mtx[y][x]

                # TODO: add player 2 info
                if (x, y) == horizon_env.state.players[0].position:
                    # player 1 (HUMAN)

                    if horizon_env.state.players[0].orientation == (0, -1):
                        imgfile = image_folderpath + '/blue_up.png'
                        if horizon_env.state.players[0].held_object is not None and horizon_env.state.players[0].held_object.name == 'onion':
                            imgfile = image_folderpath + '/blue_up_onion.png'
                    elif horizon_env.state.players[0].orientation == (0, 1):
                        imgfile = image_folderpath + '/blue_down.png'
                        if horizon_env.state.players[0].held_object is not None and horizon_env.state.players[0].held_object.name == 'onion':
                            imgfile = image_folderpath + '/blue_down_onion.png'
                    elif horizon_env.state.players[0].orientation == (-1, 0):
                        imgfile = image_folderpath + '/blue_left.png'
                        if horizon_env.state.players[0].held_object is not None and horizon_env.state.players[0].held_object.name == 'onion':
                            imgfile = image_folderpath + '/blue_left_onion.png'
                    elif horizon_env.state.players[0].orientation == (1, 0):
                        imgfile = image_folderpath + '/blue_right.png'
                        if horizon_env.state.players[0].held_object is not None and horizon_env.state.players[0].held_object.name == 'onion':
                            imgfile = image_folderpath + '/blue_right_onion.png'
                    else:
                        raise ValueError('Invalid player orientation')

                    tilescale = 1
                    self.draw_tile(board_surf, imgfile, tilepos, tilescale, center=rect.center)
                elif (x, y) == horizon_env.state.players[1].position:
                    if horizon_env.state.players[1].orientation == (0, -1):
                        imgfile = image_folderpath + '/red_up.png'
                        if horizon_env.state.players[1].held_object is not None and horizon_env.state.players[1].held_object.name == 'onion':
                            imgfile = image_folderpath + '/red_up_onion.png'
                    elif horizon_env.state.players[1].orientation == (0, 1):
                        imgfile = image_folderpath + '/red_down.png'
                        if horizon_env.state.players[1].held_object is not None and horizon_env.state.players[1].held_object.name == 'onion':
                            imgfile = image_folderpath + '/red_down_onion.png'
                    elif horizon_env.state.players[1].orientation == (-1, 0):
                        imgfile = image_folderpath + '/red_left.png'
                        if horizon_env.state.players[1].held_object is not None and horizon_env.state.players[1].held_object.name == 'onion':
                            imgfile = image_folderpath + '/red_left_onion.png'
                    elif horizon_env.state.players[1].orientation == (1, 0):
                        imgfile = image_folderpath + '/red_right.png'
                        if horizon_env.state.players[1].held_object is not None and horizon_env.state.players[1].held_object.name == 'onion':
                            imgfile = image_folderpath + '/red_right_onion.png'
                    else:
                        raise ValueError('Invalid player orientation')

                    tilescale = 1
                    self.draw_tile(board_surf, imgfile, tilepos, tilescale, center=rect.center)
                elif tilespec == ' ':
                    pygame.draw.rect(board_surf, pygame.Color('beige'), rect)
                    pygame.draw.rect(board_surf, pygame.Color(self.BLACK), rect, 1)
                elif tilespec == 'S':
                    pygame.draw.rect(board_surf, pygame.Color(self.SERVE_COLOR), rect)
                    pygame.draw.rect(board_surf, pygame.Color(self.BLACK), rect, 1)
                else:
                    # Tile is a counter that can hold an object
                    tilescale = 1

                    if tilespec == 'O':
                        # onion
                        self.draw_tile(board_surf, image_folderpath + '/onions.png', tilepos, tilescale, bordercolor=self.BLACK,
                                       center=rect.center)

                    elif tilespec == "P":
                        # oven
                        self.draw_tile(board_surf, image_folderpath + '/oven.png', tilepos, tilescale, bordercolor=self.BLACK,
                                       center=rect.center)

                    elif tilespec == "D":
                        # oven
                        self.draw_tile(board_surf, image_folderpath + '/dish.png', tilepos, tilescale, bordercolor=self.BLACK,
                                       center=rect.center)


                    elif horizon_env.state.has_object((x, y)):
                        itemname = horizon_env.state.get_object((x, y)).name

                        tilescale = self.INGREDIENT_TILESCALE

                        if itemname == 'onion':
                            self.draw_tile(board_surf, image_folderpath + '/single_onion.png', tilepos, tilescale, bordercolor=self.BLACK,
                                           center=rect.center)
                        elif itemname == 'dish':
                            self.draw_tile(board_surf, image_folderpath + '/single_dish.png', tilepos, tilescale, bordercolor=self.BLACK,
                                           center=rect.center)
                        # elif itemname == 'onion':
                        #     drawtile(board_surf, 'dish image', tilepos, tilescale, bordercolor=self.BLACK, center=rect.center)
                        elif itemname == 'soup':
                            ingredients = horizon_env.state.get_object((x, y)).ingredients
                            if ingredients == ['onion']:
                                self.draw_tile(board_surf, 'soup image', tilepos, tilescale, bordercolor=self.BLACK,
                                               center=rect.center)

                    else:
                        # empty
                        pygame.draw.rect(board_surf, pygame.Color(self.SIDE_COLOR), rect)
                        pygame.draw.rect(board_surf, pygame.Color(self.BLACK), rect, 1)

        # make board
        board_dict['board_surf_range'] = {}
        board_dict['board_surf_range']['x'] = {}
        board_dict['board_surf_range']['y'] = {}

        for y in range(self.LAYOUT_Y):
            for x in range(self.LAYOUT_X):
                try:
                    board_dict['board_surf_range']['x'][x] = (
                        board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x + 1, y)][0])
                    board_dict['board_surf_range']['y'][y] = (
                        board_dict['board_surf'][(x, y)][1], board_dict['board_surf'][(x, y + 1)][1])
                except KeyError:
                    if x == self.LAYOUT_X - 1 and y != self.LAYOUT_Y - 1:
                        board_dict['board_surf_range']['x'][x] = (
                            board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x, y)][0] + self.TILE_SIZE)
                        board_dict['board_surf_range']['y'][y] = (
                            board_dict['board_surf'][(x, y)][1], board_dict['board_surf'][(x, y + 1)][1])
                    elif x != self.LAYOUT_X - 1 and y == self.LAYOUT_Y - 1:
                        board_dict['board_surf_range']['x'][x] = (
                            board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x + 1, y)][0])
                        board_dict['board_surf_range']['y'][y] = (
                            board_dict['board_surf'][(x, y)][1], board_dict['board_surf'][(x, y)][1] + self.TILE_SIZE)
                    else:
                        board_dict['board_surf_range']['x'][x] = (
                            board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x, y)][0] + self.TILE_SIZE)
                        board_dict['board_surf_range']['y'][y] = (
                            board_dict['board_surf'][(x, y)][1], board_dict['board_surf'][(x, y)][1] + self.TILE_SIZE)

        return board_surf


    def create_board(self):
        board = []
        for y in range(self.LAYOUT_Y):
            board.append([])
            for x in range(self.LAYOUT_X):
                board[y].append(None)

        for x in range(0, self.LAYOUT_X):
            board[0][x] = ('black', ' ')
        for x in range(0, self.LAYOUT_X):
            board[1][x] = ('black')

        return board

    def draw_pieces(self, screen, board, font, selected_piece, horizon_env, board_dict):
        sx, sy = None, None
        if selected_piece:
            piece, sx, sy = selected_piece

        for x in range(0, self.LAYOUT_X):
            board[0][x] = ('black', ' ')

        for x in range(0, self.LAYOUT_X):
            board[1][x] = ('black', ' ')

        for y in range(self.LAYOUT_Y):
            for x in range(self.LAYOUT_X):
                piece = board[y][x]
                if piece:
                    selected = x == sx and y == sy
                    color, type = piece
                    s1 = font.render(type, True, pygame.Color('red' if selected else color))
                    s2 = font.render(type, True, pygame.Color('darkgrey'))
                    pos = pygame.Rect(self.BOARD_POS[0] + x * self.TILE_SIZE + 1, self.BOARD_POS[1] + y * self.TILE_SIZE + 1, self.TILE_SIZE, self.TILE_SIZE)
                    if not False:
                        board_dict['draw_pieces'][(x, y)] = (
                            self.BOARD_POS[0] + x * self.TILE_SIZE + 1, self.BOARD_POS[1] + y * self.TILE_SIZE + 1)
                    screen.blit(s2, s2.get_rect(center=pos.center).move(1, 1))
                    screen.blit(s1, s1.get_rect(center=pos.center))


    def draw_selector(self, screen, piece, x, y, board_dict):
        if piece != None:
            rect = (self.BOARD_POS[0] + x * self.TILE_SIZE, self.BOARD_POS[1] + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            board_dict['draw_selector'] = (self.BOARD_POS[0] + x * self.TILE_SIZE, self.BOARD_POS[1] + y * self.TILE_SIZE)
            pygame.draw.rect(screen, (255, 0, 0, 50), rect, 2)

    def draw_drag(self, screen, board, selected_piece, font, board_dict):
        return None

    def run_overcooked(self):
        done = False
        while not self.horizon_env.is_done():
            clock = pygame.time.Clock()
            action_not_taken = True

            board = self.create_board()
            board_surf = self.create_board_surf(self.horizon_env, self.screen, self.board_dict)
            s_t = self.horizon_env.state
            print(s_t)
            print(self.horizon_env)
            while action_not_taken and not done:
                font = pygame.font.Font('freesansbold.ttf', 16)
                events = pygame.event.get()
                for e in events:
                    if e.type == pygame.QUIT:
                        exit(0)

                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_LEFT:
                            action_not_taken = False
                            command = 'left'
                        elif e.key == pygame.K_UP:
                            action_not_taken = False
                            command = 'up'
                        elif e.key == pygame.K_DOWN:
                            action_not_taken = False
                            command = 'down'
                        elif e.key == pygame.K_RIGHT:
                            action_not_taken = False
                            command = 'right'
                        elif e.key == pygame.K_SPACE:
                            action_not_taken = False
                            command = 'interact'

                self.screen.fill(pygame.Color('white'))
                self.screen.blit(board_surf, self.BOARD_POS)
                selected_piece=None
                piece, x, y = None, None, None
                self.draw_pieces(self.screen, board, font, selected_piece, self.horizon_env, self.board_dict)
                self.draw_selector(self.screen, piece, x, y, self.board_dict)
                drop_pos = self.draw_drag(self.screen, board, selected_piece, font, self.board_dict)

                # TODO: add pot status

                # advance the clock
                pygame.display.flip()
                clock.tick(60)

            action_plan = self.skills.get_onion(self.horizon_env)
            new_actions = action_plan[0]
            all_actions = self.horizon_env.mdp.get_actions(self.horizon_env.state)
            # a_t, a_info_t = agent.action(s_t)
            joint_action_and_infos = self.agent_pair.joint_action(s_t)
            # command is integrated by replacing part of the joint actions!

            a_t, a_info_t = zip(*joint_action_and_infos)

            modified_a_t = list(a_t)
            modified_a_t[0] = self.action_dict[command]

            obs_p2 = self.horizon_env.featurize_state_mdp(s_t)[1]
            modified_a_t[1] = self.idx_to_action[self.other_agent.predict(obs_p2)]
            modified_a_t[1] = new_actions[0]

            modified_a_t = tuple(modified_a_t)
            print('a_t', a_t)
            print('modified', modified_a_t)

            assert all(a in Action.ALL_ACTIONS for a in a_t)
            assert all(type(a_info) is dict for a_info in a_info_t)
            display_phi = False
            s_tp1, r_t, done, info = self.horizon_env.step(modified_a_t, a_info_t, display_phi)
            # # Getting actions and action infos (optional) for both agents
            # joint_action_and_infos = agent_pair.joint_action(s_t)

if __name__ == "__main__":
    demo = OvercookedGameDemo()
    demo.run_overcooked()
