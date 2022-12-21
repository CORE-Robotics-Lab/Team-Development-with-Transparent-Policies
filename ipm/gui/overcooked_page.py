import sys

import gym
from stable_baselines3 import PPO

sys.path.insert(0, '../../overcooked_ai/src/overcooked_ai_py')
sys.path.insert(0, '../../overcooked_ai/src')
import numpy as np
import pygame
import time
import random
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai.src.overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from ipm.envs.overcooked_bkup import OvercookedSelfPlayEnv

# colors for pygame
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
side_color = (154, 100, 23)
SERVE_COLOR = (183, 183, 183)
DISH_COLOR = (255, 204, 102)

# layout size
layout_x = 5
layout_y = 4

# some pygame constants
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 800
TILESIZE = 150
INGREDIENT_TILESCALE = 0.99
BOARDWIDTH = layout_x * TILESIZE
BOARDHEIGHT = layout_y * TILESIZE
BORDERWIDTH = 1
BOARDERPAD = 60
SCREENCENTERX = SCREEN_WIDTH / 2
SCREENCENTERY = SCREEN_HEIGHT / 2

BOARD_POS = (SCREENCENTERX - BOARDWIDTH / 2, SCREENCENTERY - BOARDHEIGHT / 2)

idx_to_action = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), 'interact']
# translating commands into actions
action_dict = {'right': (1, 0),
               'left': (-1, 0),
               'up': (0, -1),
               'down': (0, 1),
               'interact':'interact',
               'stay':(0, 0)}


def drawtile(surface, imgname, tilepos, tilescale, bordercolor=None, **kwargs):
    img = pygame.image.load(imgname)
    img = pygame.transform.scale(img, (tilescale * TILESIZE, tilescale * TILESIZE))

    rect = img.get_rect()
    for key, val in kwargs.items():
        setattr(rect, key, val)
    surface.blit(img, tilepos)

    if bordercolor is not None:
        pygame.draw.rect(img, bordercolor, [0, 0, tilepos[0], tilepos[1]], BORDERWIDTH)


def create_board_surf(horizon_env, screen, board_dict):
    board_surf = pygame.Surface((TILESIZE * layout_x, TILESIZE * layout_y))
    image_folderpath = '../../images'

    for y in range(layout_y):
        for x in range(layout_x):
            rect = pygame.Rect(x * TILESIZE, y * TILESIZE, TILESIZE, TILESIZE)
            board_dict['board_surf'][(x, y)] = (x * TILESIZE, y * TILESIZE)

            tilepos = (rect.center[0] - TILESIZE / 2, rect.center[1] - TILESIZE / 2)

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

                tilescale = 1
                drawtile(board_surf, imgfile, tilepos, tilescale, center=rect.center)
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

                tilescale = 1
                drawtile(board_surf, imgfile, tilepos, tilescale, center=rect.center)
            elif tilespec == ' ':
                pygame.draw.rect(board_surf, pygame.Color('beige'), rect)
                pygame.draw.rect(board_surf, pygame.Color(BLACK), rect, 1)
            elif tilespec == 'S':
                pygame.draw.rect(board_surf, pygame.Color(SERVE_COLOR), rect)
                pygame.draw.rect(board_surf, pygame.Color(BLACK), rect, 1)
            else:
                # Tile is a counter that can hold an object
                tilescale = 1

                if tilespec == 'O':
                    # onion
                    drawtile(board_surf, image_folderpath + '/onions.png', tilepos, tilescale, bordercolor=BLACK,
                             center=rect.center)

                elif tilespec == "P":
                    # oven
                    drawtile(board_surf, image_folderpath + '/oven.png', tilepos, tilescale, bordercolor=BLACK,
                             center=rect.center)

                elif tilespec == "D":
                    # oven
                    drawtile(board_surf, image_folderpath + '/dish.png', tilepos, tilescale, bordercolor=BLACK,
                             center=rect.center)


                elif horizon_env.state.has_object((x, y)):
                    itemname = horizon_env.state.get_object((x, y)).name

                    tilescale = INGREDIENT_TILESCALE

                    if itemname == 'onion':
                        drawtile(board_surf, image_folderpath + '/single_onion.png', tilepos, tilescale, bordercolor=BLACK,
                                 center=rect.center)
                    elif itemname == 'dish':
                        drawtile(board_surf, image_folderpath + '/single_dish.png', tilepos, tilescale, bordercolor=BLACK,
                                 center=rect.center)
                    # elif itemname == 'onion':
                    #     drawtile(board_surf, 'dish image', tilepos, tilescale, bordercolor=BLACK, center=rect.center)
                    elif itemname == 'soup':
                        ingredients = horizon_env.state.get_object((x, y)).ingredients
                        if ingredients == ['onion']:
                            drawtile(board_surf, 'soup image', tilepos, tilescale, bordercolor=BLACK,
                                     center=rect.center)

                else:
                    # empty
                    pygame.draw.rect(board_surf, pygame.Color(side_color), rect)
                    pygame.draw.rect(board_surf, pygame.Color(BLACK), rect, 1)

    # make board
    board_dict['board_surf_range'] = {}
    board_dict['board_surf_range']['x'] = {}
    board_dict['board_surf_range']['y'] = {}

    for y in range(layout_y):
        for x in range(layout_x):
            try:
                board_dict['board_surf_range']['x'][x] = (
                    board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x + 1, y)][0])
                board_dict['board_surf_range']['y'][y] = (
                    board_dict['board_surf'][(x, y)][1], board_dict['board_surf'][(x, y + 1)][1])
            except KeyError:
                if x == layout_x - 1 and y != layout_y - 1:
                    board_dict['board_surf_range']['x'][x] = (
                        board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x, y)][0] + TILESIZE)
                    board_dict['board_surf_range']['y'][y] = (
                        board_dict['board_surf'][(x, y)][1], board_dict['board_surf'][(x, y + 1)][1])
                elif x != layout_x - 1 and y == layout_y - 1:
                    board_dict['board_surf_range']['x'][x] = (
                        board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x + 1, y)][0])
                    board_dict['board_surf_range']['y'][y] = (
                        board_dict['board_surf'][(x, y)][1], board_dict['board_surf'][(x, y)][1] + TILESIZE)
                else:
                    board_dict['board_surf_range']['x'][x] = (
                        board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x, y)][0] + TILESIZE)
                    board_dict['board_surf_range']['y'][y] = (
                        board_dict['board_surf'][(x, y)][1], board_dict['board_surf'][(x, y)][1] + TILESIZE)

    return board_surf


def create_board():
    board = []
    for y in range(layout_y):
        board.append([])
        for x in range(layout_x):
            board[y].append(None)

    for x in range(0, layout_x):
        board[0][x] = ('black', ' ')
    for x in range(0, layout_x):
        board[1][x] = ('black')

    return board

def draw_pieces(screen, board, font, selected_piece, horizon_env, board_dict):
    sx, sy = None, None
    if selected_piece:
        piece, sx, sy = selected_piece

    for x in range(0, layout_x):
        board[0][x] = ('black', ' ')

    for x in range(0, layout_x):
        board[1][x] = ('black', ' ')

    for y in range(layout_y):
        for x in range(layout_x):
            piece = board[y][x]
            if piece:
                selected = x == sx and y == sy
                color, type = piece
                s1 = font.render(type, True, pygame.Color('red' if selected else color))
                s2 = font.render(type, True, pygame.Color('darkgrey'))
                pos = pygame.Rect(BOARD_POS[0] + x * TILESIZE + 1, BOARD_POS[1] + y * TILESIZE + 1, TILESIZE, TILESIZE)
                if not False:
                    board_dict['draw_pieces'][(x, y)] = (
                    BOARD_POS[0] + x * TILESIZE + 1, BOARD_POS[1] + y * TILESIZE + 1)
                screen.blit(s2, s2.get_rect(center=pos.center).move(1, 1))
                screen.blit(s1, s1.get_rect(center=pos.center))


def draw_selector(screen, piece, x, y, board_dict):
    if piece != None:
        rect = (BOARD_POS[0] + x * TILESIZE, BOARD_POS[1] + y * TILESIZE, TILESIZE, TILESIZE)
        board_dict['draw_selector'] = (BOARD_POS[0] + x * TILESIZE, BOARD_POS[1] + y * TILESIZE)
        pygame.draw.rect(screen, (255, 0, 0, 50), rect, 2)

def draw_drag(screen, board, selected_piece, font, board_dict):
    return None

TIMES_RAN = 0

def run_overcooked(screen=None, other_agent=None):
    global TIMES_RAN

    if screen is None:
        # initialize some pygame things
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    board_dict = {}
    board_dict['get_square'] = {}
    board_dict['board_surf'] = {}
    board_dict['draw_pieces'] = {}
    board_dict['draw_selector'] = {}
    board_dict['draw_diag'] = {}

    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": "cramped_room"},
        env_params={"horizon": 10},
    )

    horizon_env = ae.env.copy()
    horizon_env.start_state_fn = ae.env.start_state_fn
    horizon_env.reset()
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    # agent.set_mdp(horizon_env.mdp)
    agent_pair = AgentPair(agent1, agent2)
    horizon_env.reset()
    done = False

    #env = OvercookedSelfPlayEnv(layout_name='cramped_room', mlam=horizon_env.mlam)
    import os
    a = os.getcwd()
    other_agent = PPO.load('../../../cs-7648/logs/results/1213_01_44/modified_agent/rl_model_500000_steps.zip')

    while not horizon_env.is_done():
        clock = pygame.time.Clock()
        action_not_taken = True

        board = create_board()
        board_surf = create_board_surf(horizon_env, screen, board_dict)
        s_t = horizon_env.state
        print(s_t)
        print(horizon_env)
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

            screen.fill(pygame.Color('white'))
            screen.blit(board_surf, BOARD_POS)
            selected_piece=None
            piece, x, y = None, None, None
            draw_pieces(screen, board, font, selected_piece, horizon_env, board_dict)
            draw_selector(screen, piece, x, y, board_dict)
            drop_pos = draw_drag(screen, board, selected_piece, font, board_dict)

            # TODO: add pot status

            # advance the clock
            pygame.display.flip()
            clock.tick(60)

        all_actions = horizon_env.mdp.get_actions(horizon_env.state)
        # a_t, a_info_t = agent.action(s_t)
        joint_action_and_infos = agent_pair.joint_action(s_t)
        # command is integrated by replacing part of the joint actions!

        a_t, a_info_t = zip(*joint_action_and_infos)

        modified_a_t = list(a_t)
        modified_a_t[0] = action_dict[command]
        if other_agent is not None:
            obs_p2 = horizon_env.featurize_state_mdp(s_t)[1]
            modified_a_t[1] = idx_to_action[other_agent.predict(obs_p2)]

            # DEMO lol remove this soon!
            # using this bcus a bug in idct constructor? everything else working fine!
            if TIMES_RAN == 0:
                modified_a_t[1] = action_dict['stay']
            else:
                if obs_p2[3] == 1:
                    modified_a_t[1] = 'interact'
                else:
                    modified_a_t[1] = action_dict['left']

        modified_a_t = tuple(modified_a_t)
        print('a_t', a_t)
        print('modified', modified_a_t)

        assert all(a in Action.ALL_ACTIONS for a in a_t)
        assert all(type(a_info) is dict for a_info in a_info_t)
        display_phi = False
        s_tp1, r_t, done, info = horizon_env.step(modified_a_t, a_info_t, display_phi)
        # # Getting actions and action infos (optional) for both agents
        # joint_action_and_infos = agent_pair.joint_action(s_t)
    TIMES_RAN += 1

if __name__ == "__main__":
    run_overcooked()
