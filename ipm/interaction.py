import sys

sys.path.insert(0, '../overcooked_ai/src/overcooked_ai_py')
sys.path.insert(0, '../overcooked_ai/src')
import numpy as np
import pygame
import time
import random
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai.src.overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action



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
TILESIZE = 60
INGREDIENT_TILESCALE = 0.99
BOARDWIDTH = layout_x * TILESIZE
BOARDHEIGHT = layout_y * TILESIZE
BOARDERWIDTH = 1
BOARDERPAD = 60
SCREENCENTERX = SCREEN_WIDTH / 2
SCREENCENTERY = SCREEN_HEIGHT / 2

BOARD_POS = (SCREENCENTERX -BOARDWIDTH/2, BOARDERPAD)


def drawtile(surface, imgname, tilepos, tilescale, bordercolor=None, **kwargs)
    img = pygame.image.load(imgname)
    img = pygame.transform.scale(img, (tilescale * TILESIZE, tilescale * TILESIZE))

    rect = img.get_rect()
    for key, val in kwargs.items():
        setattr(rect, key, val)
    surface.blit(img, tilepos)

    if bordercolor is not None:
        pygame.draw_rect(img, bordercolor, [0, 0, tilepos[0], tilepos[1]], BORDERWIDTH)


def create_board_surf(horizon_env, screen, board_dict):
    board_surf = pygame.Surface(TILESIZE * layout_x, TILESIZE * layout_y)

    for y in range(layout_y):
        for x in range(layout_x):
            rect = pygame.Rect(x * TILESIZE, y * TILESIZE, TILESIZE, TILESIZE)
            board_dict['board_surf'][(x, y)] = (x * TILESIZE, y * TILESIZE)

            tilepos = (rect.center[0] - TILESIZE / 2, rect.center[1] - TILESIZE / 2)

            tilespec = horizon_env.mdp.terrain_mtx[y][x]

            # TODO: add player 2 info
            if (x, y) == horizon_env.state.player[0].position:
                # player 1 (HUMAN)

                if horizon_env.state.players[0].orientation == (0, -1):
                    imgfile = 'image of up'
                elif horizon_env.state.players[0].orientation == (0, 1):
                    imgfile = 'image of down'
                elif horizon_env.state.players[0].orientation == (-1, 0):
                    imgfile = 'image of left'
                elif horizon_env.state.players[0].orientation == (1, 0):
                    imgfile = 'image of right'

                tilescale = 1
                drawtile(board_surf, imgfile, tilepos, tilescale, center=rect.center)
            elif tilespec == ' ':
                pygame.draw(board_surf, pygame.Color('beige'), rect)
                pygame.draw(board_surf, pygame.Color(BLACK), rect, 1)
            elif tilespec == 'S':
                pygame.draw(board_surf, pygame.Color(SERVE_COLOR), rect)
                pygame.draw(board_surf, pygame.Color(BLACK), rect, 1)
            else:
                # Tile is a counter that can hold an object
                tilescale = 1

                if tilespec == 'O':
                    # onion
                    drawtile(board_surf, 'onion image', tilepos, tilescale, bordercolor=BLACK, center=rect.center)

                elif tilespec == "P":
                    # oven
                    drawtile(board_surf, 'oven image', tilepos, tilescale, bordercolor=BLACK, center=rect.center)

                elif tilespec == "D":
                    # oven
                    drawtile(board_surf, 'dish image', tilepos, tilescale, bordercolor=BLACK, center=rect.center)


                elif horizon_env.state.has_object((x, y)):
                    itemname = horizon_env.state.get_object((x, y)).name

                    tilescale = INGREDIENT_TILESCALE

                    if itemname == 'onion':
                        drawtile(board_surf, 'dish image', tilepos, tilescale, bordercolor=BLACK, center=rect.center)
                    elif itemname == 'dish':
                        drawtile(board_surf, 'dish image', tilepos, tilescale, bordercolor=BLACK, center=rect.center)
                    elif itemname == 'onion':
                        drawtile(board_surf, 'dish image', tilepos, tilescale, bordercolor=BLACK, center=rect.center)
                    elif itemname == 'soup':
                        ingredients = horizon_env.state.get_object((x, y)).ingredients

                        if ingredients == ['onion']:
                            drawtile(board_surf, 'dish image', tilepos, tilescale, bordercolor=BLACK,
                                     center=rect.center)

                else:
                    # empty
                    pygame.draw(board_surf, pygame.Color(side_color), rect)
                    pygame.draw(board_surf, pygame.Color(BLACK), rect, 1)

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
                board_dict['board_surf'][(x, y)][1], board_dict['board_surf'][(x, y+1)][1])
            except KeyError:
                if x == layout_x - 1 and y != layout_y - 1:
                    board_dict['board_surf_range']['x'][x] = (
                        board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x, y)][0]+ TILESIZE)
                    board_dict['board_surf_range']['y'][y] = (
                        board_dict['board_surf'][(x, y)][1], board_dict['board_surf'][(x, y + 1)][1])
                elif x != layout_x - 1 and y != layout_y - 1:
                    board_dict['board_surf_range']['x'][x] = (
                        board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x + 1, y)][0])
                    board_dict['board_surf_range']['y'][y] = (
                        board_dict['board_surf'][(x, y)][1], board_dict['board_surf'][(x, y + 1)][1]+TILESIZE)
                else:
                    board_dict['board_surf_range']['x'][x] = (
                        board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x + 1, y)][0]+TILESIZE)
                    board_dict['board_surf_range']['y'][y] = (
                        board_dict['board_surf'][(x, y)][0], board_dict['board_surf'][(x, y + 1)][1]+TILESIZE)

    return board_surf




def main():
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
        env_params={"horizon": 400},
    )

    horizon_env = ae.env.copy()
    horizon_env.start_state_fn = ae.env.start_state_fn
    horizon_env.reset()
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    # agent.set_mdp(horizon_env.mdp)
    agent_pair = AgentPair(agent1, agent2)
    horizon_env.reset()


    action_not_taken = True

    while not horizon_env.is_done():
        board = create_board()
        board_surf = create_board_surf((horizon_env, screen, board_dict))
        s_t = horizon_env.state
        print(s_t)
        all_actions = horizon_env.mdp.get_actions(horizon_env.state)
        # a_t, a_info_t = agent.action(s_t)
        joint_action_and_infos = agent_pair.joint_action(s_t)
        a_t, a_info_t = zip(*joint_action_and_infos)
        assert all(a in Action.ALL_ACTIONS for a in a_t)
        assert all(type(a_info) is dict for a_info in a_info_t)
        display_phi = False
        s_tp1, r_t, done, info = horizon_env.step(a_t, a_info_t, display_phi)
        print(horizon_env)
        # # Getting actions and action infos (optional) for both agents
        # joint_action_and_infos = agent_pair.joint_action(s_t)

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
        draw_pieces(screen, board, font, selected_piece, horizon_env, board_dict)
        draw_selector(screen, piece, x, y, board_dict)
        drop_pos = draw_drag(screen, board, selected_piece, font, board_dict)



if __name__ == "__main__":
    main()