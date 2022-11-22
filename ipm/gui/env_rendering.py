import gym
import pygame
from pygame import gfxdraw
import numpy as np

def render_cartpole(env, mode="human"):
    # copied with minimal modifications from openai gym cartpole file
    screen_width = env.screen_width
    screen_height = env.screen_height

    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    polewidth = 10.0
    polelen = scale * (2 * env.length)
    cartwidth = 50.0
    cartheight = 30.0

    if env.state is None:
        return None

    x = env.state

    if env.screen is None:
        pygame.init()
        pygame.display.init()
        env.screen = pygame.display.set_mode((screen_width, screen_height))
    if env.clock is None:
        env.clock = pygame.time.Clock()

    env.surf = pygame.Surface((screen_width, screen_height))
    env.surf.fill((255, 255, 255))

    l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    axleoffset = cartheight / 4.0
    cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    carty = 100  # TOP OF CART
    cart_coords = [(l, b), (l, t), (r, t), (r, b)]
    cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
    gfxdraw.aapolygon(env.surf, cart_coords, (0, 0, 0))
    gfxdraw.filled_polygon(env.surf, cart_coords, (0, 0, 0))

    l, r, t, b = (
        -polewidth / 2,
        polewidth / 2,
        polelen - polewidth / 2,
        -polewidth / 2,
    )

    pole_coords = []
    for coord in [(l, b), (l, t), (r, t), (r, b)]:
        coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
        coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
        pole_coords.append(coord)
    gfxdraw.aapolygon(env.surf, pole_coords, (202, 152, 101))
    gfxdraw.filled_polygon(env.surf, pole_coords, (202, 152, 101))

    gfxdraw.aacircle(
        env.surf,
        int(cartx),
        int(carty + axleoffset),
        int(polewidth / 2),
        (129, 132, 203),
    )
    gfxdraw.filled_circle(
        env.surf,
        int(cartx),
        int(carty + axleoffset),
        int(polewidth / 2),
        (129, 132, 203),
    )

    gfxdraw.hline(env.surf, 0, screen_width, carty, (0, 0, 0))

    env.surf = pygame.transform.flip(env.surf, False, True)
    env.screen.blit(env.surf, (0, 0))
    if mode == "human":
        pygame.event.pump()
        env.clock.tick(env.metadata["render_fps"])
        pygame.display.flip()

    if mode == "rgb_array":
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(env.screen)), axes=(1, 0, 2)
        )
    else:
        return env.isopen
