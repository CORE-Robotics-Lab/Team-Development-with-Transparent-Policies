"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""
import argparse
import os

import gym
from overcookedgym.overcooked_utils import LAYOUT_LIST
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

layout = 'simple'
assert layout in LAYOUT_LIST


def main(seed, n_steps):
    env = gym.make('OvercookedSelfPlayEnv-v0', layout_name=layout)
    # gym_env = get_vectorized_gym_env(
    #   env, 'Overcooked-v0', featurize_fn=lambda x: mdp.lossless_state_encoding(x), **params
    # )
    agent = PPO('MlpPolicy', env, verbose=1, seed=seed)

    # Finally, you can construct an ego agent and train it in the environment
    # ego = PPO('MlpPolicy', env, verbose=1)

    checkpoint_freq = n_steps // 100

    save_dir = os.path.join('logs', 'ppo_self_play', 'seed_{}'.format(seed))

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./" + save_dir + "/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    agent.learn(total_timesteps=n_steps, callback=checkpoint_callback)
    # To visualize the agent:
    # python overcookedgym/overcooked-flask/app.py --modelpath_p0 ../logs/rl_model_500000_steps --modelpath_p1 ../logs/rl_model_50000_steps --layout_name simple


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains self-play agent on overcooked with checkpointing')
    parser.add_argument('--seed', help='the seed number to use', type=int, required=True)
    parser.add_argument('--n_steps', help='the number of steps to train for', type=int, default=500000)
    args = parser.parse_args()
    main(args.seed, args.n_steps)
