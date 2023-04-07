import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO

from ipm.overcooked.overcooked_envs import OvercookedJointEnvironment


def load_experts(directory_of_experts: str) -> list:
    """
    Loads in all the experts from the directory
    :param directory_of_experts: filepath to the directory of experts
    :return: list of experts
    """

    # if new python version, then we need to add in custom objects
    # this is for compatibility due to stable-baselines3
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    experts = []
    for root, dirs, files in os.walk(directory_of_experts):
        for file in files:
            if file.endswith('final_model.zip'):
                experts.append(PPO.load(os.path.join(root, file), custom_objects=custom_objects))
    return experts


def generate_trajectories(experts_directory, layout_name, use_backtracking=False):
    experts_p0 = load_experts(experts_directory)
    experts_p1 = load_experts(experts_directory)

    # load in the joint environment
    env = OvercookedJointEnvironment(layout_name=layout_name)

    reduced_observations_p0 = []
    reduced_observations_p1 = []
    states_p0 = []
    states_p1 = []
    actions_p0 = []
    actions_p1 = []

    for p0, p1 in zip(experts_p0, experts_p1):
        env.reset()
        done = False
        while not done:
            obs = env.reduced_obs
            # get the actions
            action_p0, _ = p0.predict(obs[0])
            action_p1, _ = p1.predict(obs[1])

            reduced_observations_p0.append(obs[0])
            reduced_observations_p1.append(obs[1])
            states_p0.append(env.state)
            states_p1.append(env.state)
            actions_p0.append(action_p0)
            actions_p1.append(action_p1)

            # take the actions
            obs, reward, done, info = env.step((action_p0, action_p1))


    if use_backtracking:
        raise Exception('Backtracking not implemented yet')

    # concat all data and convert to numpy arrays
    X = np.concatenate([np.array(reduced_observations_p0), np.array(reduced_observations_p1)], axis=1)
    Y = np.concatenate([np.array(actions_p0), np.array(actions_p1)], axis=1)

    # save X and Y to files
    np.save('X.npy', X)
    np.save('Y.npy', Y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script will generate expert trajectories')
    parser.add_argument('--experts_directory', help='folder location for the expert agents', type=str)
    parser.add_argument('--layout_name', help='layout name for the environment', type=str)
    parser.add_argument('--use_backtracking', help='whether to use backtracking to get macro actions', type=bool, default=True)
    args = parser.parse_args()
    generate_trajectories(experts_directory=args.experts_directory, layout_name=args.layout_name)

