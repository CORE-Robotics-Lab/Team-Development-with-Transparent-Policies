import argparse
import os
import sys

from stable_baselines3 import PPO


def load_experts(dir):
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    experts = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('final_model.zip'):
                experts.append(PPO.load(os.path.join(root, file), custom_objects=custom_objects))
    return experts


def generate_traejectories(experts_directory):
    experts_p0 = load_experts(experts_directory)
    experts_p1 = load_experts(experts_directory)

    # load in the joint environment
    env = OvercookedJointPlayer...
    reduced_observations_p0 = []
    reduced_observations_p1 = []
    states_p0 = []
    states_p1 = []
    actions_p0 = []
    actions_p1 = []

    for p0, p1 in zip(experts_p0, experts_p1):


    # now that we've recorded many trajectories, we need to apply backtracking to deduce macro-actions

    # save X and Y to files



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script will generate expert trajectories')
    parser.add_argument('--experts_directory', help='folder location for the expert agents', type=str)
    args = parser.parse_args()
    generate_traejectories(experts_directory=args.experts_directory)

