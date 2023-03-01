# this script will correct trajectories that were collected with the previous action space
import pandas as pd
import os
import argparse
import numpy as np

def old_obs_to_new(obs, layout_name):
    obs = obs.replace('\n', '')
    obs = np.fromstring(obs[1:-1], dtype=float, sep=' ')

    if layout_name == 'forced_coordination' or layout_name == 'two_rooms'\
            or layout_name == 'tutorial':
        assert len(obs) == 19 # first make sure its the old obs
        # remove features 0-3 and 7-10
        obs = np.delete(obs, [0, 1, 2, 3,
                              7, 8, 9, 10])
        assert len(obs) == 11
    else:
        assert len(obs) == 22
        # remove features 0-3 and 8-11
        obs = np.delete(obs, [0, 1, 2, 3,
                              8, 9, 10, 11])
        assert len(obs) == 14
    return np.array2string(obs)

def correct_demonstrations(traj_directory, layout_name):
    for filename in os.listdir(traj_directory):
        if filename.endswith(".csv") and layout_name in filename:
            if layout_name == 'two_rooms' and 'narrow' in filename:
                continue
            df = pd.read_csv(os.path.join(traj_directory, filename))
            df['obs'] = df['obs'].apply(lambda x: old_obs_to_new(x, layout_name))
            df.to_csv(os.path.join(traj_directory, filename), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Corrects OLD demonstrations')
    parser.add_argument('--traj_directory', type=str, default='trajectories')
    parser.add_argument('--layout_name', type=str, default='tutorial')
    args = parser.parse_args()
    correct_demonstrations(args.traj_directory, args.layout_name)