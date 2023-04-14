import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import sys
if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    import pickle5 as pickle
else:
    import pickle
from sklearn.model_selection import train_test_split
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from sklearn.model_selection import train_test_split
from ipm.overcooked.observation_reducer import ObservationReducer
from ipm.models.idct import IDCT
from ipm.models.intent_model import IntentModel
import torch


def load_human_data(traj_directory: str, layout_name: str, bc_agent_idx: int):
    """
    This function performs three steps:
    1) Load in human data from a recent gameplay. This can be done by pointing to a directory containing data
    2) Convert data into X, Y pairs for behavior cloning
    3) Perform gradient descent and display reduced loss.
    Args:
        traj_directory:
        layout_name:
        bc_agent_idx:

    Returns:

    """
    # load each csv file into a dataframe
    dfs = []
    raw_states = []
    reduced_observations = []
    # traj_lengths = {'forced_coordination':[],
    #                 'two_rooms':[],
    #                 'two_rooms_narrow':[]}
    traj_lengths = []
    episode_num = 0
    num_files = 0
    for filename in os.listdir(traj_directory):
        if filename.endswith(".csv") and layout_name in filename:
            if layout_name == 'two_rooms' and 'narrow' in filename:
                continue

            df = pd.read_csv(os.path.join(traj_directory, filename))

            states_filename = filename.replace('.csv', '_states.pkl')
            with open(os.path.join(traj_directory, states_filename), 'rb') as f:
                # check python version and use pickle5 if necessary
                raw_states.append(pickle.load(f))

            dfs.append(df)
            dfs[-1]['episode_num'] = episode_num
            episode_num += 1
            num_files += 1
            n_observations = (df.agent_idx == bc_agent_idx).sum()
            if n_observations > 0:
                traj_lengths.append(n_observations)

    if num_files == 0:
        raise ValueError(f'No csv files found for layout {layout_name}')

    # aggregate all dataframes into one
    df = pd.concat(dfs, ignore_index=True)

    raw_states = np.concatenate(raw_states, axis=0)
    # convert states to observations

    df = df[df['agent_idx'] == bc_agent_idx]
    assert len(raw_states) == len(df)

    observations = []
    for obs_str in df['obs'].values:
        obs_str = obs_str.replace('\n', '')
        observations.append(np.fromstring(obs_str[1:-1], dtype=float, sep=' '))

    actions = df['action'].values

    if len(observations) == 0:
        raise ValueError(f'No observations found for agent {bc_agent_idx} in layout {layout_name}')

    intent_model = IntentModel(layout_name, observations, actions, bc_agent_idx, raw_states, traj_lengths)

    X = intent_model.training_observations_reduced
    Y = intent_model.training_actions
    return X, Y

def finetune_model_to_human_data(nn_ppo_policy, traj_directory: str, layout_name: str, bc_agent_idx: int):
    """
    PPO + BC
    :param nn_ppo_policy: the prior model to finetune
    :param traj_directory: directory containing the trajectories
    :param layout_name: layout_name for overcooked
    :param bc_agent_idx: either 0 or 1.
    :return: the finetuned model
    """

    X, Y = load_human_data(traj_directory, layout_name, bc_agent_idx)

    # fine-tune the prior idct model
    # setup torch training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    idct_ppo_policy.to(device)

    # put data on device
    X = torch.from_numpy(X).float().to(device)
    Y = torch.from_numpy(Y).long().to(device)

    optimizer = torch.optim.Adam(idct_ppo_policy.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    n_epochs = 30
    batch_size = 32
    n_batches = int(np.ceil(len(X) / batch_size))
    for epoch in range(n_epochs):
        epoch_loss = 0
        for i in range(n_batches):
            optimizer.zero_grad()
            batch_X = X[i*batch_size:(i+1)*batch_size]
            batch_Y = Y[i*batch_size:(i+1)*batch_size]
            batch_X = torch.from_numpy(batch_X).float().to(device)
            batch_Y = torch.from_numpy(batch_Y).long().to(device)
            logits = idct_ppo_policy(batch_X)
            loss = criterion(logits, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} loss: {epoch_loss / n_batches}")

    return idct_ppo_policy