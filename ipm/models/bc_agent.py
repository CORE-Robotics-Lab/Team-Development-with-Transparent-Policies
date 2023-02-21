import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class BCAgent:
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions
        # by default, use sklearn random forest
        # self.model = RandomForestClassifier(n_estimators=3, max_depth=10, random_state=0)
        self.model = DecisionTreeClassifier(max_depth=3, random_state=0)
        # self.model.fit(self.observations, self.actions)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.observations, self.actions, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        # check validation accuracy
        print("Validation accuracy for BC model: ", self.model.score(self.X_test, self.y_test))

        self.deep_model = DecisionTreeClassifier(max_depth=10, random_state=0)
        self.deep_model.fit(self.X_train, self.y_train)
        print("Validation accuracy for BC model (deep): ", self.deep_model.score(self.X_test, self.y_test))

        accuracy_threshold = 0.6
        #if self.model.score(self.X_test, self.y_test) < accuracy_threshold:
        #    raise ValueError("BC model accuracy is too low! Please collect more data or use a different model.")

        # train on all the data
        self.model.fit(self.observations, self.actions)

    def predict(self, observation):
        _states = None
        return self.model.predict(observation.reshape(1, -1))[0], _states

class AgentWrapper:
    def __init__(self, agent):
        self.agent = agent

    def predict(self, observation):
        return self.agent.predict(observation), None

class StayAgent:
    def __init__(self):
        pass

    def predict(self, observation):
        return 4, None

def get_human_bc_partner(traj_directory, layout_name, alt_idx):
    # load each csv file into a dataframe
    dfs = []
    episode_num = 0
    num_files = 0
    for filename in os.listdir(traj_directory):
        if filename.endswith(".csv") and layout_name in filename:
            if layout_name == 'two_rooms' and 'narrow' in filename:
                continue
            dfs.append(pd.read_csv(os.path.join(traj_directory, filename)))
            dfs[-1]['episode_num'] = episode_num
            episode_num += 1
            num_files += 1

    if num_files == 0:
        print('No csv files found in directory: ', traj_directory)
        print('Using "STAY" agent instead')
        return StayAgent()

    # aggregate all dataframes into one
    df = pd.concat(dfs, ignore_index=True)
    # convert states to observations

    # we simply want to use the state -> observation fn from this env
    # env = OvercookedSelfPlayEnv(layout_name=layout_name, seed_num=0,
    #                             reduced_state_space_ego=False,
    #                             reduced_state_space_alt=False)
    # states = df['state']
    # for i in range(len(states)):
    #     state = json.loads(states[i])
    #     observations.append(env.featurize_fn(state))

    # only get rows where alt_idx is the alt agent
    df = df[df['agent_idx'] == alt_idx]

    # string obs to numpy array
    observations = []
    for obs_str in df['obs'].values:
        obs_str = obs_str.replace('\n', '')
        observations.append(np.fromstring(obs_str[1:-1], dtype=float, sep=' '))

    actions = df['action'].values

    if len(observations) == 0:
        print('No data found for alt_idx: ', alt_idx)
        print('Using "STAY" agent instead')
        return StayAgent()
    return BCAgent(observations, actions)