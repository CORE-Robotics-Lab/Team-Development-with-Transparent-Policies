import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class HighLevelBCAgent:
    def __init__(self, observations, actions, traj_lengths=None):
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


        # split data into episodes
        trajectory_observations = []
        trajectory_actions = []
        counter = 0
        for i in traj_lengths:
            trajectory_observations.append(self.observations[counter:counter+i])
            trajectory_actions.append(self.actions[counter:counter+i])
            counter += i

        # need flag for two_rooms_narrow
        item_mapping = {(1,0,0):'onion',
                        (0,1,0):'soup',
                        (0,0,1):'dish',
                        (0,0,0): "nothing"}
        action_mapping = {
            'Onion picked up':0,
            "Soup picked up":1,
            "Dish picked up":2,
            "Onion put down":3,
            "Dish put down":4,
            "Soup put down":5,
            "No action taken":6
        }


        total_observations = []
        total_high_level_actions = []

        for k in range(len(traj_lengths)):

            # go through and find all the indices where the action is 5
            indices = [i for i in range(len(trajectory_actions[k])) if trajectory_actions[k][i] == 5]

            if indices[-1] == traj_lengths[k]-1:
                # if last action is an interact, then there will be no next timestep.
                indices.remove(indices[-1])


            indices_array = np.array(indices)
            episode_observations = []
            episode_actions = []
            episode_action_dict = {
            }
            for e,i in enumerate(indices):
                # look at transition and find out what happend
                before_object = item_mapping[tuple(trajectory_observations[k][i][4:7])]
                after_object = item_mapping[tuple(trajectory_observations[k][i+1][4:7])]


                if after_object == 'onion' and before_object == "nothing":
                    action_taken = "Onion picked up"
                elif after_object == 'soup' and before_object == "dish":
                    action_taken = "Soup picked up"
                elif after_object == 'dish' and before_object == "nothing":
                    action_taken = "Dish picked up"
                elif after_object == 'nothing' and before_object == "onion":
                    action_taken = "Onion put down"
                elif after_object == 'nothing' and before_object == "dish":
                    action_taken = 'Dish put down'
                elif after_object == 'nothing' and before_object == "soup":
                    action_taken = 'Soup put down'
                else:
                    action_taken = "No action taken"

                episode_action_dict[i] = action_taken

            print(episode_action_dict)

            # go through a second time and pair each observation with action
            for e, k in enumerate(trajectory_observations[k]):
                try:
                    next_action = indices_array[indices_array > e].min()
                except:
                    # no next action
                    continue
                episode_observations.append(k)
                episode_actions.append(action_mapping[episode_action_dict[next_action]])


            total_observations.extend(episode_observations)
            total_high_level_actions.extend(episode_actions)




        # training
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(total_observations,
                                                                                total_high_level_actions, test_size=0.2,
                                                                                random_state=42)
        self.model.fit(self.X_train, self.y_train)
        # check validation accuracy
        print("Validation accuracy for BC model: ", self.model.score(self.X_test, self.y_test))
        # accuracy_threshold = 0.6
        #if self.model.score(self.X_test, self.y_test) < accuracy_threshold:
        #    raise ValueError("BC model accuracy is too low! Please collect more data or use a different model.")

        # train on all the data
        self.model.fit(total_observations, total_high_level_actions)
        # self.model.fit(self.observations, self.actions)

    def predict(self, observation):
        _states = None
        return self.model.predict(observation.reshape(1, -1))[0], _states

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
            dfs.append(pd.read_csv(os.path.join(traj_directory, filename)))
            dfs[-1]['episode_num'] = episode_num
            episode_num += 1
            num_files += 1
            if (dfs[-1].agent_idx == 1).sum() > 0:
                traj_lengths.append((dfs[-1].agent_idx == 1).sum())



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

    # # get the length of each trajectory
    # print(len(dfs[-1].episode_num == episode_num))
    # print(len(dfs[-1].values))
    # traj_lengths[layout_name].append(len(dfs[-1].values))

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

    high_level = True
    if high_level:
        return HighLevelBCAgent(observations, actions, traj_lengths)
    else:
        return BCAgent(observations, actions)