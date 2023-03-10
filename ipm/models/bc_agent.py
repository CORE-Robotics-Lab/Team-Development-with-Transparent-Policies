import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sys
if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    import pickle5 as pickle
else:
    import pickle
from sklearn.model_selection import train_test_split


class IntentModel:
    def __init__(self, layout, observations, reduced_observations, actions, player_idx, states=None, traj_lengths=None):
        self.layout = layout
        self.observations = observations
        self.reduced_observations = reduced_observations
        if not layout == 'tutorial:':
            assert len(observations[0]) == 96 # check that we are using raw observations
        self.actions = actions
        self.verify_with_states = states is not None
        self.states = states
        self.player_idx = player_idx
        # by default, use sklearn random forest
        # self.model = RandomForestClassifier(n_estimators=3, max_depth=10, random_state=0)
        self.model = DecisionTreeClassifier(max_depth=3, random_state=0)
        # self.model.fit(self.observations, self.actions)
        self.rf_model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)

        # split data into episodes
        trajectory_observations_raw = []
        trajectory_observations_reduced = []
        trajectory_states = []
        trajectory_actions = []
        counter = 0
        for i in traj_lengths:
            trajectory_observations_raw.append(self.observations[counter:counter+i])
            trajectory_observations_reduced.append(self.reduced_observations[counter:counter+i])
            trajectory_actions.append(self.actions[counter:counter+i])
            if self.verify_with_states:
                trajectory_states.append(self.states[counter:counter+i])
            counter += i

        if self.layout == "two_rooms_narrow":
            item_mapping = {(1,0,0,0):'onion',
                            (0,1,0,0):'soup',
                            (0,0,1,0):'dish',
                            (0,0,0,1):'tomato',
                            (0,0,0,0): "nothing"}
        else:
            item_mapping = {(1,0,0):'onion',
                            (0,1,0):'soup',
                            (0,0,1):'dish',
                            (0,0,0): "nothing"}
        if not self.layout == "two_rooms_narrow":
            action_mapping = {
                "Nothing": 0,
                "Picking Up Onion From Dispenser": 1,
                "Picking Up Onion From Counter": 2,
                "Picking Up Dish From Dispenser": 3,
                "Picking Up Dish From Counter": 4,
                "Picking Up Soup From Pot": 5,
                "Picking Up Soup From Counter": 6,
                "Serving At Dispensary": 7,
                "Bringing To Closest Pot": 8,
                "Placing On Closest Counter": 9,
                "Turning On Cook Timer": 10,
            }
        else:
            action_mapping = {
                "Nothing": 0,
                "Picking Up Onion From Dispenser": 1,
                "Picking Up Onion From Counter": 2,
                "Picking Up Tomato From Dispenser": 3,
                "Picking Up Tomato From Counter": 4,
                "Picking Up Dish From Dispenser": 5,
                "Picking Up Dish From Counter": 6,
                "Picking Up Soup From Pot": 7,
                "Picking Up Soup From Counter": 8,
                "Serving At Dispensary": 9,
                "Bringing To Closest Pot": 10,
                "Placing On Closest Counter": 11,
                "Turning On Cook Timer": 12,
            }

        intent_mapping = {
            "Nothing": 0,
            "Picking Up Ingredient": 1,
            "Picking Up Dish": 3,
            "Picking Up Soup": 4,
            "Serving Soup": 5,
            "Bringing To Closest Pot": 10,
            "Placing Item Down": 11,
            "Turning On Cook Timer": 12,
        }

        total_observations_raw = []
        total_observations_reduced = []
        total_high_level_actions = []
        total_primitive_actions = []

        for k in range(len(traj_lengths)):

            # go through and find all the indices where the action is 5
            indices = [i for i in range(len(trajectory_actions[k])) if trajectory_actions[k][i] == 5]

            if indices[-1] == traj_lengths[k]-1:
                # if last action is an interact, then there will be no next timestep.
                indices.remove(indices[-1])

            indices_array = np.array(indices)
            episode_observations = []
            episode_observations_reduced = []
            episode_high_level_actions = []
            episode_primitive_actions = []
            episode_action_dict = {
            }
            for e,i in enumerate(indices):
                # look at transition and find out what happened
                if self.layout == "two_rooms_narrow":
                    n_ingredients = 4
                else:
                    n_ingredients = 3

                start_idx = 4
                before_object = item_mapping[tuple(trajectory_observations_raw[k][i][start_idx:start_idx + n_ingredients])]
                after_object = item_mapping[tuple(trajectory_observations_raw[k][i+1][start_idx:start_idx + n_ingredients])]

                start_idx = 4 + 46
                before_object_other_p = item_mapping[tuple(trajectory_observations_raw[k][i][start_idx:start_idx + n_ingredients])]
                after_object_other_p = item_mapping[tuple(trajectory_observations_raw[k][i+1][start_idx:start_idx + n_ingredients])]

                if self.verify_with_states:
                    before_state = trajectory_states[k][i]
                    after_state = trajectory_states[k][i+1]

                if after_object == 'onion' and before_object == "nothing":
                    onion_on_counter_idx = -2
                    if self.layout == "two_rooms_narrow":
                        onion_on_counter_idx -= 1
                    onion_on_counter_before = trajectory_observations_reduced[k][i][onion_on_counter_idx]
                    onion_on_counter_after = trajectory_observations_reduced[k][i+1][onion_on_counter_idx]
                    if onion_on_counter_before == 1 and onion_on_counter_after == 0:
                        action_taken = "Picking Up Onion From Counter"
                    else:
                        action_taken = "Picking Up Onion From Dispenser"
                    if self.verify_with_states:
                        assert before_state.players[self.player_idx].held_object is None and \
                               after_state.players[self.player_idx].held_object.name == 'onion'
                elif after_object == 'soup' and before_object == "dish":
                    soup_on_counter_idx = -1
                    soup_on_counter_before = trajectory_observations_reduced[k][i][soup_on_counter_idx]
                    soup_on_counter_after = trajectory_observations_reduced[k][i+1][soup_on_counter_idx]
                    if soup_on_counter_before == 1 and soup_on_counter_after == 0:
                        action_taken = "Picking Up Soup From Counter"
                    else:
                        action_taken = "Picking Up Soup From Pot"
                    if self.verify_with_states:
                        assert before_state.players[self.player_idx].held_object.name == 'dish' and \
                               after_state.players[self.player_idx].held_object.name == 'soup'
                elif after_object == 'dish' and before_object == "nothing":
                    dish_on_counter_idx = -1
                    dish_on_counter_before = trajectory_observations_reduced[k][i][dish_on_counter_idx]
                    dish_on_counter_after = trajectory_observations_reduced[k][i+1][dish_on_counter_idx]
                    if dish_on_counter_before == 1 and dish_on_counter_after == 0:
                        action_taken = "Picking Up Dish From Counter"
                    else:
                        action_taken = "Picking Up Dish From Dispenser"
                    if self.verify_with_states:
                        assert before_state.players[self.player_idx].held_object is None and \
                               after_state.players[self.player_idx].held_object.name == 'dish'
                elif after_object == 'nothing' and before_object == "onion":
                    onion_on_counter_idx = -2
                    if self.layout == "two_rooms_narrow":
                        onion_on_counter_idx -= 1
                    onion_on_counter_before = trajectory_observations_reduced[k][i][onion_on_counter_idx]
                    onion_on_counter_after = trajectory_observations_reduced[k][i+1][onion_on_counter_idx]
                    if onion_on_counter_before == 0 and onion_on_counter_after == 1:
                        # let's also check if an extra ingredient is in the pot, if so then
                        # let's take a 50/50 chance on either option
                        action_taken = "Placing On Closest Counter"
                    else:
                        action_taken = "Bringing To Closest Pot"
                    if self.verify_with_states:
                        assert before_state.players[self.player_idx].held_object.name == 'onion' and \
                               after_state.players[self.player_idx].held_object is None
                elif after_object == 'nothing' and before_object == "dish":
                    dish_on_counter_idx = -3
                    if self.layout == "two_rooms_narrow":
                        dish_on_counter_idx -= 1
                    dish_on_counter_before = trajectory_observations_reduced[k][i][dish_on_counter_idx]
                    dish_on_counter_after = trajectory_observations_reduced[k][i + 1][dish_on_counter_idx]
                    if dish_on_counter_before == 0 and dish_on_counter_after == 1:
                        action_taken = "Placing On Closest Counter"
                    else:
                        # we ended up here because the other player picked up the dish
                        action_taken = "Placing On Closest Counter"
                        if self.verify_with_states:
                            raise ValueError("This should not happen")
                    if self.verify_with_states:
                        assert before_state.players[self.player_idx].held_object.name == 'dish' and \
                               after_state.players[self.player_idx].held_object is None
                elif after_object == 'nothing' and before_object == "soup":
                    # decide between Placing On Closest Counter and Serving At Dispensary
                    soup_on_counter_idx = -1
                    soup_on_counter_before = trajectory_observations_reduced[k][i][soup_on_counter_idx]
                    soup_on_counter_after = trajectory_observations_reduced[k][i + 1][soup_on_counter_idx]
                    if soup_on_counter_before == 0 and soup_on_counter_after == 1:
                        action_taken = 'Placing On Closest Counter'
                    else:
                        action_taken = "Serving At Dispensary"
                    if self.verify_with_states:
                        assert before_state.players[self.player_idx].held_object.name == 'soup' and \
                               after_state.players[self.player_idx].held_object is None
                else:
                    # decide between put oven timer on or nothing
                    action_taken = "Nothing"

                episode_action_dict[i] = action_taken

               # now let's get intent

            # go through a second time and pair each observation with action
            for timestep, trajectories in enumerate(trajectory_observations_raw[k]):
                try:
                    next_action = indices_array[indices_array > timestep].min()
                except:
                    # no next action
                    continue
                episode_observations.append(trajectories)
                episode_primitive_actions.append(trajectory_actions[k][timestep])
                episode_high_level_actions.append(action_mapping[episode_action_dict[next_action]])

            total_observations_raw.extend(episode_observations)
            total_primitive_actions.extend(episode_primitive_actions)
            total_high_level_actions.extend(episode_high_level_actions)

        self.high_level_actions = total_high_level_actions

        # aggregate every 2 states and 2 actions into a single vector, then use the high_level_action as the label
        X = []
        Y = []
        for i in range(1, len(total_observations_raw)):
            features = [total_observations_raw[i-1], [total_primitive_actions[i-1]],
                        total_observations_raw[i]]
            features = np.concatenate(features, axis=0)
            X.append(features)
            Y.append(total_high_level_actions[i])

        assert len(X[0]) == 2 * len(total_observations_raw[0]) + 1
        assert len(X) == len(Y)

        # training
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2,
                                                                                random_state=42)
        self.model.fit(self.X_train, self.y_train)
        # check validation accuracy
        print("Validation accuracy for high-level BC model: ", self.model.score(self.X_test, self.y_test))

        self.rf_model.fit(self.X_train, self.y_train)
        print("Validation accuracy for high-level BC model (more complex): ", self.rf_model.score(self.X_test, self.y_test))
        # accuracy_threshold = 0.6
        #if self.model.score(self.X_test, self.y_test) < accuracy_threshold:
        #    raise ValueError("BC model accuracy is too low! Please collect more data or use a different model.")

        # train on all the data
        self.model.fit(X, Y)
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

def get_human_bc_partner(traj_directory, layout_name, bc_agent_idx, include_states=False, get_intent_model=False):
    # load each csv file into a dataframe
    dfs = []
    raw_states = []
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

            if include_states:
                states_filename = filename.replace('.csv', '_states.pkl')
                with open(os.path.join(traj_directory, states_filename), 'rb') as f:
                    # check python version and use pickle5 if necessary
                    raw_states.append(pickle.load(f))

                # this code FIXES raw observations which were saved incorrectly. Code may not be necessary,
                # but it's here just in case.
                # from ipm.overcooked.overcooked_multi import OvercookedSelfPlayEnv
                # env = OvercookedSelfPlayEnv(layout_name=layout_name, ego_idx=bc_agent_idx,
                #                                  reduced_state_space_ego=True,
                #                                  reduced_state_space_alt=True,
                #                                  use_skills_ego=True,
                #                                  use_skills_alt=True,
                #                                  n_timesteps=200)
                # df = pd.read_csv(os.path.join(traj_directory, filename))
                # state2obs = env.featurize_fn
                # all_old_obs_str = df['obs']
                #
                # def get_raw_obs(row):
                #     row_idx = row.name
                #     return state2obs(raw_states[-1][row_idx])[row['agent_idx']]
                #
                # all_raw_obs = df.apply(get_raw_obs, axis=1)
                #
                # # validate results
                # for i in range(len(all_raw_obs)):
                #     raw_obs = all_raw_obs[i]
                #     obs_str = all_old_obs_str[i].replace('\n', '')
                #     old_obs = np.fromstring(obs_str[1:-1], dtype=float, sep=' ')
                #     if not layout_name == 'two_rooms_narrow':
                #         n_ingredients = 3
                #     else:
                #         n_ingredients = 4
                #     for j in range(n_ingredients):
                #         # make sure we are looking at the right agent
                #         assert raw_obs[4 + j] == old_obs[j]
                #
                # df['raw_obs'] = all_raw_obs
                # df.to_csv(os.path.join(traj_directory, filename), index=False)

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

    if include_states:
        raw_states = np.concatenate(raw_states, axis=0)
    # convert states to observations

    # we simply want to use the state -> observation fn from this env
    # env = OvercookedSelfPlayEnv(layout_name=layout_name, seed_num=0,
    #                             reduced_state_space_ego=False,
    #                             reduced_state_space_alt=False)
    # states = df['state']
    # for i in range(len(states)):
    #     state = json.loads(states[i])
    #     observations.append(env.featurize_fn(state))

    indices_with_alt_raw = df['agent_idx'] == bc_agent_idx

    indices_with_alt = [i for (i, is_alt) in enumerate(indices_with_alt_raw.values) if is_alt]
    # only get rows where alt_idx is the alt agent
    df = df[indices_with_alt_raw]
    # do the same thing for states

    if include_states:
        states = []
        for i in range(len(raw_states)):
            if i in indices_with_alt:
                states.append(raw_states[i])
        assert len(states) == len(df)
    else:
        states = None

    # # get the length of each trajectory
    # print(len(dfs[-1].episode_num == episode_num))
    # print(len(dfs[-1].values))
    # traj_lengths[layout_name].append(len(dfs[-1].values))

    # string obs to numpy array
    raw_observations = []
    for obs_str in df['raw_obs'].values:
        obs_str = obs_str.replace('\n', '')
        raw_observations.append(np.fromstring(obs_str[1:-1], dtype=float, sep=' '))

    reduced_observations = []
    for obs_str in df['obs'].values:
        obs_str = obs_str.replace('\n', '')
        reduced_observations.append(np.fromstring(obs_str[1:-1], dtype=float, sep=' '))

    actions = df['action'].values

    if len(raw_observations) == 0:
        print('No data found for alt_idx: ', bc_agent_idx)
        print('Using "STAY" agent instead')
        return StayAgent()

    if get_intent_model:
        intent_model = IntentModel(layout_name, raw_observations, reduced_observations, actions, bc_agent_idx, states, traj_lengths)
        high_level_actions = intent_model.high_level_actions
        return (intent_model,
                BCAgent(raw_observations, high_level_actions)
                )
    else:
        return BCAgent(raw_observations, actions)
