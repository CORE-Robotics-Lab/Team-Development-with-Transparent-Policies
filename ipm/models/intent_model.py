import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch
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

class AgentWrapper:
    def __init__(self, agent):
        self.agent = agent

    def predict(self, observation):
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
        return self.agent.predict(observation), None

class StayAgent:
    def __init__(self):
        pass

    def predict(self, observation):
        return 4, None

class IntentModel:
    def __init__(self, layout, observations, actions, player_idx, states, traj_lengths):
        """
        Args:
            layout:
            observations:
            actions:
            player_idx:
            states:
            traj_lengths:
        """

        self.layout = layout
        self.observations = observations
        if not layout == 'tutorial:':
            assert len(observations[0]) == 96 # check that we are using raw observations
        self.actions = actions
        self.verify_with_states = states is not None
        self.states = states
        self.player_idx = player_idx
        self.alt_idx = (player_idx + 1) % 2

        DEFAULT_ENV_PARAMS = {
            # add one because when we reset it takes up a timestep
            "horizon": 200 + 1,
            "info_level": 0,
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        mdp = OvercookedGridworld.from_layout_name(layout_name=layout, rew_shaping_params=rew_shaping_params)
        base_env = OvercookedEnv.from_mdp(mdp, **DEFAULT_ENV_PARAMS)
        featurize_fn = base_env.featurize_state_mdp
        # replace with other function
        self.observation_reducer = ObservationReducer(layout, base_env, cook_time_threshold=5)

        # split data into episodes
        trajectory_observations_raw = []
        trajectory_observations_reduced = []
        trajectory_states = []
        trajectory_actions = []
        counter = 0

        # reduced observation of the human
        for i in traj_lengths:
            trajectory_observations_raw.append(self.observations[counter:counter+i])
            trajectory_actions.append(self.actions[counter:counter+i])
            trajectory_states.append(self.states[counter:counter+i])

            reduced_obs_in_ep = []
            for j in range(i):
                # get reduced observations!
                latest_state = trajectory_states[-1][j]
                (p0_obs, p1_obs) = featurize_fn(latest_state)
                expected_player_observation = (p0_obs, p1_obs)[player_idx]
                latest_observation = trajectory_observations_raw[-1][j]
                assert np.array_equal(expected_player_observation, latest_observation)
                p0_reduced_obs = self.observation_reducer.get_reduced_obs(obs=p0_obs,
                                                                          is_ego=True, # confusing but is_ego basically
                                                                          # means is it player 0's observation
                                                                          state=latest_state)
                p1_reduced_obs = self.observation_reducer.get_reduced_obs(obs=p1_obs,
                                                                          is_ego=False,
                                                                          state=latest_state)
                current_reduced_obs = (p0_reduced_obs, p1_reduced_obs)[player_idx]
                reduced_obs_in_ep.append(current_reduced_obs)

            trajectory_observations_reduced.append(reduced_obs_in_ep)

            counter += i

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
            "Picking Up Onion From Dispenser": 0, # picking up ingredient
            "Picking Up Onion From Counter": 0, # picking up ingredient
            "Picking Up Tomato From Dispenser": 0, # picking up ingredient
            "Picking Up Tomato From Counter": 0, # picking up ingredient
            "Picking Up Dish From Dispenser": 1, # picking up dish
            "Picking Up Dish From Counter": 1, # picking up dish
            "Picking Up Soup From Pot": 2, # picking up soup
            "Picking Up Soup From Counter": 2, # picking up soup
            "Serving At Dispensary": 3, # serving dish
            "Bringing To Closest Pot": 4, # placing item down
            "Placing On Closest Counter": 4, # placing item down
            "Turning On Cook Timer": 5, # for now, we don't care about this action
            "Nothing": 5,
        }

        total_observations_raw = []
        total_observations_reduced = []
        total_high_level_actions = []
        total_primitive_actions = []
        total_intents = []

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
            episode_intents = []
            episode_primitive_actions = []
            episode_action_dict = {
            }
            for e,i in enumerate(indices):
                before_state = trajectory_states[k][i]
                after_state = trajectory_states[k][i + 1]

                before_object = before_state.players[self.player_idx].held_object
                if before_object is None:
                    before_object = "nothing"
                else:
                    before_object = before_object.name
                after_object = after_state.players[self.player_idx].held_object
                if after_object is None:
                    after_object = "nothing"
                else:
                    after_object = after_object.name

                def item_is_on_counter(state, item_str):
                    item_on_counter = 0
                    for key, obj in state.objects.items():
                        if obj.name == item_str:
                            item_on_counter = 1
                    return item_on_counter

                onion_on_counter_before = item_is_on_counter(before_state, 'onion')
                onion_on_counter_after = item_is_on_counter(after_state, 'onion')
                soup_on_counter_before = item_is_on_counter(before_state, 'soup')
                soup_on_counter_after = item_is_on_counter(after_state, 'soup')
                dish_on_counter_before = item_is_on_counter(before_state, 'dish')
                dish_on_counter_after = item_is_on_counter(after_state, 'dish')
                tomato_on_counter_before = item_is_on_counter(before_state, 'tomato')
                tomato_on_counter_after = item_is_on_counter(after_state, 'tomato')

                def get_num_steps_to_loc(state, loc_name):

                    if loc_name == 'onion_dispenser':
                        obj_loc = mdp.get_onion_dispenser_locations()
                    elif loc_name == 'tomato_dispenser':
                        obj_loc = mdp.get_tomato_dispenser_locations()
                    elif loc_name == 'dish_dispenser':
                        obj_loc = mdp.get_dish_dispenser_locations()
                    elif loc_name == 'soup_pot':
                        potential_locs = mdp.get_pot_locations()
                        obj_loc = []
                        for pos in potential_locs:
                            if base_env.mdp.soup_ready_at_location(state, pos):
                                obj_loc.append(pos)
                    elif loc_name == 'serve':
                        obj_loc = mdp.get_serving_locations()
                    elif loc_name == 'pot':
                        obj_loc = mdp.get_pot_locations()
                    else:
                        raise 'Unknown location name'

                    pos_and_or = state.players[self.player_idx].pos_and_or
                    min_dist = np.Inf

                    for loc in obj_loc:
                        results = base_env.mlam.motion_planner.motion_goals_for_pos[loc]
                        for result in results:
                            if base_env.mlam.motion_planner.positions_are_connected(pos_and_or, result):
                                plan = base_env.mp._get_position_plan_from_graph(pos_and_or, result)
                                plan_results = base_env.mp.action_plan_from_positions(plan, pos_and_or, result)
                                curr_dist = len(plan_results[1])
                                if curr_dist < min_dist:
                                    min_dist = curr_dist
                    return min_dist

                n_steps_onion_dispenser_before = get_num_steps_to_loc(before_state, 'onion_dispenser')
                n_steps_tomato_dispenser_before = get_num_steps_to_loc(before_state, 'tomato_dispenser')
                n_steps_dish_dispenser_before = get_num_steps_to_loc(before_state, 'dish_dispenser')
                n_steps_soup_pot_before = get_num_steps_to_loc(before_state, 'soup_pot')
                n_steps_pot_before = get_num_steps_to_loc(before_state, 'pot')
                n_steps_serve_before = get_num_steps_to_loc(before_state, 'serve')

                if after_object == 'onion' and before_object == "nothing":
                    if n_steps_onion_dispenser_before == 1:
                        action_taken = "Picking Up Onion From Dispenser"
                    else:
                        action_taken = "Picking Up Onion From Counter"
                elif after_object == 'tomato' and before_object == "nothing":
                    if n_steps_tomato_dispenser_before == 1:
                        action_taken = "Picking Up Tomato From Dispenser"
                    else:
                        action_taken = "Picking Up Tomato From Counter"
                elif after_object == 'soup' and before_object == "dish":
                    if n_steps_soup_pot_before == 1:
                        action_taken = "Picking Up Soup From Pot"
                    else:
                        print('WARNING: Soup was picked up somehow even though we were not at the pot')
                        action_taken = "Picking Up Soup From Pot"
                elif after_object == 'dish' and before_object == "nothing":
                    if n_steps_dish_dispenser_before == 1:
                        action_taken = "Picking Up Dish From Dispenser"
                    else:
                        action_taken = "Picking Up Dish From Counter"
                elif after_object == 'nothing' and before_object == "onion":
                    if n_steps_pot_before == 1:
                        action_taken = "Bringing To Closest Pot"
                    else:
                        action_taken = "Placing On Closest Counter"
                elif after_object == 'nothing' and before_object == "tomato":
                    if n_steps_pot_before == 1:
                        action_taken = "Bringing To Closest Pot"
                    else:
                        action_taken = "Placing On Closest Counter"
                elif after_object == 'nothing' and before_object == "dish":
                    action_taken = "Placing On Closest Counter"
                elif after_object == 'nothing' and before_object == "soup":
                    if n_steps_serve_before == 1:
                        action_taken = "Serving At Dispensary"
                    else:
                        action_taken = 'Placing On Closest Counter'
                else:
                    # check if timer was put on
                    turned_on_timer = False
                    if n_steps_pot_before == 1:
                        pot_locs = mdp.get_pot_locations()

                        for pot_loc in pot_locs:
                            pos_and_or = before_state.players[self.player_idx].pos_and_or
                            min_dist = np.Inf
                            results = base_env.mlam.motion_planner.motion_goals_for_pos[pot_loc]
                            for result in results:
                                if base_env.mlam.motion_planner.positions_are_connected(pos_and_or, result):
                                    plan = base_env.mp._get_position_plan_from_graph(pos_and_or, result)
                                    plan_results = base_env.mp.action_plan_from_positions(plan, pos_and_or, result)
                                    curr_dist = len(plan_results[1])
                                    if curr_dist < min_dist:
                                        min_dist = curr_dist
                            if min_dist == 1:
                                if before_state.objects[pot_loc].is_cooking is False and after_state.objects[pot_loc].is_cooking is True:
                                    turned_on_timer = True
                    if turned_on_timer:
                        action_taken = "Turning On Cook Timer"
                    else:
                        action_taken = "Nothing"

                # high_level action
                episode_action_dict[i] = action_taken

            # go through a second time and pair each observation with action
            for timestep, (trajectories, trajectories_reduced) in enumerate(zip(trajectory_observations_raw[k],
                                                                                trajectory_observations_reduced[k])):
                try:
                    next_action = indices_array[indices_array > timestep].min()
                except:
                    # no next action
                    continue
                episode_observations.append(trajectories)
                episode_observations_reduced.append(trajectories_reduced)
                episode_primitive_actions.append(trajectory_actions[k][timestep])
                episode_high_level_actions.append(action_mapping[episode_action_dict[next_action]])
                episode_intents.append(intent_mapping[episode_action_dict[next_action]])

            total_observations_raw.extend(episode_observations)
            total_observations_reduced.extend(episode_observations_reduced)
            total_primitive_actions.extend(episode_primitive_actions)
            total_high_level_actions.extend(episode_high_level_actions)
            total_intents.extend(episode_intents)

        self.high_level_actions = total_high_level_actions
        self.intents = total_intents
        self.primitives = total_primitive_actions

        # aggregate every 2 states and 2 actions into a single vector, then use the high_level_action as the label
        X = []

        self.training_intent_features = []
        self.training_intents = []

        self.training_observations = []
        self.training_observations_reduced = []
        self.training_primitives = []
        self.training_actions = []
        for i in range(1, len(total_observations_raw)):
            features = [total_observations_raw[i-1], [total_primitive_actions[i-1]],
                        total_observations_raw[i]]
            features = np.concatenate(features, axis=0)
            self.training_intent_features.append(features)
            self.training_observations.append(total_observations_raw[i])
            self.training_observations_reduced.append(total_observations_reduced[i])
            self.training_actions.append(total_high_level_actions[i])
            self.training_primitives.append(total_primitive_actions[i])
            self.training_intents.append(total_intents[i])

        X = self.training_intent_features
        Y = self.training_intents

        # print distribution for self.training_intents and self.training_actions
        print("Distribution of intents: ", Counter(self.training_intents))
        print("Distribution of actions: ", Counter(self.training_actions))
        print("Distribution of primitives: ", Counter(self.training_primitives))

        assert len(X[0]) == 2 * len(total_observations_raw[0]) + 1
        assert len(X) == len(Y)

        # training
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2,
                                                                                random_state=42)

        self.model = DecisionTreeClassifier(max_depth=3, random_state=0)
        self.rf_model = DecisionTreeClassifier(max_depth=5, random_state=0)

        self.model.fit(self.X_train, self.y_train)
        # check validation accuracy
        print("Validation accuracy for intents model: ", self.model.score(self.X_test, self.y_test))

        self.rf_model.fit(self.X_train, self.y_train)
        print("Validation accuracy for intents model (more complex): ", self.rf_model.score(self.X_test, self.y_test))

        # train on all the data
        self.model = self.rf_model
        self.model.fit(X, Y)

    def predict(self, observation):
        _states = None
        return self.model.predict(observation.reshape(1, -1))[0], _states

def get_pretrained_intent_model(layout, intent_model_file=None):
    if intent_model_file is None:
        intent_model_file = os.path.join('data', 'intent_models', layout + '.pt')
    intent_model = torch.load(intent_model_file)
    # intent_model = AgentWrapper(intent_model)
    return intent_model

def get_intent_model_from_human_data(traj_directory, layout_name, bc_agent_idx):
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

    # we simply want to use the state -> observation fn from this env
    # env = OvercookedSelfPlayEnv(layout_name=layout_name, seed_num=0,
    #                             reduced_state_space_ego=False,
    #                             reduced_state_space_alt=False)
    # states = df['state']
    # for i in range(len(states)):
    #     state = json.loads(states[i])
    #     observations.append(env.featurize_fn(state))

    df = df[df['agent_idx'] == bc_agent_idx]
    assert len(raw_states) == len(df)

    observations = []
    for obs_str in df['obs'].values:
        obs_str = obs_str.replace('\n', '')
        observations.append(np.fromstring(obs_str[1:-1], dtype=float, sep=' '))

    actions = df['action'].values

    if len(observations) == 0:
        raise ValueError(f'No observations found for agent {bc_agent_idx} in layout {layout_name}')

    return IntentModel(layout_name, observations, actions, bc_agent_idx, raw_states, traj_lengths)



