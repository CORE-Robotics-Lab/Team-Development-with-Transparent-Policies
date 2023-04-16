import os
import sys
from collections import Counter

import pandas as pd
from algos.genetic_algorithm import GA_DT_Structure_Optimizer
from matplotlib import pyplot as plt
from models.decision_tree import decision_tree_to_ddt
from overcooked.overcooked_envs import OvercookedPlayWithFixedPartner

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    pass
else:
    pass
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.torch_layers import FlattenExtractor
from ipm.bin.utils import CheckpointCallbackWithRew
from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3 import PPO


class Classifier_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Classifier_MLP, self).__init__()
        self.h1 = nn.Linear(in_dim, hidden_dim)
        self.h2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.log_softmax(self.out(x), dim=0)
        return x


class RobotModel:
    def __init__(self, layout, idct_policy, human_policy, intent_model=None):

        self.robot_idct_policy = idct_policy
        self.robot_idct_policy.to('cpu')
        self.human_policy = human_policy
        self.layout = layout

        intent_input_size_dict = {'forced_coordination': 26,
                                  'two_rooms': 26,
                                  'tutorial': 26,
                                  'two_rooms_narrow': 30}
        self.intent_input_dim_size = intent_input_size_dict[layout]
        intent_output_size_dict = {'forced_coordination': 6,
                                   'two_rooms': 6,
                                   'tutorial': 6,
                                   'two_rooms_narrow': 7}
        self.intent_model = Classifier_MLP(in_dim=self.intent_input_dim_size, hidden_dim=64,
                                           out_dim=intent_output_size_dict[layout])

        if intent_model is not None:
            self.intent_model.load_state_dict(intent_model)

        self.player_idx = 0
        if not self.layout == "two_rooms_narrow":
            self.action_mapping = {
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
            }
            self.intent_mapping = {
                "Picking Up Onion From Dispenser": 0,  # picking up ingredient
                "Picking Up Onion From Counter": 0,  # picking up ingredient
                "Picking Up Dish From Dispenser": 1,  # picking up dish
                "Picking Up Dish From Counter": 1,  # picking up dish
                "Picking Up Soup From Counter": 2,  # picking up soup
                "Picking Up Soup From Pot": 2,  # picking up soup
                "Serving At Dispensary": 3,  # serving dish
                "Bringing To Closest Pot": 4,  # placing item down
                "Placing On Closest Counter": 4,  # placing item down
                "Nothing": 5,
            }
        else:
            self.action_mapping = {
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
                "Placing On Closest Counter": 11
            }
            self.intent_mapping = {
                "Picking Up Onion From Dispenser": 0,  # picking up ingredient
                "Picking Up Onion From Counter": 0,  # picking up ingredient
                "Picking Up Tomato From Dispenser": 1,  # picking up ingredient
                "Picking Up Tomato From Counter": 1,  # picking up ingredient
                "Picking Up Dish From Dispenser": 2,  # picking up dish
                "Picking Up Dish From Counter": 2,  # picking up dish
                "Picking Up Soup From Counter": 3,  # picking up soup
                "Picking Up Soup From Pot": 3,  # picking up soup
                "Serving At Dispensary": 4,  # serving dish
                "Bringing To Closest Pot": 5,  # placing item down
                "Placing On Closest Counter": 5,  # placing item down
                "Nothing": 6,
            }

        DEFAULT_ENV_PARAMS = {
            # add one because when we reset it takes up a timestep
            "horizon": 200,
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

        self.mdp = OvercookedGridworld.from_layout_name(layout_name=self.layout, rew_shaping_params=rew_shaping_params)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, **DEFAULT_ENV_PARAMS)

    def translate_recent_data_to_labels(self, recent_data_loc):
        """
        For now, assumes one trajectory
        Args:
            recent_data_loc:

        Returns:

        """
        import torch
        recent_data = torch.load(recent_data_loc)
        reduced_observations_human = recent_data['human_obs']
        reduced_observations_AI = recent_data['AI_obs']
        actions = recent_data['human_action']
        trajectory_states = recent_data['states']
        traj_lengths = len(trajectory_states)
        self.reduced_observations_human = reduced_observations_human
        self.reduced_observations_AI = reduced_observations_AI

        # for each trajectory in this data set
        for k in range(1):
            # go through and find all the indices where the action is 5
            indices = [i for i in range(len(actions)) if actions[i] == 5]

            if indices[-1] == traj_lengths - 1:
                # if last action is an interact, then there will be no next timestep.
                indices.remove(indices[-1])
            indices_array = np.array(indices)
            episode_observations = []
            episode_observations_reduced = []
            episode_observations_reduced_no_intent = []
            episode_high_level_actions = []
            episode_intents = []
            episode_primitive_actions = []
            episode_action_dict = {
            }
            for e, i in enumerate(indices):
                before_state = trajectory_states[i]
                after_state = trajectory_states[i + 1]

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
                        obj_loc = self.mdp.get_onion_dispenser_locations()
                    elif loc_name == 'tomato_dispenser':
                        obj_loc = self.mdp.get_tomato_dispenser_locations()
                    elif loc_name == 'dish_dispenser':
                        obj_loc = self.mdp.get_dish_dispenser_locations()
                    elif loc_name == 'soup_pot':
                        potential_locs = self.mdp.get_pot_locations()
                        obj_loc = []
                        for pos in potential_locs:
                            if self.base_env.mdp.soup_ready_at_location(state, pos):
                                obj_loc.append(pos)
                    elif loc_name == 'serve':
                        obj_loc = self.mdp.get_serving_locations()
                    elif loc_name == 'pot':
                        obj_loc = self.mdp.get_pot_locations()
                    else:
                        raise 'Unknown location name'

                    pos_and_or = state.players[self.player_idx].pos_and_or
                    min_dist = np.Inf

                    for loc in obj_loc:
                        results = self.base_env.mlam.motion_planner.motion_goals_for_pos[loc]
                        for result in results:
                            if self.base_env.mlam.motion_planner.positions_are_connected(pos_and_or, result):
                                plan = self.base_env.mp._get_position_plan_from_graph(pos_and_or, result)
                                plan_results = self.base_env.mp.action_plan_from_positions(plan, pos_and_or, result)
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
                        pot_locs = self.mdp.get_pot_locations()

                        for pot_loc in pot_locs:
                            pos_and_or = before_state.players[self.player_idx].pos_and_or
                            min_dist = np.Inf
                            results = self.base_env.mlam.motion_planner.motion_goals_for_pos[pot_loc]
                            for result in results:
                                if self.base_env.mlam.motion_planner.positions_are_connected(pos_and_or, result):
                                    plan = self.base_env.mp._get_position_plan_from_graph(pos_and_or, result)
                                    plan_results = self.base_env.mp.action_plan_from_positions(plan, pos_and_or, result)
                                    curr_dist = len(plan_results[1])
                                    if curr_dist < min_dist:
                                        min_dist = curr_dist
                            if min_dist == 1:
                                if before_state.objects[pot_loc].is_cooking is False and after_state.objects[
                                    pot_loc].is_cooking is True:
                                    turned_on_timer = True
                    if turned_on_timer:
                        action_taken = "Turning On Cook Timer"
                    else:
                        action_taken = "Nothing"

                # high_level action
                episode_action_dict[i] = action_taken

            # go through a second time and pair each observation with action
            for timestep in range(len(trajectory_states)):
                try:
                    next_action = indices_array[indices_array > timestep].min()
                except:
                    # no next action
                    continue
                # episode_observations.append(reduced_observations[timestep])
                episode_observations_reduced.append(
                    [reduced_observations_human[timestep], reduced_observations_AI[timestep]])
                episode_observations_reduced_no_intent.append(
                    [reduced_observations_human[timestep][:int(self.intent_input_dim_size / 2)],
                     reduced_observations_AI[timestep][:int(self.intent_input_dim_size / 2)]])
                episode_primitive_actions.append(actions[timestep])
                episode_high_level_actions.append(self.action_mapping[episode_action_dict[next_action]])
                # TODO: check the mapping below.
                episode_intents.append(self.intent_mapping[episode_action_dict[next_action]])

        self.episode_high_level_actions = episode_high_level_actions
        self.episode_intents = episode_intents
        self.episode_primitive_actions = episode_primitive_actions
        self.episode_observations_reduced = episode_observations_reduced
        self.episode_observations_reduced_no_intent = episode_observations_reduced_no_intent

        # print distribution for self.training_intents and self.training_actions
        print("Distribution of intents: ", Counter(self.episode_intents))
        print("Distribution of actions: ", Counter(self.episode_high_level_actions))
        print("Distribution of primitives: ", Counter(self.episode_primitive_actions))

    def finetune_intent_model(self):
        """
        Function assumes you just translated recent data
        Returns:

        """
        X = []
        Y = []
        # reformat data
        for i in range(len(self.episode_observations_reduced_no_intent)):
            X.append(np.array(self.episode_observations_reduced_no_intent[i]).flatten())
            Y.append(self.episode_intents[i])
        # put data on device
        X = torch.tensor(X).float()
        Y = torch.tensor(Y).long()

        optimizer = torch.optim.Adam(self.intent_model.parameters(), lr=1e-3)
        n_epochs = 30
        batch_size = 32
        n_batches = int(np.ceil(len(X) / batch_size))
        for epoch in range(n_epochs):
            epoch_loss = 0
            for i in range(n_batches):
                optimizer.zero_grad()
                batch_X = X[i * batch_size:(i + 1) * batch_size]
                batch_Y = Y[i * batch_size:(i + 1) * batch_size]
                pred = self.intent_model(batch_X)
                loss = F.cross_entropy(pred.reshape(-1, 6), batch_Y)

                # logits = idct_ppo_policy(batch_X)
                # loss = criterion(logits, batch_Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch} loss: {epoch_loss / n_batches}")

    def finetune_robot_idct_policy(self, n_steps=20000, recent_data_file='data/11_trajs_tar'):

        checkpoint_freq = n_steps // 100
        save_models = True
        ego_idx = 1  # robot policy is always the second player

        seed = 0

        env = OvercookedPlayWithFixedPartner(partner=self.human_policy,
                                             layout_name=self.layout,
                                             seed_num=seed,
                                             behavioral_model=self.intent_model,
                                             ego_idx=ego_idx,
                                             reduced_state_space_ego=True,
                                             reduced_state_space_alt=True,
                                             use_skills_ego=True,
                                             use_skills_alt=True,
                                             use_true_intent_ego=True,
                                             use_true_intent_alt=True)

        initial_model_path = os.path.join('data', self.layout, 'robot_online_optimization', 'initial_model.zip')
        medium_model_path = os.path.join('data', self.layout, 'robot_online_optimization', 'medium_model.zip')
        final_model_path = os.path.join('data', self.layout, 'robot_online_optimization', 'final_model.zip')

        save_dir = os.path.join('data', self.layout, 'robot_online_optimization', 'robot_idct_policy')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        checkpoint_callback = CheckpointCallbackWithRew(
            n_steps=n_steps,
            save_freq=checkpoint_freq,
            save_path=save_dir,
            name_prefix="rl_model",
            save_replay_buffer=True,
            initial_model_path=initial_model_path,
            medium_model_path=medium_model_path,
            final_model_path=final_model_path,
            save_model=save_models,
            verbose=1
        )

        full_save_dir = "./" + save_dir + "/"
        if not os.path.exists(full_save_dir):
            os.makedirs(full_save_dir)

        env = Monitor(env, full_save_dir)

        input_dim = get_obs_shape(env.observation_space)[0]
        output_dim = env.n_actions_ego
        initial_depth = 1
        max_depth = 1
        num_gens = 100
        seed = 1

        ga = GA_DT_Structure_Optimizer(trajectories_file=recent_data_file,
                                       initial_depth=initial_depth,
                                       max_depth=max_depth,
                                       n_vars=input_dim,
                                       n_actions=output_dim,
                                       num_gens=num_gens,
                                       seed=seed)
        ga.run()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # final_tree = ga.final_trees[0]
        final_tree = ga.best_tree
        model = decision_tree_to_ddt(tree=final_tree,
                                     input_dim=input_dim,
                                     output_dim=output_dim,
                                     device=device)

        ppo_lr = 0.0003
        ppo_batch_size = 64
        ppo_n_steps = 10000

        ddt_kwargs = {
            'num_leaves': len(model.leaf_init_information),
            'hard_node': False,
            'weights': model.layers,
            'alpha': 1.0,
            'comparators': model.comparators,
            'leaves': model.leaf_init_information,
            'fixed_idct': False,
            'device': 'cuda',
            'argmax_tau': 1.0,
            'ddt_lr': 0.001,  # this param is irrelevant for the IDCT
            'use_individual_alpha': True,
            'l1_reg_coeff': 1.0,
            'l1_reg_bias': 1.0,
            'l1_hard_attn': 1.0,
            'use_gumbel_softmax': False,
            'alg_type': 'ppo'
        }

        features_extractor = FlattenExtractor
        policy_kwargs = dict(features_extractor_class=features_extractor, ddt_kwargs=ddt_kwargs)

        agent = PPO("DDT_PPOPolicy", env,
                    n_steps=ppo_n_steps,
                    # batch_size=args.batch_size,
                    # buffer_size=args.buffer_size,
                    learning_rate=ppo_lr,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log='log',
                    gamma=0.99,
                    verbose=1,
                    # seed=1
                    )

        print(f'Agent training...')
        agent.learn(total_timesteps=n_steps, callback=checkpoint_callback)
        print(f'Finished training agent with best average reward of {checkpoint_callback.best_mean_reward}')

    def finetune_robot_idct_policy_legacy(self):
        """
        Function assumes you just translated recent data
        Returns:

        """
        X = []
        Y = []
        # reformat data
        for i in range(len(self.episode_observations_reduced_no_intent)):
            # get AI agent state
            if i == 0:
                continue
            X.append(np.array(self.reduced_observations_human[i]).flatten())
            Y.append(self.episode_high_level_actions[i])

        # put data on device
        X = torch.tensor(X).float()
        Y = torch.tensor(Y).long()

        optimizer = torch.optim.Adam(self.robot_idct_policy.parameters(), lr=1e-3)
        n_epochs = 30
        batch_size = 32
        n_batches = int(np.ceil(len(X) / batch_size))
        for epoch in range(n_epochs):
            epoch_loss = 0
            for i in range(n_batches):
                optimizer.zero_grad()
                batch_X = X[i * batch_size:(i + 1) * batch_size]
                batch_Y = Y[i * batch_size:(i + 1) * batch_size]
                pred = self.robot_idct_policy.forward_alt(batch_X)
                # pred here is log probs
                loss = F.cross_entropy(pred.reshape(-1, 10), batch_Y)

                # logits = idct_ppo_policy(batch_X)
                # loss = criterion(logits, batch_Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch} loss: {epoch_loss / n_batches}")
