from abc import abstractmethod, ABC

import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from typing import List, Tuple, Dict, Optional


class OvercookedMultiAgentEnv(gym.Env, ABC):
    def __init__(self, layout_name, ego_idx=None,
                 reduced_state_space_ego=False, use_skills_ego=True,
                 reduced_state_space_alt=False, use_skills_alt=True,
                 seed_num=None, n_timesteps=200,
                 behavioral_model=None, failed_skill_rew = -0.01,
                 use_true_intent=False,
                 double_cook_times=False):
        """
        base_env: OvercookedEnv
        """
        super(OvercookedMultiAgentEnv, self).__init__()

        self._obs: Tuple[Optional[np.ndarray], ...] = tuple()
        self._old_ego_obs: Optional[np.ndarray] = None

        self.initial_ego_idx: int = ego_idx
        self.initialize_agent_indices()

        self.failed_skill_rew = failed_skill_rew

        self.alt_red_obs = None
        self.behavioral_model = behavioral_model
        self.use_true_intent = use_true_intent

        self.cook_time_threshold = 5

        double_cook_times = 'demonstrations' in layout_name
        if double_cook_times:
            # self.base_env.mdp.cook_time = 2 * self.base_env.mdp.cook_time not what we want, but might be useful
            self.cook_time_threshold = 2 * self.cook_time_threshold

        self.layout_name: str = layout_name
        self.n_timesteps = n_timesteps
        self.set_env()

        if seed_num is not None:
            np.random.seed(seed_num)

        self.timestep = 0
        self.reduced_state_space_ego: bool = reduced_state_space_ego
        self.use_skills_ego: bool = use_skills_ego
        self.reduced_state_space_alt: bool = reduced_state_space_alt
        self.use_skills_alt: bool = use_skills_alt
        self.observation_space = self._setup_observation_space()
        self.n_primitive_actions = len(Action.ALL_ACTIONS)

        self.ego_currently_performing_skill = False
        self.ego_current_skill_type = None
        self.alt_currently_performing_skill = False
        self.alt_current_skill_type = None
        self.prev_raw_obs = None

        self.carry_out_skills = False

        if self.use_skills_ego:
            # include skills
            self.idx_to_skill_ego = [
                                     self.stand_still,
                                     self.get_onion_from_dispenser, self.pickup_onion_from_counter,
            self.get_dish_from_dispenser, self.pickup_dish_from_counter,
                                      self.get_soup_from_pot, self.pickup_soup_from_counter,
                                      self.serve_at_dispensary,
                                      self.bring_to_closest_pot, self.place_on_closest_counter,
                                      ]

            if layout_name == 'two_rooms_narrow':
                self.idx_to_skill_ego += [self.get_tomato_from_dispenser, self.pickup_tomato_from_counter]

        else:
            # otherwise, only include primitive actions
            self.idx_to_skill_ego = [self.move_up, self.move_down,
                                 self.move_right, self.move_left,
                                 self.stand_still, self.interact]

        if layout_name == 'two_rooms_narrow':
            self.macro_to_intent = {0:6,
                                    1:0, 2:0,
                                    3:1, 4:1,
                                    5:2, 6:2,
                                    7:3, 8:3,
                                    9:4, 10:5,
                                    11:5, 12:6}
        else:
            self.macro_to_intent = {0:5,
                                    1:0, 2:0,
                                    3:1, 4:1,
                                    5:2, 6:2,
                                    7:3, 8:4,
                                    9:4, 10:5}

        self.n_actions_ego = len(self.idx_to_skill_ego)
        if not self.use_skills_ego:
            assert self.n_actions_ego == self.n_primitive_actions

        if self.use_skills_alt:
            # include skills
            # move up
            # move down
            # move right
            # move left
            # stand still
            # interact
            # get onion from dispenser
            # pickup onion from counter
            # get dish from dispenser
            # pickup dish from counter
            # get soup from pot
            # pickup soup from counter
            # serve at dispensary
            # bring to closest pot
            # place on closest counter
            # self.idx_to_skill_alt = [
            #                          # self.move_up, self.move_down,
            #                          # self.move_right, self.move_left,
            #                          # self.stand_still, self.interact,
            #                          self.stand_still,
            #                          self.get_onion_from_dispenser, self.pickup_onion_from_counter]
            # if 'two_rooms_narrow' in self.layout_name:
            #     self.idx_to_skill_alt += [self.get_tomato_from_dispenser, self.pickup_tomato_from_counter]
            # self.idx_to_skill_alt += [self.get_dish_from_dispenser, self.pickup_dish_from_counter,
            #                           self.get_soup_from_pot, self.pickup_soup_from_counter,
            #                           self.serve_at_dispensary,
            #                           self.bring_to_closest_pot, self.place_on_closest_counter,
            #                           self.turn_on_cook_timer,
            #                           # self.random_action
            #                           ]
            self.idx_to_skill_alt = [
                self.stand_still,
                self.get_onion_from_dispenser, self.pickup_onion_from_counter,
                self.get_dish_from_dispenser, self.pickup_dish_from_counter,
                self.get_soup_from_pot, self.pickup_soup_from_counter,
                self.serve_at_dispensary,
                self.bring_to_closest_pot, self.place_on_closest_counter,
                ]

            if layout_name == 'two_rooms_narrow':
                self.idx_to_skill_alt += [self.get_tomato_from_dispenser, self.pickup_tomato_from_counter]
        else:
            # otherwise, only include primitive actions
            self.idx_to_skill_alt = [self.move_up, self.move_down,
                                 self.move_right, self.move_left,
                                 self.stand_still, self.interact]


        self.idx_to_skill_strings = [
                                     ['stand_still'],
                                     ['get_onion_from_dispenser'], ['pickup_onion_from_counter'],
            ['get_dish_from_dispenser'], ['pickup_dish_from_counter'],
                                      ['get_soup_from_pot'], ['pickup_soup_from_counter'],
                                      ['serve_at_dispensary'],
                                      ['bring_to_closest_pot'], ['place_on_closest_counter'],
                                      ]
        if layout_name == 'two_rooms_narrow':
            self.idx_to_skill_strings+= [['get_tomato_from_dispenser'], ['pickup_tomato_from_counter']]
        self.n_actions_alt = len(self.idx_to_skill_alt)
        if not self.use_skills_alt:
            assert self.n_actions_alt == self.n_primitive_actions

        self.action_space  = gym.spaces.Discrete(self.n_actions_ego)
        self.check_conditions()

    def get_onion_from_dispenser(self, agent_idx):
        return self.perform_skill(agent_idx, 'get_onion_dispenser')

    def get_tomato_from_dispenser(self, agent_idx):
        return self.perform_skill(agent_idx, 'get_tomato_dispenser')

    def get_dish_from_dispenser(self, agent_idx):
        return self.perform_skill(agent_idx, 'get_dish_dispenser')

    def get_soup_from_pot(self, agent_idx):
        return self.perform_skill(agent_idx, 'get_soup_pot')

    def pickup_onion_from_counter(self, agent_idx):
        return self.perform_skill(agent_idx, 'pickup_onion_counter')

    def pickup_tomato_from_counter(self, agent_idx):
        return self.perform_skill(agent_idx, 'pickup_tomato_counter')

    def pickup_dish_from_counter(self, agent_idx):
        return self.perform_skill(agent_idx, 'pickup_dish_counter')

    def pickup_soup_from_counter(self, agent_idx):
        return self.perform_skill(agent_idx, 'pickup_soup_counter')

    def serve_at_dispensary(self, agent_idx):
        return self.perform_skill(agent_idx, 'serve')

    def bring_to_closest_pot(self, agent_idx):
        return self.perform_skill(agent_idx, 'place_in_pot')

    def place_on_closest_counter(self, agent_idx):
        return self.perform_skill(agent_idx, 'place_on_closest_counter')

    def turn_on_cook_timer(self, agent_idx):
        return self.perform_skill(agent_idx, 'turn_on_cook_timer')

    def move_up(self, agent_idx):
        return (0, -1), 0

    def move_down(self, agent_idx):
        return (0, 1), 0

    def move_right(self, agent_idx):
        return (1, 0), 0

    def move_left(self, agent_idx):
        return (-1, 0), 0

    def stand_still(self, agent_idx):
        return (0, 0), 0

    def interact(self, agent_idx):
        return 'interact', 0

    def random_action(self, agent_idx):
        n_actions = self.n_actions_ego if agent_idx == 0 else self.n_actions_alt
        n_actions -= 1
        return self.idx_to_skill_ego[np.random.randint(n_actions)](agent_idx)

    def perform_skill(self, agent_idx, skill_type='onion') -> Tuple[Tuple[int, int], float]:

        stand_still = (0, 0)
        failed_skill_rew = self.failed_skill_rew

        state = self.base_env.state
        held_item = self.base_env.state.players[agent_idx].held_object

        # if held_item is not None:
        #     possible_items = ['onion', 'tomato', 'dish', 'soup']
        #     assert held_item.name in possible_items, 'held item is not in possible items'

        ignore_closest_pot, ignore_furthest_pot = False, False

        if 'get' in skill_type:
            if skill_type == 'get_onion_dispenser':
                if held_item is not None:
                    return stand_still, failed_skill_rew
                obj_loc = self.mdp.get_onion_dispenser_locations()
            elif skill_type == 'get_tomato_dispenser':
                if held_item is not None:
                    return stand_still, failed_skill_rew
                obj_loc = self.mdp.get_tomato_dispenser_locations()
            elif skill_type == 'get_dish_dispenser':
                if held_item is not None:
                    return stand_still, failed_skill_rew
                obj_loc = self.mdp.get_dish_dispenser_locations()
            elif skill_type == 'get_soup_pot':
                if held_item is None or held_item.name != 'dish':
                    return stand_still, failed_skill_rew
                potential_locs = self.mdp.get_pot_locations()
                obj_loc = []
                for pos in potential_locs:
                    if self.base_env.mdp.soup_ready_at_location(state, pos):
                        obj_loc.append(pos)
            else:
                raise ValueError('Unknown get type')
        elif 'pickup' in skill_type:
            if held_item is not None:
                return stand_still, failed_skill_rew

            obj_loc = []
            if skill_type == 'pickup_onion_counter':
                counter_objects = list(self.mdp.get_counter_objects_dict(state)['onion'])
            elif skill_type == 'pickup_tomato_counter':
                counter_objects = list(self.mdp.get_counter_objects_dict(state)['tomato'])
            elif skill_type == 'pickup_dish_counter':
                counter_objects = list(self.mdp.get_counter_objects_dict(state)['dish'])
            elif skill_type == 'pickup_soup_counter':
                counter_objects = list(self.mdp.get_counter_objects_dict(state)['soup'])
            else:
                raise ValueError('Unknown item on counter')
            if len(counter_objects) > 0:
                obj_loc.extend(counter_objects)
            else:
                return stand_still, failed_skill_rew
                # obj_loc = self.mdp.get_empty_counter_locations(state)
                # skill_type = 'place_on_closest_counter' # hacky way to get what we want here
        elif skill_type == 'serve':
            if held_item is None or held_item.name != 'soup':
                return stand_still, failed_skill_rew
            obj_loc = self.mdp.get_serving_locations()
        elif skill_type == 'place_in_pot':
            if held_item is None or held_item.name == 'soup' or held_item.name == 'dish':
                return stand_still, failed_skill_rew

            obj_loc = self.mdp.get_pot_locations()
            # check if closest pot is full
            obs = self.raw_obs[agent_idx]
            if (3 == obs[27] + obs[28]) or obs[26] == 1:
                # then ignore the closest counter
                ignore_closest_pot = True
            if (3 == obs[37] + obs[38]) or obs[36] == 1:
                # then ignore the furthest counter
                ignore_furthest_pot = True
            if ignore_closest_pot is True and ignore_furthest_pot is True:
                return stand_still, failed_skill_rew
        elif skill_type == 'place_on_closest_counter':
            if held_item is None:
                return stand_still, failed_skill_rew
            obj_loc = self.mdp.get_empty_counter_locations(state)
        elif skill_type == 'turn_on_cook_timer':
            if held_item is not None:
                return stand_still, failed_skill_rew
            obj_loc = self.mdp.get_pot_locations()

            # check if closest pot is empty or already cooking
            obs = self.raw_obs[agent_idx]
            if (0 == obs[27] + obs[28]) or obs[26] == 1:
                # then ignore the closest counter
                ignore_closest_pot = True
            if (0 == obs[37] + obs[38]) or obs[36] == 1:
                # then ignore the furthest counter
                ignore_furthest_pot = True
            if ignore_closest_pot is True and ignore_furthest_pot is True:
                return stand_still, failed_skill_rew

        else:
            raise ValueError('Unknown skill type')

        pos_and_or = state.players[agent_idx].pos_and_or

        min_dist = np.Inf
        goto_pos_and_or = None
        success_skill_rew = 0

        if skill_type == 'serve' or skill_type == 'place_in_pot' or skill_type == 'turn_on_cook_timer':
            _, closest_obj_loc = self.base_env.mp.min_cost_to_feature(pos_and_or, obj_loc, with_argmin=True)
            if closest_obj_loc is None:
                # stand still because we can't do anything
                return stand_still, failed_skill_rew
            else:
                if (skill_type == 'place_in_pot' or skill_type == 'turn_on_cook_timer') and ignore_closest_pot:
                    obj_loc.remove(closest_obj_loc)
                if (skill_type == 'place_in_pot' or skill_type == 'turn_on_cook_timer') and ignore_furthest_pot:
                    for item in obj_loc:
                        if item is not closest_obj_loc:
                            obj_loc.remove(item)
                _, closest_obj_loc = self.base_env.mp.min_cost_to_feature(pos_and_or, obj_loc, with_argmin=True)
                if closest_obj_loc is None:
                    # stand still because we can't do anything
                    return stand_still, failed_skill_rew
                goto_pos_and_or = self.base_env.mlam._get_ml_actions_for_positions([closest_obj_loc])[0]
        else:
            if obj_loc is None:
                return stand_still, failed_skill_rew
            if skill_type == 'place_on_closest_counter':
                # filter, only items of length 2
                obj_loc_old = obj_loc
                obj_loc = []
                for loc in obj_loc_old:
                    results = self.base_env.mlam.motion_planner.motion_goals_for_pos[loc]
                    if len(results) == 2:
                        obj_loc.append(loc) # reachable for both players
            for loc in obj_loc:
                results = self.base_env.mlam.motion_planner.motion_goals_for_pos[loc]
                for result in results:
                    if self.base_env.mlam.motion_planner.positions_are_connected(pos_and_or, result):
                        plan = self.base_env.mp._get_position_plan_from_graph(pos_and_or, result)
                        plan_results = self.base_env.mp.action_plan_from_positions(plan, pos_and_or, result)
                        curr_dist = len(plan_results[1])
                        if curr_dist < min_dist:
                            goto_pos_and_or = result
                            min_dist = curr_dist
            if goto_pos_and_or is None: # if we found nothing
                return stand_still, failed_skill_rew

        plan = self.base_env.mp._get_position_plan_from_graph(pos_and_or, goto_pos_and_or)
        actions, _, _ = self.base_env.mp.action_plan_from_positions(plan, pos_and_or, goto_pos_and_or)
        if self.carry_out_skills and self.current_ego_idx == agent_idx and len(actions) > 1:
            self.ego_currently_performing_skill = True
            self.ego_current_skill_type = skill_type
            self.ego_current_action_seq = actions[1:]
        elif self.carry_out_skills and self.current_alt_idx == agent_idx and len(actions) > 1:
            self.alt_currently_performing_skill = True
            self.alt_current_skill_type = skill_type
            self.alt_current_action_seq = actions[1:]
        return actions[0], success_skill_rew

    def initialize_agent_indices(self):
        if self.initial_ego_idx is None:
            ego_idx = np.random.randint(2)
        else:
            ego_idx = self.initial_ego_idx
        self.current_ego_idx = ego_idx
        self.current_alt_idx = (ego_idx + 1) % 2

    def set_env(self,
                placing_in_pot_multiplier=1,
                dish_pickup_multiplier=1,
                soup_pickup_multiplier=1,
                ):
        DEFAULT_ENV_PARAMS = {
            # add one because when we reset it takes up a timestep
            "horizon": self.n_timesteps + 1,
            "info_level": 0,
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": placing_in_pot_multiplier * 3,
            "DISH_PICKUP_REWARD": dish_pickup_multiplier * 3,
            "SOUP_PICKUP_REWARD": soup_pickup_multiplier * 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        self.mdp = OvercookedGridworld.from_layout_name(layout_name=self.layout_name, rew_shaping_params=rew_shaping_params)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, **DEFAULT_ENV_PARAMS)
        self.featurize_fn = self.base_env.featurize_state_mdp
        self.reduced_featurize_fn = self.base_env.featurize_state_mdp_reduced

    @abstractmethod
    def check_conditions(self):
        pass

    @abstractmethod
    def get_teammate_action(self):
        pass

    @abstractmethod
    def update_ego(self):
        pass

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        # below is original obs shape
        self.raw_obs = self.featurize_fn(dummy_state)
        obs = self.raw_obs[self.current_ego_idx]
        # reduced_obs = self.get_reduced_obs(obs, is_ego=True)
        reduced_obs = self.reduced_featurize_fn(dummy_state)[self.current_ego_idx]
        obs_shape = reduced_obs.shape
        self.n_reduced_feats = obs_shape[0]
        # if below fails, we need to update shape of alt_red_obs
        high = np.ones(obs_shape, dtype=np.float32) * np.inf  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)
        return gym.spaces.Box(-high, high, dtype=np.float64)

    def add_intent(self, obs, agent_idx):
        assert self.reduced_state_space_ego is True, "You are trying to use raw observation space for ego but have an intent model"
        new_features = np.zeros(5)
        if self.layout_name == 'two_rooms_narrow':
            new_features = np.zeros(6)

        if self.timestep <= 1:
            return np.concatenate([obs, new_features])


        other_idx = self.current_alt_idx if agent_idx == self.current_ego_idx else self.current_ego_idx
        is_ego = agent_idx == self.current_ego_idx


        if self.use_true_intent:
            error_msg = "You are trying to get the true intent of an agent but cannot infer without macro-actions"
            if is_ego:
                assert self.use_skills_alt is True, error_msg
            else:
                assert self.use_skills_ego is True, error_msg
            # get the other agents previous macro-action then convert it into an intent
            other_action = self.prev_macro_action[other_idx]
            intent = self.macro_to_intent[other_action]
        else:
            other_obs = self.get_reduced_obs(self.raw_obs[other_idx], is_ego=is_ego, include_intent=False)
            features = np.concatenate([obs, other_obs])
            intent = self.behavioral_model.predict(features)[0]
        if intent < len(new_features):
            new_features[intent] = 1
        obs = np.concatenate([obs, new_features])
        return obs

    def get_reduced_obs(self, obs, is_ego, include_intent=True):
        reduced_obs = (is_ego and self.reduced_state_space_ego) or (not is_ego and self.reduced_state_space_alt)

        if not reduced_obs:
            assert obs.shape[0] > 22
            if is_ego and self.behavioral_model is not None:
                if include_intent:
                    return self.add_intent(obs, is_ego)
            return obs

        # assumes 2 pots!
        assert obs.shape[0] == 96

        reduced_obs = []
        # first four features (direction facing)
        # reduced_obs.append(obs[0])
        # reduced_obs.append(obs[1])
        # reduced_obs.append(obs[2])
        # reduced_obs.append(obs[3])

        # next four features (held items)
        reduced_obs.append(obs[4])
        reduced_obs.append(obs[5])
        reduced_obs.append(obs[6])
        if 'two_rooms_narrow' in self.layout_name:
            reduced_obs.append(obs[7])

        # other agent facing direction
        # reduced_obs.append(obs[46])
        # reduced_obs.append(obs[47])
        # reduced_obs.append(obs[48])
        # reduced_obs.append(obs[49])

        # other player holding onion, soup, dish, or tomato
        reduced_obs.append(obs[50])
        reduced_obs.append(obs[51])
        reduced_obs.append(obs[52])
        if 'two_rooms_narrow' in self.layout_name:
            reduced_obs.append(obs[53])

        # # closest soup # onions and # tomatoes
        # reduced_obs.append(obs[16])
        # reduced_obs.append(obs[17])

        # Note: these next 2 features require that the ego plays first (blue)

        onion_on_counter = 0
        for key, obj in self.base_env.state.objects.items():
            if obj.name == 'onion':
                onion_on_counter = 1
        reduced_obs.append(onion_on_counter)

        if 'two_rooms_narrow' in self.layout_name:
            tomato_on_counter = 0
            for key, obj in self.base_env.state.objects.items():
                if obj.name == 'tomato':
                    tomato_on_counter = 1
            reduced_obs.append(tomato_on_counter)

        either_pot_needs_ingredients = 0
        pot_states = self.base_env.mdp.get_pot_states(self.base_env.state)
        if len(pot_states['empty']) >= 0:
            either_pot_needs_ingredients = 1
        elif len(pot_states['partially_full']) >= 0:
            either_pot_needs_ingredients = 1
        reduced_obs.append(either_pot_needs_ingredients)

        pot_ready = 0
        if len(pot_states['onion']) > 0 and len(pot_states['onion']['ready']) > 0:
            pot_ready = 1
        if self.layout_name == 'two_rooms_narrow':
            if len(pot_states['tomato']) > 0 and len(pot_states['tomato']['ready']) > 0:
                pot_ready = 1
        reduced_obs.append(pot_ready)

        dish_on_counter = 0
        for key, obj in self.base_env.state.objects.items():
            if obj.name == 'dish':
                dish_on_counter = 1
        reduced_obs.append(dish_on_counter)

        soup_on_counter = 0
        for key, obj in self.base_env.state.objects.items():
            if obj.name == 'soup':
                soup_on_counter = 1
        reduced_obs.append(soup_on_counter)

        reduced_obs = np.array(reduced_obs)

        # if is_ego and self.behavioral_model is not None and include_intent:
        return self.add_intent(reduced_obs, agent_idx=self.current_ego_idx if is_ego else self.current_alt_idx)
        return reduced_obs

    def getDummyEnv(self, player_num: int):
        """
        Returns a dummy environment with just an observation and action
        space that a partner agent can use to construct their policy network.

        :param player_num: the partner number to query
        """
        return self

    def multi_step(self, current_player_action):
        """
        action: agent for the ego agent

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        alt_macro_action = 0 # we assign this later
        ego_macro_action = current_player_action

        if not self.ego_currently_performing_skill:
            ego_action, skill_rew_ego = self.idx_to_skill_ego[current_player_action](agent_idx=self.current_ego_idx)
        else:
            ego_action = self.ego_current_action_seq.pop(0)
            skill_rew_ego = 0
            if len(self.ego_current_action_seq) == 0:
                self.ego_currently_performing_skill = False
        if not self.alt_currently_performing_skill:
            alt_macro_action = self.get_teammate_action()
            print('Time:', self.base_env.state.timestep)
            print('alt macro action', self.idx_to_skill_strings[alt_macro_action])
            print(self.base_env)
            alt_action, skill_rew_alt = self.idx_to_skill_alt[alt_macro_action](agent_idx=self.current_alt_idx)
        else:
            alt_action = self.alt_current_action_seq.pop(0)
            skill_rew_alt = 0
            if len(self.alt_current_action_seq) == 0:
                self.alt_currently_performing_skill = False

        self.previous_ego_action = ego_action

        if self.current_ego_idx == 0:
            joint_action = (ego_action, alt_action)
            self.prev_macro_action = (ego_macro_action, alt_macro_action)
        else:
            joint_action = (alt_action, ego_action)
            self.prev_macro_action = (alt_macro_action, ego_macro_action)

        self.prev_action = joint_action

        next_state, reward, done, info = self.base_env.step(joint_action)
        self.state = next_state

        # reward shaping
        reward_ego = reward + info['shaped_r_by_agent'][self.current_ego_idx] + skill_rew_ego
        reward_alt = reward + info['shaped_r_by_agent'][self.current_alt_idx] + skill_rew_alt

        (obs_p0, obs_p1) = self.featurize_fn(next_state)
        self.prev_raw_obs = self.raw_obs
        self.raw_obs = (obs_p0, obs_p1)
        self.ego_raw_obs = obs_p0 if self.current_ego_idx == 0 else obs_p1
        self.alt_raw_obs = obs_p1 if self.current_ego_idx == 0 else obs_p0
        self.ego_raw_act = ego_action
        self.alt_raw_act = alt_action
        # obs_p0 = self.get_reduced_obs(obs_p0, is_ego=self.current_ego_idx == 0)
        # obs_p1 = self.get_reduced_obs(obs_p1, is_ego=self.current_ego_idx == 1)
        obs_p0, obs_p1 = self.reduced_featurize_fn(self.base_env.state)


        self.alt_red_obs = obs_p1 if self.current_ego_idx == 0 else obs_p0

        return (obs_p0, obs_p1), (reward_ego, reward_alt), done, {}

    def n_step(
                    self,
                    action: int,
                ) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[np.ndarray], ...],
                           Tuple[float, ...],
                           bool,
                           Dict]:
        step_results = self.multi_step(action)
        self.update_ego()
        return step_results

    def step(
                self,
                action: int
            ) -> Tuple[Optional[np.ndarray], float, bool, Dict]:
        """
        Run one timestep from the perspective of the ego-agent. This involves
        calling the ego_step function and the alt_step function to get to the
        next observation of the ego agent.

        Accepts the ego-agent's action and returns a tuple of (observation,
        reward, done, info) from the perspective of the ego agent.

        :param action: An action provided by the ego-agent.

        :returns:
            observation: Ego-agent's next observation
            reward: Amount of reward returned after previous action
            done: Whether the episode has ended (need to call reset() if True)
            info: Extra information about the environment
        """
        self._obs, (self.ego_rew, self.alt_rew), done, info  = self.n_step(action)
        self.joint_reward = self.ego_rew + self.alt_rew

        if done:
            return self._old_ego_obs, self.ego_rew, done, info

        self._old_ego_obs = self.ego_obs
        self.ego_obs = self._obs[self.current_ego_idx]
        self.timestep += 1

        return self.ego_obs, self.joint_reward, done, info

    def reset(self) -> np.ndarray:
        """
        Reset environment to an initial state and return the first observation
        for the ego agent.

        :returns: Ego-agent's first observation
        """
        self.initialize_agent_indices()
        self.base_env.reset()
        self.timestep = 0

        self.state = self.base_env.state
        obs_p0, obs_p1 = self.featurize_fn(self.base_env.state)
        self.raw_obs = (obs_p0, obs_p1)
        # why do we need this
        self.ego_raw_obs = obs_p0 if self.current_ego_idx == 0 else obs_p1
        self.alt_raw_obs = obs_p1 if self.current_ego_idx == 0 else obs_p0
        obs_p0, obs_p1 = self.reduced_featurize_fn(self.base_env.state)
        self.reduced_obs = (obs_p0, obs_p1)
        # obs_p0 = self.get_reduced_obs(obs_p0, is_ego=self.current_ego_idx == 0)
        # obs_p1 = self.get_reduced_obs(obs_p1, is_ego=self.current_ego_idx == 1)
        self.alt_red_obs = obs_p1 if self.current_ego_idx == 0 else obs_p0
        self._obs = (obs_p0, obs_p1)

        # # when we start a new episode, we get an observation when the agent stands still
        # stay_idx = 4
        # self._obs, (self.ego_rew, self.alt_rew), done, _ = self.n_step(stay_idx)
        # if done:
        #     raise Exception("Game ended before ego moved")

        self.ego_obs = self._obs[self.current_ego_idx]

        assert self.ego_obs is not None
        self._old_ego_obs = self.ego_obs
        return self.ego_obs

    def render(self, mode='human', close=False):
        pass
