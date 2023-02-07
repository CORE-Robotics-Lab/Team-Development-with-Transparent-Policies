import os
from abc import abstractmethod, ABC

import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS
from typing import List, Tuple, Dict, Optional

from sklearn.ensemble import RandomForestClassifier
from stable_baselines3 import PPO

from ipm.models.bc_agent import get_human_bc_partner


class OvercookedMultiAgentEnv(gym.Env, ABC):
    def __init__(self, layout_name, ego_idx=None,
                 reduced_state_space_ego=False, use_skills_ego=True,
                 reduced_state_space_alt=False, use_skills_alt=True,
                 seed_num=None, n_timesteps=800,
                 behavioral_model_path=None,
                 double_cook_times=False):
        """
        base_env: OvercookedEnv
        """
        super(OvercookedMultiAgentEnv, self).__init__()

        self._obs: Tuple[Optional[np.ndarray], ...] = tuple()
        self._old_ego_obs: Optional[np.ndarray] = None

        self.initial_ego_idx: int = ego_idx
        self.initialize_agent_indices()

        self.alt_red_obs = None
        self.behavioral_model_path = behavioral_model_path
        if behavioral_model_path is not None:
            self.behavioral_model = get_human_bc_partner(behavioral_model_path, layout_name, self.current_alt_idx)
            assert reduced_state_space_alt is True
        else:
            self.behavioral_model = None

        self.cook_time_threshold = 5
        if double_cook_times:
            # self.base_env.mdp.cook_time = 2 * self.base_env.mdp.cook_time not what we want, but might be useful
            self.cook_time_threshold = 2 * self.cook_time_threshold

        self.layout_name: str = layout_name
        self.n_timesteps = n_timesteps
        self.set_env()

        if seed_num is not None:
            np.random.seed(seed_num)

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

        self.carry_out_skills = False

        if self.use_skills_ego:
            # include skills
            self.idx_to_skill_ego = [
                                     self.move_up, self.move_down,
                                     self.move_right, self.move_left,
                                     self.stand_still,
                                     self.interact,
                                     self.get_closest_onion, self.get_closest_tomato,
                                     self.get_closest_dish, self.get_closest_soup,
                                     self.serve_at_closest_dispensary,
                                     self.bring_to_closest_pot, self.place_on_closest_counter]
        else:
            # otherwise, only include primitive actions
            self.idx_to_skill_ego = [self.move_up, self.move_down,
                                 self.move_right, self.move_left,
                                 self.stand_still, self.interact]
        self.n_actions_ego = len(self.idx_to_skill_ego)
        if not self.use_skills_ego:
            assert self.n_actions_ego == self.n_primitive_actions

        if self.use_skills_alt:
            # include skills
            self.idx_to_skill_alt = [
                                    self.move_up, self.move_down,
                                     self.move_right, self.move_left,
                                     self.stand_still,
                                     self.interact,
                                     self.get_closest_onion, self.get_closest_tomato,
                                     self.get_closest_dish, self.get_closest_soup,
                                     self.serve_at_closest_dispensary,
                                     self.bring_to_closest_pot, self.place_on_closest_counter]
        else:
            # otherwise, only include primitive actions
            self.idx_to_skill_alt = [self.move_up, self.move_down,
                                 self.move_right, self.move_left,
                                 self.stand_still, self.interact]
        self.n_actions_alt = len(self.idx_to_skill_alt)
        if not self.use_skills_alt:
            assert self.n_actions_alt == self.n_primitive_actions

        self.action_space  = gym.spaces.Discrete(self.n_actions_ego)
        self.check_conditions()

    def get_closest_onion(self, agent_idx):
        return self.perform_skill(agent_idx, 'pickup_onion')

    def get_closest_tomato(self, agent_idx):
        return self.perform_skill(agent_idx, 'pickup_tomato')

    def get_closest_dish(self, agent_idx):
        return self.perform_skill(agent_idx, 'pickup_dish')

    def get_closest_soup(self, agent_idx):
        return self.perform_skill(agent_idx, 'pickup_soup')

    def serve_at_closest_dispensary(self, agent_idx):
        return self.perform_skill(agent_idx, 'serving')

    def bring_to_closest_pot(self, agent_idx):
        return self.perform_skill(agent_idx, 'pot')

    def place_on_closest_counter(self, agent_idx):
        return self.perform_skill(agent_idx, 'place_on_closest_counter')

    def move_up(self, agent_idx):
        return (0, -1)

    def move_down(self, agent_idx):
        return (0, 1)

    def move_right(self, agent_idx):
        return (1, 0)

    def move_left(self, agent_idx):
        return (-1, 0)

    def stand_still(self, agent_idx):
        return (0, 0)

    def interact(self, agent_idx):
        return 'interact'

    def perform_skill(self, agent_idx, skill_type='onion') -> Tuple[Tuple[int, int], int]:
        if 'pickup' in skill_type:
            if skill_type == 'pickup_onion':
                counter_objects = list(self.mdp.get_counter_objects_dict(self.base_env.state)['onion'])
                obj_loc = self.mdp.get_onion_dispenser_locations()
            elif skill_type == 'pickup_tomato':
                counter_objects = list(self.mdp.get_counter_objects_dict(self.base_env.state)['tomato'])
                obj_loc = self.mdp.get_tomato_dispenser_locations()
            elif skill_type == 'pickup_dish':
                counter_objects = list(self.mdp.get_counter_objects_dict(self.base_env.state)['dish'])
                obj_loc = self.mdp.get_dish_dispenser_locations()
            elif skill_type == 'pickup_soup':
                counter_objects = list(self.mdp.get_counter_objects_dict(self.base_env.state)['soup'])
                potential_locs = self.mdp.get_pot_locations()
                obj_loc = []
                for pos in potential_locs:
                    if self.base_env.mdp.soup_ready_at_location(self.base_env.state, pos):
                        obj_loc.append(pos)
            else:
                raise ValueError('Unknown pickup type')
            if len(counter_objects) > 0:
                obj_loc.extend(counter_objects)
        elif skill_type == 'serving':
            obj_loc = self.mdp.get_serving_locations()
        elif skill_type == 'pot':
            obj_loc = self.mdp.get_pot_locations()
        elif skill_type == 'place_on_closest_counter':
            obj_loc = self.mdp.get_empty_counter_locations(self.base_env.state)
        else:
            raise ValueError('Unknown obj type')

        pos_and_or = self.base_env.state.players[agent_idx].pos_and_or

        min_dist = np.Inf
        goto_pos_and_or = None

        if 'place' not in skill_type and 'pickup' not in skill_type:
            _, closest_obj_loc = self.base_env.mp.min_cost_to_feature(pos_and_or, obj_loc, with_argmin=True)
            if closest_obj_loc is None:
                # stand still because we can't do anything
                return (0, 0)
            else:
                goto_pos_and_or = self.base_env.mlam._get_ml_actions_for_positions([closest_obj_loc])[0]
        else:
            if obj_loc is None:
                return (0, 0)
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
                return (0, 0)

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
        return actions[0]

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
        obs = self.featurize_fn(dummy_state)[0]
        reduced_obs = self.get_reduced_obs(obs, is_ego=True)
        obs_shape = reduced_obs.shape
        self.n_reduced_feats = obs_shape[0]
        # if below fails, we need to update shape of alt_red_obs
        high = np.ones(obs_shape, dtype=np.float32) * np.inf  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)
        return gym.spaces.Box(-high, high, dtype=np.float64)

    def add_teammate_action(self, obs):
        if self.alt_red_obs is None:
            if self.reduced_state_space_ego:
                self.alt_red_obs = obs
            else:
                self.alt_red_obs = self.get_reduced_obs(obs, is_ego=False)
        action = self.behavioral_model.predict(self.alt_red_obs)[0]
        obs = np.concatenate([obs, [action]])
        return obs

    def get_reduced_obs(self, obs, is_ego):
        reduced_obs = (is_ego and self.reduced_state_space_ego) or (not is_ego and self.reduced_state_space_alt)

        if not reduced_obs:
            assert obs.shape[0] > 22
            if is_ego and self.behavioral_model is not None:
                return self.add_teammate_action(obs)
            return obs

        # # assumes 2 pots!
        assert obs.shape[0] == 96

        reduced_obs = []
        # first four features (direction facing)
        reduced_obs.append(obs[0])
        reduced_obs.append(obs[1])
        reduced_obs.append(obs[2])
        reduced_obs.append(obs[3])

        # next four features (held items)
        reduced_obs.append(obs[4])
        reduced_obs.append(obs[5])
        reduced_obs.append(obs[6])
        reduced_obs.append(obs[7])

        # # closest soup # onions and # tomatoes
        # reduced_obs.append(obs[16])
        # reduced_obs.append(obs[17])

        # is cooking and ready
        reduced_obs.append(obs[25])
        reduced_obs.append(obs[26])

        # closest POT # onions and tomatoes
        if 3 > obs[27] + obs[28] >= 0:
            # needs more ingredients
            reduced_obs.append(1)
        else:
            reduced_obs.append(0)

        # closest pot cook time
        # pot almost done
        if self.cook_time_threshold > obs[29] > 0:
            reduced_obs.append(1)
        else:
            reduced_obs.append(0)

        # 2nd closest pot is cooking and ready
        reduced_obs.append(obs[35])
        reduced_obs.append(obs[36])

        # 2nd closest pot # onions and tomatoes
        if 3 > obs[37] + obs[38] >= 0:
            # needs more ingredients
            reduced_obs.append(1.0)
        else:
            reduced_obs.append(0.0)

        # 2nd closest pot cook time
        # pot almost done
        if self.cook_time_threshold > obs[39] > 0:
            reduced_obs.append(1.0)
        else:
            reduced_obs.append(0.0)

        # x and y position
        reduced_obs.append(obs[-2])
        reduced_obs.append(obs[-1])

        # other agent (absolute) position
        reduced_obs.append(obs[-2] + obs[-4])
        reduced_obs.append(obs[-1] + obs[-3])

        # other agent facing direction
        reduced_obs.append(obs[46])
        reduced_obs.append(obs[47])
        reduced_obs.append(obs[48])
        reduced_obs.append(obs[49])

        # other player holding onion, soup, dish, or tomato
        reduced_obs.append(obs[50])
        reduced_obs.append(obs[51])
        reduced_obs.append(obs[52])
        reduced_obs.append(obs[53])

        # num_dishes_on_counter = 0
        # for obj in self.base_env.state.objects:
        #     if obj.name == 'dish':
        #         num_dishes_on_counter += 1
        # reduced_obs.append(num_dishes_on_counter)
        #
        # num_onions_on_counter = 0
        # for obj in self.base_env.state.objects:
        #     if obj.name == 'onion':
        #         num_onions_on_counter += 1
        # reduced_obs.append(num_onions_on_counter)
        #
        # num_tomatoes_on_counter = 0
        # for obj in self.base_env.state.objects:
        #     if obj.name == 'tomato':
        #         num_tomatoes_on_counter += 1
        # reduced_obs.append(num_tomatoes_on_counter)
        #
        # num_soups_on_counter = 0
        # for obj in self.base_env.state.objects:
        #     if obj.name == 'soup':
        #         num_soups_on_counter += 1
        # reduced_obs.append(num_soups_on_counter)

        reduced_obs = np.array(reduced_obs)

        if is_ego and self.behavioral_model is not None:
            return self.add_teammate_action(reduced_obs)
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

        if not self.ego_currently_performing_skill:
            ego_action = self.idx_to_skill_ego[current_player_action](agent_idx=self.current_ego_idx)
        else:
            ego_action = self.ego_current_action_seq.pop(0)
            if len(self.ego_current_action_seq) == 0:
                self.ego_currently_performing_skill = False
        if not self.alt_currently_performing_skill:
            alt_action = self.idx_to_skill_alt[self.get_teammate_action()](agent_idx=self.current_alt_idx)
        else:
            alt_action = self.alt_current_action_seq.pop(0)
            if len(self.alt_current_action_seq) == 0:
                self.alt_currently_performing_skill = False

        self.previous_ego_action = ego_action

        if self.current_ego_idx == 0:
            joint_action = (ego_action, alt_action)
        else:
            joint_action = (alt_action, ego_action)

        next_state, reward, done, info = self.base_env.step(joint_action)
        self.state = next_state

        # reward shaping
        reward_ego = reward + info['shaped_r_by_agent'][self.current_ego_idx]
        reward_alt = reward + info['shaped_r_by_agent'][self.current_alt_idx]

        (obs_p0, obs_p1) = self.featurize_fn(next_state)
        self.ego_raw_obs = obs_p0 if self.current_ego_idx == 0 else obs_p1
        self.alt_raw_obs = obs_p1 if self.current_ego_idx == 0 else obs_p0
        obs_p0 = self.get_reduced_obs(obs_p0, is_ego=self.current_ego_idx == 0)
        obs_p1 = self.get_reduced_obs(obs_p1, is_ego=self.current_ego_idx == 1)
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

        if done:
            return self._old_ego_obs, self.ego_rew, done, info

        self._old_ego_obs = self.ego_obs
        self.ego_obs = self._obs[self.current_ego_idx]

        return self.ego_obs, self.ego_rew, done, info

    def reset(self) -> np.ndarray:
        """
        Reset environment to an initial state and return the first observation
        for the ego agent.

        :returns: Ego-agent's first observation
        """
        self.initialize_agent_indices()
        self.base_env.reset()

        self.state = self.base_env.state
        obs_p0, obs_p1 = self.featurize_fn(self.base_env.state)
        self.ego_raw_obs = obs_p0 if self.current_ego_idx == 0 else obs_p1
        self.alt_raw_obs = obs_p1 if self.current_ego_idx == 0 else obs_p0
        obs_p0 = self.get_reduced_obs(obs_p0, is_ego=self.current_ego_idx == 0)
        obs_p1 = self.get_reduced_obs(obs_p1, is_ego=self.current_ego_idx == 1)
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

class OvercookedSelfPlayEnv(OvercookedMultiAgentEnv):
    def check_conditions(self):
        assert self.reduced_state_space_ego == self.reduced_state_space_alt

    def get_teammate_action(self):
        # for the self-play case, we just stand still while we
        # alternate between the agents
        # this is because we need a joint action at every timestep
        # but would like to use this same policy for a single agent
        STAY_IDX = 4
        return STAY_IDX

    def update_ego(self):
        # for the self-play case, we alternate between both the agents
        # where one agent makes a move, and the other stands still
        self.current_alt_idx = self.current_ego_idx
        self.current_ego_idx = (self.current_ego_idx + 1) % 2

class OvercookedPlayWithFixedPartner(OvercookedMultiAgentEnv):
    def __init__(self, partner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load in the potential teammates
        self.partner = partner

    def check_conditions(self):
        pass

    def get_teammate_action(self):
        obs = self._obs[self.current_alt_idx]
        teammate_action, _states = self.partner.predict(obs)
        return teammate_action

    def update_ego(self):
        # for round-robin, we never change these since we have 1 single agent that is learning
        pass

    def reset(self):
        self.ego_obs = super().reset()
        return self.ego_obs

class OvercookedRoundRobinEnv(OvercookedMultiAgentEnv):
    def __init__(self, teammate_locations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load in the potential teammates

        self.teammate_locations = teammate_locations
        # potential teammates are in the data folder.
        self.all_teammates = []
        # iterate through all subfolders and load all files
        for root, dirs, files in os.walk( self.teammate_locations):
            for file in files:
                if file.endswith('.zip'):
                    agent = PPO.load(os.path.join(root, file))
                    self.all_teammates.append(agent)
        self.teammate_idx = np.random.randint(len(self.all_teammates))

    def check_conditions(self):
        pass

    def get_teammate_action(self):
        obs = self._obs[self.current_alt_idx]
        teammate_action, _states = self.all_teammates[self.teammate_idx].predict(obs)
        return teammate_action

    def update_ego(self):
        # for round-robin, we never change these since we have 1 single agent that is learning
        pass

    def reset(self):
        self.ego_obs = super().reset()
        self.teammate_idx = np.random.randint(len(self.all_teammates))
        return self.ego_obs
