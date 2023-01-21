import os
from abc import abstractmethod, ABC

import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS
from typing import List, Tuple, Dict, Optional

from stable_baselines3 import PPO


class OvercookedMultiAgentEnv(gym.Env, ABC):
    def __init__(self, layout_name, ego_idx=None,
                 reduced_state_space_ego=False, use_skills_ego=True,
                 reduced_state_space_alt=False, use_skills_alt=True,
                 seed_num=None):
        """
        base_env: OvercookedEnv
        """
        super(OvercookedMultiAgentEnv, self).__init__()

        self._obs: Tuple[Optional[np.ndarray], ...] = tuple()
        self._old_ego_obs: Optional[np.ndarray] = None

        self.layout_name: str = layout_name
        self.set_env()

        if seed_num is not None:
            np.random.seed(seed_num)

        self.initial_ego_idx: int = ego_idx
        self.initialize_agent_indices()
        self.reduced_state_space_ego: bool = reduced_state_space_ego
        self.use_skills_ego: bool = use_skills_ego
        self.reduced_state_space_alt: bool = reduced_state_space_alt
        self.use_skills_alt: bool = use_skills_alt
        self.observation_space = self._setup_observation_space()
        self.n_primitive_actions = len(Action.ALL_ACTIONS)

        if self.use_skills_ego:
            # include skills
            self.idx_to_skill_ego = [self.move_up, self.move_down,
                             self.move_right, self.move_left,
                             self.stand_still, self.interact,
                             self.get_onion, self.get_tomato,
                             self.get_dish, self.serve_dish,
                             self.bring_to_pot, self.place_on_counter]
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
            self.idx_to_skill_alt = [self.move_up, self.move_down,
                             self.move_right, self.move_left,
                             self.stand_still, self.interact,
                             self.get_onion, self.get_tomato,
                             self.get_dish, self.serve_dish,
                             self.bring_to_pot, self.place_on_counter]
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

    def get_onion(self, agent_idx, last_pos=None, last_or=None):
        return self.interact_with_obj(agent_idx, last_pos, last_or, 'onion')

    def get_tomato(self, agent_idx, last_pos=None, last_or=None):
        return self.interact_with_obj(agent_idx, last_pos, last_or, 'tomato')

    def get_dish(self, agent_idx, last_pos=None, last_or=None):
        return self.interact_with_obj(agent_idx, last_pos, last_or, 'dish')

    def serve_dish(self, agent_idx, last_pos=None, last_or=None):
        return self.interact_with_obj(agent_idx, last_pos, last_or, 'serving')

    def bring_to_pot(self, agent_idx, last_pos=None, last_or=None):
        return self.interact_with_obj(agent_idx, last_pos, last_or, 'pot')

    def place_on_counter(self, agent_idx, last_pos=None, last_or=None):
        return self.interact_with_obj(agent_idx, last_pos, last_or, 'counter')

    def move_up(self, agent_idx, last_pos=None, last_or=None):
        return [(0, -1)], None, None

    def move_down(self, agent_idx, last_pos=None, last_or=None):
        return [(0, 1)], None, None

    def move_right(self, agent_idx, last_pos=None, last_or=None):
        return [(1, 0)], None, None

    def move_left(self, agent_idx, last_pos=None, last_or=None):
        return [(-1, 0)], None, None

    def stand_still(self, agent_idx, last_pos=None, last_or=None):
        return [(0, 0)], None, None

    def interact(self, agent_idx, last_pos=None, last_or=None):
        return ['interact'], None, None

    def interact_with_obj(self, agent_idx, last_pos=None, last_or=None, obj_type='onion'):
        counter_objects = self.mdp.get_counter_objects_dict(self.base_env.state)

        if obj_type == 'onion':
            all_obj_locs = self.mdp.get_onion_dispenser_locations()
        elif obj_type == 'tomato':
            all_obj_locs = self.mdp.get_tomato_dispenser_locations()
        elif obj_type == 'dish':
            all_obj_locs = self.mdp.get_dish_dispenser_locations()
        elif obj_type == 'serving':
            all_obj_locs = self.mdp.get_serving_locations()
        elif obj_type == 'pot':
            all_obj_locs = self.mdp.get_pot_locations()
        elif obj_type == 'counter':
            all_obj_locs = self.mdp.get_counter_locations()
        else:
            raise ValueError('Object type not recognized')

        obj_loc = all_obj_locs + counter_objects[obj_type]

        if last_pos is None:
            _, closest_obj_loc =  self.base_env.mp.min_cost_to_feature(
                self.base_env.state.players[agent_idx].pos_and_or,
                obj_loc, with_argmin=True)
        else:
            _, closest_obj_loc = self.base_env.mp.min_cost_to_feature(
                (last_pos, last_or),
                obj_loc, with_argmin=True)

        if closest_obj_loc is None:
            # means that we can't find the object
            # so we stay in the same position!
            goto_pos, goto_or =  self.base_env.state.players[agent_idx].pos_and_or
        else:
            # Determine where to stand to pick up
            goto_pos, goto_or =  self.base_env.mlam._get_ml_actions_for_positions([closest_obj_loc])[0]

        if last_pos is None:
            plan =  self.base_env.mp._get_position_plan_from_graph(
                self.base_env.state.players[agent_idx].pos_and_or, (goto_pos, goto_or))
            action_list = self.base_env.mp.action_plan_from_positions(plan,  self.base_env.state.players[
                agent_idx].pos_and_or, (goto_pos, goto_or))
        else:
            plan =  self.base_env.mp._get_position_plan_from_graph(
                (last_pos, last_or), (goto_pos, goto_or))
            action_list = self.base_env.mp.action_plan_from_positions(plan, (last_pos, last_or), (goto_pos, goto_or))[0]

        # save where plan should end up
        self.last_pos = goto_pos
        self.last_or = goto_or
        return action_list

    def initialize_agent_indices(self):
        if self.initial_ego_idx is None:
            ego_idx = np.random.randint(2)
        else:
            ego_idx = self.initial_ego_idx
        self.current_ego_idx = ego_idx
        self.current_alt_idx = (ego_idx + 1) % 2

    def set_env(self,
                placing_in_pot_multiplier=3,
                dish_pickup_multiplier=3,
                soup_pickup_multiplier=5,
                ):
        DEFAULT_ENV_PARAMS = {
            # add one because when we reset it takes up a timestep
            "horizon": 800 + 1,
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
        self.n_reduced_feats = 22
        obs = self.get_reduced_obs(obs, is_ego=True)
        obs_shape = obs.shape
        self.n_reduced_feats = obs_shape[0]
        high = np.ones(obs_shape, dtype=np.float32) * np.inf  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)
        return gym.spaces.Box(-high, high, dtype=np.float64)

    def get_reduced_obs(self, obs, is_ego):
        reduced_obs = (is_ego and self.reduced_state_space_ego) or (not is_ego and self.reduced_state_space_alt)

        if not reduced_obs:
            return obs
        # assumes 2 pots!
        assert self.n_reduced_feats == 22

        # if our obs is already reduced, return
        if obs.shape[0] == self.n_reduced_feats:
            return obs

        reduced_obs = np.zeros(self.n_reduced_feats)
        # first four features
        reduced_obs[:4] = obs[:4]
        # next four features (held items)
        reduced_obs[4:8] = obs[4:8]
        # closest soup # onions and # tomatoes
        reduced_obs[8] = obs[16]
        reduced_obs[9] = obs[17]
        # is cooking and ready
        reduced_obs[10] = obs[25]
        reduced_obs[11] = obs[26]
        # closest POT # onions and tomatoes
        reduced_obs[12] = obs[27]
        reduced_obs[13] = obs[28]
        # closest pot cook time
        reduced_obs[14] = obs[29]
        # 2nd closest pot is cooking and ready
        reduced_obs[15] = obs[35]
        reduced_obs[16] = obs[36]
        # 2nd closest pot # onions and tomatoes
        reduced_obs[17] = obs[37]
        reduced_obs[18] = obs[38]
        # 2nd closest pot cook time
        reduced_obs[19] = obs[39]
        # x and y position
        reduced_obs[20] = obs[-2]
        reduced_obs[21] = obs[-1]
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
        ego_actions, _, _ = self.idx_to_skill_ego[current_player_action](agent_idx=self.current_ego_idx)
        alt_actions, _, _ = self.idx_to_skill_alt[self.get_teammate_action()](agent_idx=self.current_alt_idx)

        # grab the first action from the sequence of actions
        ego_action = ego_actions[0]
        alt_action = alt_actions[0]

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
        obs_p0 = self.get_reduced_obs(obs_p0, is_ego=self.current_ego_idx == 0)
        obs_p1 = self.get_reduced_obs(obs_p1, is_ego=self.current_ego_idx == 1)

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
        self._obs = self.n_reset()
        self.initialize_agent_indices()

        # when we start a new episode, we get an observation when the agent stands still
        stay_idx = 4
        self._obs, (self.ego_rew, self.alt_rew), done, _ = self.n_step(stay_idx)
        if done:
            raise Exception("Game ended before ego moved")

        self.ego_obs = self._obs[self.current_ego_idx]
        self.ego_obs = self.get_reduced_obs(self.ego_obs, is_ego=True)

        assert self.ego_obs is not None
        self._old_ego_obs = self.ego_obs
        return self.ego_obs

    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[np.ndarray], ...]]:
        return self.multi_reset()

    def multi_reset(self) -> np.ndarray:
        self.base_env.reset()
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        ob_p0 = self.get_reduced_obs(ob_p0, is_ego=self.current_ego_idx == 0)
        ob_p1 = self.get_reduced_obs(ob_p1, is_ego=self.current_ego_idx == 1)
        return (ob_p0, ob_p1)

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
