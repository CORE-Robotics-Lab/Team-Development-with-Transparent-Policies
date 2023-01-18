import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS
from typing import List, Tuple, Dict, Optional


class OvercookedSelfPlayEnv(gym.Env):
    def __init__(self, layout_name, baselines=False):
        """
        base_env: OvercookedEnv
        """
        super(OvercookedSelfPlayEnv, self).__init__()

        self._players: Tuple[int, ...] = tuple()
        self._obs: Tuple[Optional[np.ndarray], ...] = tuple()
        self._old_ego_obs: Optional[np.ndarray] = None

        self.total_rews = [0] * 1
        self.ego_moved = False
        self.layout_name = layout_name
        self.set_env()

        if baselines: np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        self.current_turn = 0

        self.idx_to_skill = [self.move_up, self.move_down,
                             self.move_left, self.move_right,
                             self.stand_still,
                             self.get_onion, self.get_tomato,
                             self.get_dish, self.serve_dish,
                             self.bring_to_pot, self.place_on_counter]
        self.n_primitive_actions = len(Action.ALL_ACTIONS)
        self.n_skills = len(self.idx_to_skill)
        self.action_space  = gym.spaces.Discrete(self.n_skills)

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

    def move_left(self, agent_idx, last_pos=None, last_or=None):
        return [(-1, 0)], None, None

    def move_right(self, agent_idx, last_pos=None, last_or=None):
        return [(1, 0)], None, None

    def stand_still(self, agent_idx, last_pos=None, last_or=None):
        return [(0, 0)], None, None

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

    def set_env(self,
                placing_in_pot_multiplier=3,
                dish_pickup_multiplier=3,
                soup_pickup_multiplier=5,
                ):
        DEFAULT_ENV_PARAMS = {
            "horizon": 800,
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": placing_in_pot_multiplier,
            "DISH_PICKUP_REWARD": dish_pickup_multiplier,
            "SOUP_PICKUP_REWARD": soup_pickup_multiplier,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        self.mdp = OvercookedGridworld.from_layout_name(layout_name=self.layout_name, rew_shaping_params=rew_shaping_params)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, **DEFAULT_ENV_PARAMS)
        self.featurize_fn = self.base_env.featurize_state_mdp

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        # below is original obs shape
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        self.n_reduced_feats = 14
        obs_shape = (self.n_reduced_feats,)
        high = np.ones(obs_shape, dtype=np.float32) * np.inf  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)
        return gym.spaces.Box(-high, high, dtype=np.float64)

    def get_reduced_obs(self, obs):
        # assumes 2 pots!
        assert self.n_reduced_feats == 14
        reduced_obs = np.zeros(self.n_reduced_feats)
        # get argmax over first four features
        reduced_obs[0] = np.argmax(obs[:4])
        # get argmax over next four features (held items)
        reduced_obs[1] = np.argmax(obs[4:8])
        # closest soup # onions and # tomatoes
        reduced_obs[2] = obs[16]
        reduced_obs[3] = obs[17]
        # is cooking and ready
        reduced_obs[4] = obs[25]
        reduced_obs[5] = obs[26]
        # closest POT # onions and tomatoes
        reduced_obs[6] = obs[27]
        reduced_obs[7] = obs[28]
        # closest pot cook time
        reduced_obs[8] = obs[29]
        # 2nd closest pot is cooking and ready
        reduced_obs[9] = obs[35]
        reduced_obs[10] = obs[36]
        # 2nd closest pot # onions and tomatoes
        reduced_obs[11] = obs[37]
        reduced_obs[12] = obs[38]
        # 2nd closest pot cook time
        reduced_obs[13] = obs[39]
        return reduced_obs

    def getDummyEnv(self, player_num: int):
        """
        Returns a dummy environment with just an observation and action
        space that a partner agent can use to construct their policy network.

        :param player_num: the partner number to query
        """
        return self

    def _get_actions(self, current_player_action):
        STAY = 4
        actions = []
        if self.current_turn == 0:
            actions.append(current_player_action)
            actions.append(STAY)
        else:
            actions.append(STAY)
            actions.append(current_player_action)
        self.current_turn = (self.current_turn + 1) % 2
        return np.array(actions)

    def multi_step(self, ego_action, alt_action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        # print('Current turn', self.current_turn)
        # print('Ego action', ego_action)
        # print('Alt action', alt_action)
        if self.current_turn == 0:
            main_action = Action.INDEX_TO_ACTION[ego_action]
            other_actions, _, _ = self.idx_to_skill[alt_action](agent_idx=self.current_turn)
            joint_action = (main_action, other_actions[0])
        else:
            main_action = Action.INDEX_TO_ACTION[alt_action]
            other_actions, _, _ = self.idx_to_skill[ego_action](agent_idx=self.current_turn)
            joint_action = (other_actions[0], main_action)

        next_state, reward, done, info = self.base_env.step(joint_action)

        # reward shaping
        rew_shape = info['shaped_r_by_agent'][self.current_turn]
        reward = reward + rew_shape

        #print(self.base_env.mdp.state_string(next_state))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        ob_p0 = self.get_reduced_obs(ob_p0)
        ob_p1 = self.get_reduced_obs(ob_p1)
        ego_obs, alt_obs = ob_p0, ob_p1

        return (ego_obs, alt_obs), (reward, reward), done, {}#info

    def n_step(
                    self,
                    actions: List[np.ndarray],
                ) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[np.ndarray], ...],
                           Tuple[float, ...],
                           bool,
                           Dict]:
        return ((0, 1),) + self.multi_step(actions[0], actions[1])

    def step(
                self,
                action: np.ndarray
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
        acts = self._get_actions(action)
        self._players, self._obs, rews, done, info = self.n_step(acts)
        rews = (float(rews[0]), float(rews[1]))

        prev_player = (self.current_turn + 1) % 2

        rew = rews[prev_player]

        if done:
            return self._old_ego_obs, rew, done, info

        current_player = self.current_turn

        ego_obs = self._obs[self._players[current_player]]
        self._old_ego_obs = ego_obs
        return ego_obs, rew, done, info

    def reset(self) -> np.ndarray:
        """
        Reset environment to an initial state and return the first observation
        for the ego agent.

        :returns: Ego-agent's first observation
        """
        self._players, self._obs = self.n_reset()
        self.total_rews = [0] * 1
        self.ego_moved = False

        while 0 not in self._players:
            acts = self._get_actions()
            self._players, self._obs, rews, done, _ = self.n_step(acts)

            if done:
                raise Exception("Game ended before ego moved")


        ego_obs = self._obs[self._players.index(0)]
        # ego_obs = self.get_reduced_obs(ego_obs)

        assert ego_obs is not None
        self._old_ego_obs = ego_obs
        return ego_obs

    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[np.ndarray], ...]]:
        return (0, 1), self.multi_reset()

    def multi_reset(self) -> np.ndarray:
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        ob_p0 = self.get_reduced_obs(ob_p0)
        ob_p1 = self.get_reduced_obs(ob_p1)
        ego_obs, alt_obs = ob_p0, ob_p1

        return (ego_obs, alt_obs)

    def render(self, mode='human', close=False):
        pass

