import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Following code is inspired from PantheonRL
# https://github.com/Stanford-ILIAD/PantheonRL/blob/master/pantheonrl/common/multiagentenv.py
class OvercookedWithPartner(gym.Env):
    def __init__(self, layout_name, partner, ego_ind: int = 0, baselines=False):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(OvercookedWithPartner, self).__init__()

        self._obs: Tuple[Optional[np.ndarray], ...] = tuple()
        self._old_ego_obs: Optional[np.ndarray] = None
        self.ego_ind = ego_ind

        self.total_rew = 0.0
        self.partner = partner
        self.ego_moved = False

        DEFAULT_ENV_PARAMS = {
            "horizon": 800,
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        self.mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, rew_shaping_params=rew_shaping_params)

        self.base_env = OvercookedEnv.from_mdp(self.mdp, **DEFAULT_ENV_PARAMS)
        self.featurize_fn = lambda x: self.mdp.featurize_state(x)

        if baselines: np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        self.lA = len(Action.ALL_ACTIONS)
        self.action_space  = gym.spaces.Discrete( self.lA )
        self.current_turn = 0

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32) * np.inf  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)
        # TODO: this should be multi-discrete, not Box
        return gym.spaces.Box(-high, high, dtype=np.float64)

    def getDummyEnv(self, player_num: int):
        """
        Returns a dummy environment with just an observation and action
        space that a partner agent can use to construct their policy network.

        :param player_num: the partner number to query
        """
        return self

    def _update_players(self, rews, done):
        self.total_rew += rews

    def _get_actions(self, players, obs, ego_act=None):
        actions = []
        for player, ob in zip(players, obs):
            if player == self.ego_ind:
                actions.append(ego_act)
            else:
                agent = self.partner
                actions.append(agent.get_action(ob))
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
        ego_action, alt_action = Action.INDEX_TO_ACTION[ego_action], Action.INDEX_TO_ACTION[alt_action]
        joint_action = (ego_action, alt_action)

        next_state, reward, done, info = self.base_env.step(joint_action)

        # reward shaping
        rew_shape = info['shaped_r']
        reward = reward + rew_shape

        #print(self.base_env.mdp.state_string(next_state))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
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
        ego_rew = 0.0

        while True:
            acts = self._get_actions(self._players, self._obs, action)
            self._players, self._obs, rews, done, info = self.n_step(acts)
            info['_partnerid'] = self.partnerids

            self._update_players(rews, done)

            ego_rew += rews[self.ego_ind] if self.ego_moved \
                else self.total_rew[self.ego_ind]

            self.ego_moved = True

            if done:
                return self._old_ego_obs, ego_rew, done, info

            if self.ego_ind in self._players:
                break

        ego_obs = self._obs[self._players.index(self.ego_ind)]
        self._old_ego_obs = ego_obs
        return ego_obs, ego_rew, done, info

    def reset(self) -> np.ndarray:
        """
        Reset environment to an initial state and return the first observation
        for the ego agent.

        :returns: Ego-agent's first observation
        """
        self._players, self._obs = self.n_reset()
        self.should_update = False
        self.total_rew = [0] * 2
        self.ego_moved = False

        while self.ego_ind not in self._players:
            acts = self._get_actions(self._players, self._obs)
            self._players, self._obs, rews, done, _ = self.n_step(acts)

            if done:
                raise Exception("Game ended before ego moved")

            self._update_players(rews, done)

        ego_obs = self._obs[self._players.index(self.ego_ind)]

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
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs)


    def render(self, mode='human', close=False):
        pass
