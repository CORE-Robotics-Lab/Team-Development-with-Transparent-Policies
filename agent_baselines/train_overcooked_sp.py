"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from pantheonrl.common.agents import OnPolicyAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST

layout = 'simple'
assert layout in LAYOUT_LIST

# Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
# register an environment and construct it using gym.make.
env = gym.make('OvercookedSelfPlayEnv-v0', layout_name=layout)

# gym_env = get_vectorized_gym_env(
#   env, 'Overcooked-v0', featurize_fn=lambda x: mdp.lossless_state_encoding(x), **params
# )

# Before training your ego agent, you first need to add your partner agents
# to the environment. You can create adaptive partner agents using
# OnPolicyAgent (for PPO/A2C) or OffPolicyAgent (for DQN/SAC). If you set
# verbose to true for these agents, you can also see their learning progress


seed = 20
agent = PPO('MlpPolicy', env, verbose=1, seed=seed)

# Finally, you can construct an ego agent and train it in the environment
# ego = PPO('MlpPolicy', env, verbose=1)

N_steps = 500000
checkpoint_freq = N_steps // 100

checkpoint_callback = CheckpointCallback(
  save_freq=checkpoint_freq,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

agent.learn(total_timesteps=500000, callback=checkpoint_callback)

# To visualize the agent:
# python overcookedgym/overcooked-flask/app.py --modelpath_p0 ../logs/rl_model_500000_steps --modelpath_p1 ../logs/rl_model_50000_steps --layout_name simple