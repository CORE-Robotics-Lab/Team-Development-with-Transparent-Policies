import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import gym
import pandas as pd

# Parallel environments
n_envs = 4
env = make_vec_env("CartPole-v1", n_envs=n_envs)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=120000)

all_obs = []
all_actions = []

N_TOTAL_SAMPLES = 100000
CURRENT_NUM_SAMPLES = 0
env = gym.make('CartPole-v1')
obs = env.reset()

while CURRENT_NUM_SAMPLES < N_TOTAL_SAMPLES:
    all_obs.append(obs)
    action, _states = model.predict(obs)
    all_actions.append(action)
    obs, reward, done, info = env.step(action)
    CURRENT_NUM_SAMPLES += 1
    if done:
        obs = env.reset()

all_actions = np.array(all_actions)
all_obs = np.vstack(all_obs)

data = {'Cart Position': all_obs[:, 0], 'Cart Velocity': all_obs[:, 1],
        'Pole Angle': all_obs[:, 2], 'Pole Angular Velocity': all_obs[:, 3],
        'Action': all_actions}

df = pd.DataFrame(data)
df.to_csv('expert_data_cartpole.csv')
