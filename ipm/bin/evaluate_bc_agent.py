import argparse
import os
import numpy as np
import sys
sys.path.insert(0, '../../overcooked_ai/src/')
sys.path.insert(0, '../../overcooked_ai/src/overcooked_ai_py')
from ipm.overcooked.overcooked_envs import OvercookedRoundRobinEnv
from ipm.models.bc_agent import get_human_bc_partner


def evaluate_model(model, env, num_episodes, include_obs_acts=False):
    all_episode_rewards = []
    all_episode_obs = []
    all_episode_acts = []
    for i in range(num_episodes):
        done = False
        obs = env.reset()
        total_reward = 0.0
        while not done:
            # _states are only useful when using LSTM policies
            all_episode_obs.append(obs)
            action, _ = model.predict(obs)
            # print('action:', action)
            all_episode_acts.append(action)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # all_rewards_per_timestep[seed].append(last_fitness)
        all_episode_rewards.append(total_reward)
    if include_obs_acts:
        return np.mean(all_episode_rewards), (all_episode_obs, all_episode_acts)
    else:
        return np.mean(all_episode_rewards)

def main(traj_directory, layout_name, ego_idx, high_level=True):
    alt_idx = (ego_idx + 1) % 2
    reduce_state_space = True

    teammate_paths = os.path.join('data', layout_name, 'self_play_training_models')
    env = OvercookedRoundRobinEnv(teammate_locations=teammate_paths, layout_name=layout_name, seed_num=0,
                                  ego_idx=ego_idx,
                                  reduced_state_space_ego=True, reduced_state_space_alt=False,
                                  use_skills_ego=True, use_skills_alt=False)

    # below is just used for debugging
    # from ipm.overcooked.overcooked_multi import OvercookedSelfPlayEnv
    # env = OvercookedSelfPlayEnv(layout_name=layout_name, ego_idx=ego_idx,
    #                                          reduced_state_space_ego=True,
    #                                          reduced_state_space_alt=True,
    #                                          use_skills_ego=True,
    #                                          use_skills_alt=True,
    #                                          n_timesteps=200)

    if high_level:
        human_policy_estimator, bc_model = get_human_bc_partner(traj_directory=traj_directory, layout_name=layout_name,
                                                                bc_agent_idx=ego_idx, include_states=True, get_intent_model=True)
    else:
        bc_model = get_human_bc_partner(traj_directory=traj_directory, layout_name=layout_name,
                                        bc_agent_idx=ego_idx, include_states=True,
                                        get_intent_model=False)
    avg_rew = evaluate_model(bc_model, env, num_episodes=10, include_obs_acts=False)
    print('Average reward:', avg_rew)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads BC agent')
    parser.add_argument('--traj_directory', type=str, default='trajectories')
    parser.add_argument('--layout_name', type=str, default='forced_coordination')
    parser.add_argument('--ego_idx', type=int, default=1)
    parser.add_argument('--high_level', type=bool, default=True)
    args = parser.parse_args()
    main(args.traj_directory, args.layout_name, args.ego_idx, args.high_level)

