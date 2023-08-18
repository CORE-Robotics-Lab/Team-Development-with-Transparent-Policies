#!/bin/bash


# create new .ini files with variants of hyperparameters to train (ex. vary seed, lr...)
~/PycharmProjects/PantheonRL/venv/bin/python3.7 /home/rohanpaleja/PycharmProjects/PantheonRL/trainer.py OvercookedMultiEnv-v0 PPO PPO --seed 4000 --preset 1 --verbose --remove_intent_model --high_level_action --high_level_state --domain 2 --use_idct --saving-name robot_policy_optimization_1 --load_rohan --ego_load '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/pre_robot_update.tar' --alt_load '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/pre_robot_update.tar' --total-timesteps 24000 &
~/PycharmProjects/PantheonRL/venv/bin/python3.7 /home/rohanpaleja/PycharmProjects/PantheonRL/trainer.py OvercookedMultiEnv-v0 PPO PPO --seed 4001 --preset 1 --verbose --remove_intent_model --high_level_action --high_level_state --domain 2 --use_idct --saving-name robot_policy_optimization_1 --load_rohan --ego_load '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/pre_robot_update.tar' --alt_load '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/pre_robot_update.tar' --robot_scratch --total-timesteps 24000

#python3 /home/rohanpaleja/PycharmProjects/ipm/ipm/bin/evaluate_hyperparameters.py  --config_file '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/data/test_hyperparams_4.ini' &
# use os to run multiple evals at once

# find which model is best

# clean up and load best model