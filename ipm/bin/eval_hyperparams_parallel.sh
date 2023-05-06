#!/bin/bash


# create new .ini files with variants of hyperparameters to train (ex. vary seed, lr...)
python3 /home/rohanpaleja/PycharmProjects/ipm/ipm/bin/evaluate_hyperparameters.py  --config_file '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/data/test_hyperparams_1.ini' &
python3 /home/rohanpaleja/PycharmProjects/ipm/ipm/bin/evaluate_hyperparameters.py  --config_file '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/data/test_hyperparams_2.ini' &
python3 /home/rohanpaleja/PycharmProjects/ipm/ipm/bin/evaluate_hyperparameters.py  --config_file '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/data/test_hyperparams_3.ini' &
#python3 /home/rohanpaleja/PycharmProjects/ipm/ipm/bin/evaluate_hyperparameters.py  --config_file '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/data/test_hyperparams_4.ini' &
# use os to run multiple evals at once

# find which model is best

# clean up and load best model