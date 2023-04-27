import configparser
import math
import os
import pygame
import cv2
import time
import torch
from typing import Callable
from ipm.gui.main_experiment import MainExperiment
import argparse

class HyperparameterConfig:
    def __init__(self, config_filepath):
        # load in config file
        config = configparser.ConfigParser()
        config.read(config_filepath)

        self.layout = config.get('main', 'layout')
        self.prior_iteration_data = config.get('main', 'prior_iteration_data')
        self.n_episode_samples = config.getint('main', 'n_episode_samples')
        self.n_random_seeds = config.getint('main', 'n_random_seeds')

        self.hpo = config.getboolean('main', 'hpo')
        self.hpo_lr = config.getfloat('main', 'hpo_lr')
        self.hpo_n_epochs = config.getint('main', 'hpo_n_epochs')

        self.ipo = config.getboolean('main', 'ipo')
        self.ipo_lr = config.getfloat('main', 'ipo_lr')
        self.ipo_n_epochs = config.getint('main', 'ipo_n_epochs')

        self.rpo = config.getboolean('main', 'rpo')
        self.rpo_ga = config.getboolean('main', 'rpo_ga')
        self.rpo_rl = config.getboolean('main', 'rpo_rl')
        self.rpo_random_initial_idct = config.getboolean('main', 'rpo_random_initial_idct')

        self.rpo_ga_data_file = config.get('main', 'rpo_ga_data_file')
        self.rpo_ga_depth = config.getint('main', 'rpo_ga_depth')
        self.rpo_ga_n_gens = config.getint('main', 'rpo_ga_n_gens')
        self.rpo_ga_n_pop = config.getint('main', 'rpo_ga_n_pop')
        self.rpo_ga_n_parents_mating = config.getint('main', 'rpo_ga_n_parents_mating')
        self.rpo_ga_crossover_prob = config.getfloat('main', 'rpo_ga_crossover_prob')
        self.rpo_ga_crossover_type = config.get('main', 'rpo_ga_crossover_type')
        self.rpo_ga_mutation_prob = config.getfloat('main', 'rpo_ga_mutation_prob')
        self.rpo_ga_mutation_type = config.get('main', 'rpo_ga_mutation_type')

        self.rpo_rl_n_steps = config.getint('main', 'rpo_rl_n_steps')
        self.rpo_rl_lr = config.getfloat('main', 'rpo_rl_lr')
        self.rpo_rl_only_optimize_leaves = config.getboolean('main', 'rpo_rl_only_optimize_leaves')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains self-play agent on overcooked with checkpointing')
    # first four would get env reward performance + tree visualization (static), others get env performance
    # agent starts with same policy for all conditions

    # human_modifies_tree
    # intro -> overcooked-tutorial -> specific policy modification tutorial
    # -> (interaction -> assessment pages -> specific policy modification (replace old) -> recommend new policy vs old) -> end survey

    # optimization_via_rl_or_ga
    # intro -> overcooked-tutorial -> tutorial stating that agent will update policy to maximize reward based on recent data
    # -> (interaction -> assessment pages -> recommend new policy vs old) -> end survey

    # optimization_via_rl_or_ga_while_modifying_reward
    # intro -> overcooked-tutorial -> specific reward modifying tutorial
    # -> (interaction -> env reward modification page -> assessment pages -> recommend new policy vs old) -> end survey

    # no_modification_bb
    # intro -> overcooked-tutorial
    # -> (interaction -> assessment pages) -> end survey

    # no_modification_bb
    # intro -> overcooked-tutorial -> specific tree tutorial
    # -> (interaction -> assessment pages -> visualize tree) -> end survey

    conditions = ['human_modifies_tree',
                  'optimization',
                  'optimization_while_modifying_reward',
                  'no_modification_bb',
                  'no_modification_interpretable',
                  'fcp']
    parser.add_argument('--group', help='Experiment Group', type=str, default='human_modifies_tree', choices=conditions)
    parser.add_argument('--disable_surveys', help='Disable Surveys', action='store_true')
    parser.add_argument('--hyperparam_config_file', help='Config file', type=str, default='data/experiment_hyperparams.ini')
    args = parser.parse_args()

    hp_config = HyperparameterConfig(args.hyperparam_config_file)

    CURRENT_CONDITION = args.group

    experiment = MainExperiment(CURRENT_CONDITION, conditions, disable_surveys=args.disable_surveys, hp_config=hp_config)
    experiment.launch()
