import math
import os
import pygame
import cv2
import time
import torch
from typing import Callable
from ipm.gui.main_experiment import MainExperiment
import argparse


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

    # recommends_rule
    # intro -> overcooked-tutorial -> specific rule tutorial
    # -> (interaction -> recommending rule -> assessment pages -> recommend new policy vs old) -> end survey

    # fcp
    # intro -> overcooked-tutorial -> tutorial stating that fcp was trained to maximize reward for a group of agents
    # -> (interaction -> assessment pages) -> end survey

    # ha_ppo
    # intro -> overcooked-tutorial -> tutorial stating that agent will update policy to maximize reward based on recent data
    # -> (interaction -> assessment pages) -> end survey

    # no_modification_bb
    # intro -> overcooked-tutorial
    # -> (interaction -> assessment pages) -> end survey

    # no_modification_bb
    # intro -> overcooked-tutorial -> specific tree tutorial
    # -> (interaction -> assessment pages -> visualize tree) -> end survey

    conditions = ['human_modifies_tree',
                  'optimization_via_rl_or_ga',
                  'optimization_via_rl_or_ga_while_modifying_reward',
                  'recommends_rule',
                  'fcp',
                  'ha_ppo',
                  'no_modification_bb',
                  'no_modification_interpretable']
    parser.add_argument('--group', help='Experiment Group', type=str, default='human_modifies_tree', choices=conditions)
    args = parser.parse_args()

    experiment = MainExperiment(args.group)
    experiment.launch()
