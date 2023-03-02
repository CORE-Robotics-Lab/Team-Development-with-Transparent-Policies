import math
import os
import pygame
import numpy as np
import cv2
import time
import torch
from typing import Callable
import pickle5 as pickle
from pygame import gfxdraw

from ipm.gui.nasa_tlx import run_gui
from ipm.gui.pages import GUIPageCenterText, TreeCreationPage, EnvPage, EnvPerformancePage, OvercookedPage, \
    EnvRewardModificationPage, DecisionTreeCreationPage, GUIPageWithTwoTreeChoices, GUIPageWithImage, \
    GUIPageWithTextAndURL, GUIPageWithSingleTree
from ipm.gui.policy_utils import get_idct, finetune_model
from ipm.models.bc_agent import get_human_bc_partner
from ipm.overcooked.overcooked_multi import OvercookedRoundRobinEnv, OvercookedPlayWithFixedPartner
from ipm.models.decision_tree import DecisionTree

class EnvWrapper:
    def __init__(self, layout, data_folder):
        # wrapping this up in a class so that we can easily change the reward function
        # this acts like a pointer
        self.multipliers = [1, 1, 1]
        teammate_paths = os.path.join('data', layout, 'self_play_training_models')
        self.ego_idx = 0
        self.alt_idx = 1
        self.layout = layout
        self.data_folder = data_folder
        traj_directories = os.path.join('trajectories')
        self.behavioral_model, self.bc_partner = get_human_bc_partner(traj_directories, layout, self.alt_idx,
                                                                      get_human_policy_estimator=True)
        self.eval_partner = get_human_bc_partner(traj_directories, layout, self.ego_idx)
        self.rewards = []
        # TODO: reward shown on chosen page can be inaccurate if we go with the prior policy
        # this probably won't matter if we use human policy estimation to compute rewards for each tree
        self.train_env = None # for optimization conditions we want to use this
        self.team_env = OvercookedPlayWithFixedPartner(partner=self.bc_partner, layout_name=layout,
                                                       behavioral_model=self.behavioral_model,
                                                       reduced_state_space_ego=True, reduced_state_space_alt=False,
                                                       use_skills_ego=True, use_skills_alt=False, failed_skill_rew=0)
        self.save_chosen_as_prior = False
        # self.team_env = OvercookedRoundRobinEnv(teammate_locations=teammate_paths, layout_name=layout, seed_num=0,
        #                                         reduced_state_space_ego=True, reduced_state_space_alt=False,
        #                                        use_skills_ego=True, use_skills_alt=False, failed_skill_rew=0)
        self.env = self.team_env # need to change to train env
        # self.decision_tree = DecisionTree.from_sklearn(self.bc_partner.model,
        #                                                self.team_env.n_reduced_feats,
        #                                                self.team_env.n_actions_ego)
        self.prior_policy_path = os.path.join('data', 'prior_tree_policies',
                                         layout, 'policy.pkl')
        try:
            with open(self.prior_policy_path, 'rb') as inp:
                self.decision_tree = pickle.load(inp)
        except:
            import pickle5 as p
            with open(self.prior_policy_path, 'rb') as inp:
                self.decision_tree = p.load(inp)

        if self.decision_tree.num_actions != self.team_env.n_actions_ego or \
                self.decision_tree.num_vars != self.team_env.n_reduced_feats:
            # then just use a random policy
            self.decision_tree = DecisionTree(num_vars=self.team_env.n_reduced_feats,
                                              num_actions=self.team_env.n_actions_ego,
                                              depth=1)
            self.save_chosen_as_prior = True

    def initialize_env(self):
        # we keep track of the reward function that may change
        self.team_env.set_env(self.multipliers[0], self.multipliers[1], self.multipliers[2])
        # self.train_env.set_env(self.multipliers[0], self.multipliers[1], self.multipliers[2])


class SettingsWrapper:
    def __init__(self):
        self.zoom = 1
        self.old_zoom = 1
        self.max_zoom = 3
        self.min_zoom = 1
        self.width, self.height = 1920, 1080
        self.offset_x, self.offset_y = 0, 0
        self.absolute_x, self.absolute_y = self.width // 2, self.height // 2
        self.options_menus_per_domain = {-1: [], 0: [], 1: [], 2: []}

    def check_if_options_menu_open(self, domain_idx) -> (bool, int):
        for i, menu_open in enumerate(self.options_menus_per_domain[domain_idx]):
            if menu_open:
                return True, i
        return False, -1

    def zoom_out(self):
        self.old_zoom = self.zoom
        self.zoom = max(self.zoom - 0.1, self.min_zoom)
        assert self.max_zoom >= self.zoom >= self.min_zoom

    def zoom_in(self):
        self.old_zoom = self.zoom
        self.zoom = min(self.zoom + 0.1, self.max_zoom)
        assert self.max_zoom >= self.zoom >= self.min_zoom


def get_next_user_id():
    # get the next user id
    # we can get this info by looking at the folders in the data/experiments/conditions folder
    # need to iterate through each folder in each condition and get the max number
    latest_user_id = 0

    if not os.path.exists(os.path.join('data', 'experiments')):
        os.mkdir(os.path.join('data', 'experiments'))

    for condition in os.listdir(os.path.join('data', 'experiments')):
        if not condition.startswith('.'):
            for user_folder in os.listdir(os.path.join('data', 'experiments', condition)):
                if not user_folder.startswith('.'):
                    user_id = int(user_folder.split('_')[-1])
                    latest_user_id = max(latest_user_id, int(user_id))
    return latest_user_id + 1


class MainExperiment:
    def __init__(self, group: str):
        self.user_id = get_next_user_id()

        conditions = ['human_modifies_tree',
                  'optimization_via_rl_or_ga',
                  'optimization_via_rl_or_ga_while_modifying_reward',
                  'recommends_rule',
                  'fcp',
                  'ha_ppo',
                  'no_modification_bb',
                  'no_modification_interpretable']

        self.condition = conditions.index(group)
        self.data_folder = os.path.join('data', 'experiments', conditions[self.condition], 'user_' + str(self.user_id))

        self.domain_names = {-1:'tutorial', 0:'forced_coordination', 1:'two_rooms', 2:'two_rooms_narrow'} # does not include tutorial

        pygame.init()
        self.pages = []
        self.current_page = 0
        self.settings = SettingsWrapper()
        # self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA | pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((self.settings.width, self.settings.height), pygame.SRCALPHA)# | pygame.FULLSCREEN)
                                              #pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen.fill('white')

        env_wrapper_tutorial = EnvWrapper(layout='tutorial', data_folder=self.data_folder)
        env_wrapper_easy = EnvWrapper(layout='forced_coordination', data_folder=self.data_folder)
        env_wrapper_med = EnvWrapper(layout='two_rooms', data_folder=self.data_folder)
        env_wrapper_hard = EnvWrapper(layout='two_rooms_narrow', data_folder=self.data_folder)

        main_page = GUIPageCenterText(self.screen, 'Welcome to our experiment.', 24,
                                       bottom_left_button=False, bottom_right_button=True,
                                       bottom_right_fn=self.next_page)

        proceed_page = GUIPageCenterText(self.screen, 'Are you ready to proceed? (Press next when signed consent form)', 24,
                                       bottom_left_button=False, bottom_right_button=True,
                                       bottom_left_fn=None, bottom_right_fn=self.next_page)

        survey_urls = ['https://gatech.co1.qualtrics.com/jfe/form/SV_3I7z5yu8uilrc5o',
                       'https://gatech.co1.qualtrics.com/jfe/form/SV_6RraiNzIohdWYCO']

        presurveys_page = GUIPageWithTextAndURL(screen=self.screen,
                                            text='Please take these surveys so that we have more info about your background and personality.',
                                            urls=survey_urls,
                                            font_size=24,
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=False, bottom_right_fn=self.next_page)

        self.pages.append(main_page)
        self.pages.append(presurveys_page)
        self.pages.append(proceed_page)

        oc_tutorial_page = GUIPageWithImage(self.screen, 'Overcooked Gameplay Overview', 'OvercookedTutorial.png',
                                        bottom_left_button=False, bottom_right_button=True,
                                        bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(oc_tutorial_page)

        dt_tutorial_page = GUIPageWithImage(self.screen, 'Decision Tree Modification Overview', 'DTTutorial.png',
                                        bottom_left_button=False, bottom_right_button=True,
                                        bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(dt_tutorial_page)

        proceed_dt_page = GUIPageCenterText(self.screen, 'You will now play a practice round with your teammate. '
                                                         'Afterwards, you may modify it as you wish.', 36,
                                       bottom_left_button=False, bottom_right_button=True,
                                       bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(proceed_dt_page)

        self.tutorial_tree_page = DecisionTreeCreationPage(env_wrapper_tutorial, env_wrapper_tutorial.layout, -1, self.settings,
                                                  screen=self.screen,
                                                  X=self.settings.width, Y=self.settings.height,
                                                  bottom_left_button=False, bottom_right_button=True,
                                                  bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.easy_tree_page = DecisionTreeCreationPage(env_wrapper_easy, env_wrapper_easy.layout, 0, self.settings,
                                                  screen=self.screen,
                                                  X=self.settings.width, Y=self.settings.height,
                                                  bottom_left_button=False, bottom_right_button=True,
                                                  bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.med_tree_page = DecisionTreeCreationPage(env_wrapper_med, env_wrapper_med.layout, 1, self.settings,
                                                  screen=self.screen,
                                                  X=self.settings.width, Y=self.settings.height,
                                                  bottom_left_button=False, bottom_right_button=True,
                                                  bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.hard_tree_page = DecisionTreeCreationPage(env_wrapper_hard,  env_wrapper_hard.layout, 2, self.settings,
                                                  screen=self.screen,
                                                  X=self.settings.width, Y=self.settings.height,
                                                  bottom_left_button=False, bottom_right_button=True,
                                                  bottom_left_fn=None, bottom_right_fn=self.next_page)

        tutorial_env_page = OvercookedPage(self.screen, env_wrapper_tutorial, self.tutorial_tree_page, layout=env_wrapper_tutorial.layout, text=' ',
                                  font_size=24,
                                  bottom_left_button=False, bottom_right_button=True,
                                  bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(tutorial_env_page)
        self.pages.append(self.tutorial_tree_page)
        self.pages.append(tutorial_env_page)

        self.tut_tree_choice_page = GUIPageWithTwoTreeChoices(self.screen, tree_page=self.tutorial_tree_page,
                                                              env_wrapper=env_wrapper_tutorial,
                                                              font_size=24,
                                                              bottom_left_button=True,
                                                              bottom_right_button=True,
                                                              bottom_left_fn=self.pick_initial_policy,
                                                              bottom_right_fn=self.pick_final_policy)
        self.tut_initial_tree_page = GUIPageWithSingleTree(True, self.screen, tree_page=self.tutorial_tree_page,
                                                           env_wrapper=env_wrapper_tutorial,
                                                           font_size=24,
                                                           bottom_left_button=True,
                                                           bottom_right_button=True,
                                                           bottom_left_fn=self.pick_initial_policy,
                                                           bottom_right_fn=self.pick_final_policy)
        self.tut_final_tree_page = GUIPageWithSingleTree(False, self.screen, tree_page=self.tutorial_tree_page,
                                                         env_wrapper=env_wrapper_tutorial,
                                                         font_size=24,
                                                         bottom_left_button=True,
                                                         bottom_right_button=True,
                                                         bottom_left_fn=self.pick_initial_policy,
                                                         bottom_right_fn=self.pick_final_policy)
        self.pages.append(self.tut_initial_tree_page)
        self.pages.append(self.tut_final_tree_page)
        self.pages.append(self.tut_tree_choice_page)

        n_iterations = 2

        for env_wrapper in [env_wrapper_easy, env_wrapper_med, env_wrapper_hard]:

            if env_wrapper.layout == 'forced_coordination':
                tree_page = self.easy_tree_page
                self.easy_tree_choice_page = GUIPageWithTwoTreeChoices(self.screen, tree_page=tree_page,
                                                                       env_wrapper=env_wrapper,
                                                                       font_size=24,
                                                                       bottom_left_button=True,
                                                                       bottom_right_button=True,
                                                                       bottom_left_fn=self.pick_initial_policy,
                                                                       bottom_right_fn=self.pick_final_policy)
                self.easy_initial_tree_page = GUIPageWithSingleTree(True, self.screen, tree_page=tree_page,
                                                                       env_wrapper=env_wrapper,
                                                                       font_size=24,
                                                                       bottom_left_button=True,
                                                                       bottom_right_button=True,
                                                                       bottom_left_fn=self.pick_initial_policy,
                                                                       bottom_right_fn=self.pick_final_policy)
                self.easy_final_tree_page = GUIPageWithSingleTree(False, self.screen, tree_page=tree_page,
                                                                       env_wrapper=env_wrapper,
                                                                       font_size=24,
                                                                       bottom_left_button=True,
                                                                       bottom_right_button=True,
                                                                       bottom_left_fn=self.pick_initial_policy,
                                                                       bottom_right_fn=self.pick_final_policy)
                tree_choice_page = self.easy_tree_choice_page
                tree_initial_page = self.easy_initial_tree_page
                tree_final_page = self.easy_final_tree_page
            elif env_wrapper.layout == 'two_rooms':
                tree_page = self.med_tree_page
                self.med_tree_choice_page = GUIPageWithTwoTreeChoices(self.screen, tree_page=tree_page,
                                                                       env_wrapper=env_wrapper,
                                                                       font_size=24,
                                                                       bottom_left_button=True,
                                                                       bottom_right_button=True,
                                                                       bottom_left_fn=self.pick_initial_policy,
                                                                       bottom_right_fn=self.pick_final_policy)
                self.med_initial_tree_page = GUIPageWithSingleTree(True, self.screen, tree_page=tree_page,
                                                                       env_wrapper=env_wrapper,
                                                                       font_size=24,
                                                                       bottom_left_button=True,
                                                                       bottom_right_button=True,
                                                                       bottom_left_fn=self.pick_initial_policy,
                                                                       bottom_right_fn=self.pick_final_policy)
                self.med_final_tree_page = GUIPageWithSingleTree(False, self.screen, tree_page=tree_page,
                                                                       env_wrapper=env_wrapper,
                                                                       font_size=24,
                                                                       bottom_left_button=True,
                                                                       bottom_right_button=True,
                                                                       bottom_left_fn=self.pick_initial_policy,
                                                                       bottom_right_fn=self.pick_final_policy)
                tree_choice_page = self.med_tree_choice_page
                tree_initial_page = self.med_initial_tree_page
                tree_final_page = self.med_final_tree_page
            elif env_wrapper.layout == 'two_rooms_narrow':
                tree_page = self.hard_tree_page
                self.hard_tree_choice_page = GUIPageWithTwoTreeChoices(self.screen, tree_page=tree_page,
                                                                       env_wrapper=env_wrapper,
                                                                       font_size=24,
                                                                       bottom_left_button=True,
                                                                       bottom_right_button=True,
                                                                       bottom_left_fn=self.pick_initial_policy,
                                                                       bottom_right_fn=self.pick_final_policy)
                self.hard_initial_tree_page = GUIPageWithSingleTree(True, self.screen, tree_page=tree_page,
                                                                       env_wrapper=env_wrapper,
                                                                       font_size=24,
                                                                       bottom_left_button=False,
                                                                       bottom_right_button=True,
                                                                       bottom_left_fn=self.pick_initial_policy,
                                                                       bottom_right_fn=self.pick_final_policy)
                self.hard_final_tree_page = GUIPageWithSingleTree(False, self.screen, tree_page=tree_page,
                                                                       env_wrapper=env_wrapper,
                                                                       font_size=24,
                                                                       bottom_left_button=False,
                                                                       bottom_right_button=True,
                                                                       bottom_left_fn=self.pick_initial_policy,
                                                                       bottom_right_fn=self.pick_final_policy)
                tree_choice_page = self.hard_tree_choice_page
                tree_initial_page = self.hard_initial_tree_page
                tree_final_page = self.hard_final_tree_page
            else:
                raise ValueError('Invalid layout')

            env_page = OvercookedPage(self.screen, env_wrapper, tree_page,
                                      layout=env_wrapper.layout, text=' ',
                                      font_size=24,
                                      bottom_left_button=False, bottom_right_button=True,
                                      bottom_left_fn=None, bottom_right_fn=self.next_page)

            survey = GUIPageCenterText(self.screen, 'Please take survey. Press next when finished', 24,
                                       bottom_left_button=False, bottom_right_button=True,
                                       bottom_left_fn=False, bottom_right_fn=self.next_page,
                                       nasa_tlx=True)

            survey_urls = ['https://gatech.co1.qualtrics.com/jfe/form/SV_bCIZ8mjqcOtKveS',
                           'https://gatech.co1.qualtrics.com/jfe/form/SV_ezZAMpSbcQ3Vx9s',
                           'https://gatech.co1.qualtrics.com/jfe/form/SV_3gCgLUCf2sRNafA']

            survey_qual = GUIPageWithTextAndURL(screen=self.screen,
                                            text='Please take the qualtrics survey provided by the researcher.',
                                            urls=survey_urls,
                                            font_size=24,
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=False, bottom_right_fn=self.next_page)

            for i in range(n_iterations):

                if i == 0:
                    self.pages.append(env_page)
                self.pages.append(tree_page)
                self.pages.append(env_page)
                self.pages.append(tree_initial_page)
                self.pages.append(tree_final_page)
                self.pages.append(tree_choice_page)
                self.pages.append(survey)
            self.pages.append(survey_qual) # this will be outside of the iterations loop

            # env_reward_modification_page = EnvRewardModificationPage(env_wrapper, screen=self.screen, settings=self.settings,
            #                                                          X=self.settings.width, Y=self.settings.height, font_size=24,
            #                                         bottom_left_button=True, bottom_right_button=True,
            #                                         bottom_left_fn=self.previous_page, bottom_right_fn=self.next_page)

            #if group == 'reward_modification':
            #    self.pages.append(env_reward_modification_page)

        thank_you_page = GUIPageCenterText(self.screen, 'Thank you for participating in our study', 24,
                                           bottom_left_button=False, bottom_right_button=False,
                                           bottom_left_fn=False, bottom_right_fn=False,
                                           nasa_tlx=False)
        self.pages.append(thank_you_page)
        self.previous_domain = -1
        self.current_domain = -1
        self.current_iteration = 0
        folder = os.path.join(self.data_folder, self.domain_names[self.current_domain])
        if not os.path.exists(folder):
            os.makedirs(folder)

    def save_rewards_for_domain(self, domain_idx):
        folder = os.path.join(self.data_folder, self.domain_names[domain_idx])
        filepath = os.path.join(folder, 'rewards.txt')
        with open(filepath, 'w') as f:
            if domain_idx == -1:
                tree_page = self.tutorial_tree_page
            elif domain_idx == 0:
                tree_page = self.easy_tree_page
            elif domain_idx == 1:
                tree_page = self.med_tree_page
            elif domain_idx == 2:
                tree_page = self.hard_tree_page
            f.write(str(tree_page.env_wrapper.rewards))

    def next_domain(self):
        # save rewards to file
        self.save_rewards_for_domain(domain_idx=self.previous_domain)
        self.previous_domain = self.current_domain
        new_folder = os.path.join(self.data_folder, self.domain_names[self.current_domain])
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        self.current_iteration = 0

    def save_tree(self, initial=True):
        if initial:
            filename = 'initial_tree.png'
        else:
            filename = 'final_tree.png'
        pygame.image.save(self.screen, filename)
        # save to current data folder / domain / current_iteration / initial_tree.png
        folder = os.path.join(self.data_folder, self.domain_names[self.current_domain],
                              'iteration_' + str(self.current_iteration))
        if not os.path.exists(folder):
            os.makedirs(folder)
        imagepath = os.path.join(folder, filename)
        pygame.image.save(self.screen, imagepath)

    def save_initial_tree(self):
        self.save_tree(initial=True)
        self.saved_first_tree = True

    def save_final_tree(self):
        self.save_tree(initial=False)

    def new_tree_page(self, domain_idx):
        self.current_domain = domain_idx
        if self.current_domain != self.previous_domain:
            self.next_domain()
        else:
            self.current_iteration += 1
        if domain_idx == 0:
            self.easy_tree_choice_page.loaded_images = False
        elif domain_idx == 1:
            self.med_tree_choice_page.loaded_images = False
        elif domain_idx == 2:
            self.hard_tree_choice_page.loaded_images = False
        self.saved_first_tree = False

    def save_times(self):
        output_file = os.path.join(self.data_folder, 'times.csv')
        # save times and pages names to csv
        with open(output_file, 'w') as outp:
            outp.write('page,time\n')
            for i in range(len(self.pages_names)):
                outp.write(f'{self.pages_names[i]}, {self.times[i]}\n')

    def next_page(self):
        # record time spent in prior page
        self.times.append(time.time() - self.page_start_time)
        self.page_start_time = time.time()

        # save final tree if the prior page is of type DecisionTreeCreationPage
        if self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage':
            self.save_final_tree()

        self.pages[self.current_page].hide()
        self.current_page += 1

        if self.easy_tree_page == self.pages[self.current_page]:
            self.new_tree_page(domain_idx=0)
        elif self.med_tree_page == self.pages[self.current_page]:
            self.new_tree_page(domain_idx=1)
        elif self.hard_tree_page == self.pages[self.current_page]:
            self.new_tree_page(domain_idx=2)

        if self.current_page == len(self.pages) - 1:
            self.save_times()
            self.save_rewards_for_domain(domain_idx=self.current_domain)
            self.pages[self.current_page].show()
        else:
            self.pages_names.append(self.pages[self.current_page].__class__.__name__)
            self.showed_nasa_tlx = False
            self.saved_first_tree = False
            self.pages[self.current_page].show()

    def previous_page(self):
        self.pages[self.current_page].hide()
        self.current_page -= 1
        self.pages[self.current_page].show()

    def pick_initial_policy(self):
        if self.current_domain == 0:
            initial_policy = self.easy_tree_page.decision_tree_history[0]
            self.easy_tree_page.reset_initial_policy(initial_policy)
        elif self.current_domain == 1:
            initial_policy = self.med_tree_page.decision_tree_history[0]
            self.med_tree_page.reset_initial_policy(initial_policy)
        elif self.current_domain == 2:
            initial_policy = self.hard_tree_page.decision_tree_history[0]
            self.hard_tree_page.reset_initial_policy(initial_policy)
        self.next_page()

    def update_prior_policy(self, tree_page):
        final_policy = tree_page.decision_tree_history[-1]
        path = tree_page.env_wrapper.prior_policy_path
        with open(path, 'wb') as outp:
            pickle.dump(final_policy, outp, pickle.HIGHEST_PROTOCOL)


    def pick_final_policy(self):
        if self.current_domain == -1:
            if self.tutorial_tree_page.env_wrapper.save_chosen_as_prior:
                self.update_prior_policy(self.tutorial_tree_page)
        if self.current_domain == 0:
            final_policy = self.easy_tree_page.decision_tree_history[-1]
            self.easy_tree_page.reset_initial_policy(final_policy)
            if self.easy_tree_page.env_wrapper.save_chosen_as_prior:
                self.update_prior_policy(self.easy_tree_page)
        elif self.current_domain == 1:
            final_policy = self.med_tree_page.decision_tree_history[-1]
            self.med_tree_page.reset_initial_policy(final_policy)
            if self.med_tree_page.env_wrapper.save_chosen_as_prior:
                self.update_prior_policy(self.med_tree_page)
        elif self.current_domain == 2:
            final_policy = self.hard_tree_page.decision_tree_history[-1]
            self.hard_tree_page.reset_initial_policy(final_policy)
            if self.hard_tree_page.env_wrapper.save_chosen_as_prior:
                self.update_prior_policy(self.hard_tree_page)
        self.next_page()

    def process_zoom(self):
        # create pygame subsurface
        wnd_w, wnd_h = self.screen.get_size()
        zoom_size = (round(wnd_w / self.settings.zoom), round(wnd_h / self.settings.zoom))
        # when fully zoomed in, make sure it is in bounds
        x = self.settings.absolute_x
        y = self.settings.absolute_y
        x = (x + self.settings.offset_x) // self.settings.zoom
        y = (y + self.settings.offset_y) // self.settings.zoom

        # prevent any black borders
        x = max(x, zoom_size[0] // 2)
        y = max(y, zoom_size[1] // 2)
        x = min(x, wnd_w - zoom_size[0] // 2)
        y = min(y, wnd_h - zoom_size[1] // 2)

        if self.settings.zoom == 1:
            x = wnd_w // 2
            y = wnd_h // 2

        self.settings.absolute_x = x
        self.settings.absolute_y = y
        self.settings.offset_x = int(self.settings.absolute_x * self.settings.zoom) - self.settings.width // 2
        self.settings.offset_y = int(self.settings.absolute_y * self.settings.zoom) - self.settings.height // 2

        zoom_area = pygame.Rect(0, 0, *zoom_size)
        # if self.settings.zoom == previous_zoom:
        #     x, y = wnd_w // 2, wnd_h // 2
        zoom_area.center = (x, y)
        zoom_surf = pygame.Surface(zoom_area.size)
        zoom_surf.blit(self.screen, (0, 0), zoom_area)
        zoom_surf = pygame.transform.scale(zoom_surf, (wnd_w, wnd_h))
        self.screen.blit(zoom_surf, (0, 0))

    def launch(self):
        self.saved_first_tree = False
        self.showed_nasa_tlx = False

        pygame.init()
        clock = pygame.time.Clock()

        self.is_running = True
        self.pages[0].show()
        pygame.display.flip()

        # start recording time, so we can get seconds spent in each page
        self.page_start_time = time.time()
        self.times = []
        self.pages_names = [self.pages[self.current_page].__class__.__name__]

        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                    break

                if self.pages[self.current_page].__class__.__name__ == 'GUIPageWithTwoTreeChoices':
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 4:
                            self.settings.zoom_in()
                        elif event.button == 5:
                            self.settings.zoom_out()
                # if event.type == pygame.KEYDOWN:
                #     # if scroll in, zoom
                #     if event.key == pygame.K_o:
                #         self.settings.zoom_in()
                #     elif event.key == pygame.K_p:
                #         self.settings.zoom_out()
                self.is_running = self.pages[self.current_page].process_event(event)
                if self.is_running is False:
                    break
            self.pages[self.current_page].process_standby()

            # zoom in here
            # if self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage' or \
            #         self.pages[self.current_page].__class__.__name__ == 'GUIPageWithTwoTreeChoices':
            if self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage' :
                if self.settings.zoom != self.settings.old_zoom:
                    self.process_zoom()
            pygame.display.update()
            clock.tick(30)

            if not self.saved_first_tree and self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage':
                self.save_initial_tree()

            if not self.showed_nasa_tlx and self.pages[self.current_page].__class__.__name__ == 'GUIPageCenterText' \
                    and self.pages[self.current_page].nasa_tlx:
                self.showed_nasa_tlx = True
                run_gui(self.user_id, self.condition, self.current_domain)
