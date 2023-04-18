import os
import time

import pickle5 as pickle
import pygame
import torch
from ipm.gui.experiment_gui_utils import SettingsWrapper, get_next_user_id, process_zoom
from ipm.gui.nasa_tlx import run_gui
from ipm.gui.pages import GUIPageCenterText, OvercookedPage, \
    DecisionTreeCreationPage, GUIPageWithTwoTreeChoices, GUIPageWithImage, \
    GUIPageWithTextAndURL, GUIPageWithSingleTree, EnvRewardModificationPage
from ipm.models.bc_agent import StayAgent
from ipm.models.decision_tree import sparse_ddt_to_decision_tree
from ipm.overcooked.overcooked_envs import OvercookedPlayWithFixedPartner
from models.human_model import HumanModel
from models.robot_model import RobotModel
from stable_baselines3 import PPO


class EnvWrapper:
    def __init__(self, layout, data_folder):
        # wrapping this up in a class so that we can easily change the reward function
        # this acts like a pointer
        self.multipliers = [1, 1, 1]
        self.ego_idx = 0
        self.alt_idx = 1
        self.layout = layout
        self.data_folder = data_folder
        self.rewards = []
        self.save_chosen_as_prior = False
        self.latest_save_file = None

        dummy_env = OvercookedPlayWithFixedPartner(partner=StayAgent(), layout_name=layout,
                                                   behavioral_model='dummy',
                                                   reduced_state_space_ego=True, reduced_state_space_alt=True,
                                                   use_skills_ego=True, use_skills_alt=True)

        self.initial_policy_path = os.path.join('data', 'prior_tree_policies', layout + '.tar')
        intent_model_path = os.path.join('data', 'intent_models', layout + '.pt')

        model = PPO("MlpPolicy", dummy_env)
        weights = torch.load(self.initial_policy_path)
        model.policy.load_state_dict(weights['ego_state_dict'])
        human_ppo_policy = model.policy
        self.human_policy = HumanModel(layout, human_ppo_policy)

        input_dim = dummy_env.observation_space.shape[0]
        output_dim = dummy_env.n_actions_alt

        self.robot_policy = RobotModel(layout=layout,
                                       idct_policy_filepath=self.initial_policy_path,
                                       human_policy=self.human_policy,
                                       intent_model_filepath=intent_model_path,
                                       input_dim=input_dim,
                                       output_dim=output_dim)

        self.current_policy, tree_info = sparse_ddt_to_decision_tree(self.robot_policy.robot_idct_policy,
                                                                     self.robot_policy.env)
        self.intent_model = self.robot_policy.intent_model

    def initialize_env(self):
        # we keep track of the reward function that may change
        self.robot_policy.env.set_env(placing_in_pot_multiplier=self.multipliers[0],
                                      dish_pickup_multiplier=self.multipliers[1],
                                      soup_pickup_multiplier=self.multipliers[2])


class MainExperiment:
    def __init__(self, condition: str, conditions: list, disable_surveys: bool = False):
        self.user_id = get_next_user_id()
        self.condition_num = conditions.index(condition) + 1
        self.data_folder = os.path.join('data',
                                        'experiments',
                                        condition,
                                        'user_' + str(self.user_id))

        self.domain_names = ['tutorial', 'forced_coordination', 'two_rooms', 'two_rooms_narrow']
        for domain_name in self.domain_names:
            folder = os.path.join(self.data_folder, domain_name)
            if not os.path.exists(folder):
                os.makedirs(folder)

        pygame.init()
        self.settings = SettingsWrapper()
        self.disable_surveys = disable_surveys
        # self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA | pygame.FULLSCREEN | pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((self.settings.width, self.settings.height),
                                              pygame.SRCALPHA)  # | pygame.FULLSCREEN)
        self.screen.fill('white')
        self.setup_pages()

    def add_preliminary_pages(self):
        main_page = GUIPageCenterText(self.screen, 'Welcome to our experiment.', 24,
                                      bottom_left_button=False, bottom_right_button=True,
                                      bottom_right_fn=self.next_page)

        proceed_page = GUIPageCenterText(self.screen, 'Are you ready to proceed? (Press next when signed consent form)',
                                         24,
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
        if not self.disable_surveys:
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

    def setup_survey_misc_pages(self):
        self.survey_page = GUIPageCenterText(self.screen, 'Please take survey. Press next when finished', 24,
                                             bottom_left_button=False, bottom_right_button=True,
                                             bottom_left_fn=False, bottom_right_fn=self.next_page,
                                             nasa_tlx=True)

        survey_urls = ['https://gatech.co1.qualtrics.com/jfe/form/SV_bCIZ8mjqcOtKveS',
                       'https://gatech.co1.qualtrics.com/jfe/form/SV_ezZAMpSbcQ3Vx9s',
                       'https://gatech.co1.qualtrics.com/jfe/form/SV_3gCgLUCf2sRNafA']

        self.survey_qual = GUIPageWithTextAndURL(screen=self.screen,
                                                 text='Please take the qualtrics survey provided by the researcher.',
                                                 urls=survey_urls,
                                                 font_size=24,
                                                 bottom_left_button=False, bottom_right_button=True,
                                                 bottom_left_fn=False, bottom_right_fn=self.next_page)

        self.thank_you_page = GUIPageCenterText(self.screen, 'Thank you for participating in our study', 24,
                                                bottom_left_button=False, bottom_right_button=False,
                                                bottom_left_fn=False, bottom_right_fn=False,
                                                nasa_tlx=False)

    def setup_main_pages(self):
        self.modify_tree_pages = []
        self.env_pages = []
        self.two_choices_pages = []
        self.initial_tree_show_pages = []
        self.next_tree_show_pages = []
        self.reward_modify_pages = []
        for i, env_wrapper in enumerate(self.env_wrappers):
            tree_page = DecisionTreeCreationPage(env_wrapper=env_wrapper,
                                                 layout_name=env_wrapper.layout,
                                                 domain_idx=i,
                                                 settings_wrapper=self.settings,
                                                 screen=self.screen,
                                                 X=self.settings.width, Y=self.settings.height,
                                                 bottom_left_button=False, bottom_right_button=True,
                                                 bottom_left_fn=None, bottom_right_fn=self.next_page)
            self.modify_tree_pages.append(tree_page)

            env_page = OvercookedPage(self.screen, env_wrapper, tree_page,
                                      layout=env_wrapper.layout, text=' ',
                                      font_size=24,
                                      bottom_left_button=False, bottom_right_button=True,
                                      bottom_left_fn=None, bottom_right_fn=self.next_page)
            self.env_pages.append(env_page)

            two_choices_page = GUIPageWithTwoTreeChoices(self.screen,
                                                         tree_page=tree_page,
                                                         env_wrapper=env_wrapper,
                                                         font_size=24,
                                                         bottom_left_button=True,
                                                         bottom_right_button=True,
                                                         bottom_left_fn=self.pick_initial_policy,
                                                         bottom_right_fn=self.pick_final_policy)
            self.two_choices_pages.append(two_choices_page)

            initial_tree_show_page = GUIPageWithSingleTree(True, self.screen, tree_page=tree_page,
                                                           env_wrapper=env_wrapper,
                                                           font_size=24,
                                                           bottom_left_button=True,
                                                           bottom_right_button=True,
                                                           bottom_left_fn=self.pick_initial_policy,
                                                           bottom_right_fn=self.pick_final_policy)
            self.initial_tree_show_pages.append(initial_tree_show_page)

            next_tree_show_page = GUIPageWithSingleTree(False, self.screen, tree_page=tree_page,
                                                        env_wrapper=env_wrapper,
                                                        font_size=24,
                                                        bottom_left_button=True,
                                                        bottom_right_button=True,
                                                        bottom_left_fn=self.pick_initial_policy,
                                                        bottom_right_fn=self.pick_final_policy)
            self.next_tree_show_pages.append(next_tree_show_page)

            env_reward_modification_page = EnvRewardModificationPage(env_wrapper, screen=self.screen,
                                                                     settings=self.settings,
                                                                     X=self.settings.width, Y=self.settings.height,
                                                                     font_size=24,
                                                                     bottom_left_button=True, bottom_right_button=True,
                                                                     bottom_left_fn=self.previous_page,
                                                                     bottom_right_fn=self.next_page)
            self.reward_modify_pages.append(env_reward_modification_page)


    def setup_pages(self):

        self.pages = []
        self.current_page = 0

        self.add_preliminary_pages()
        self.setup_survey_misc_pages()

        self.env_wrappers = [EnvWrapper(layout=layout, data_folder=self.data_folder) for layout in self.domain_names]
        self.setup_main_pages()

        only_show_tree_no_modify = self.condition_num == 2 or self.condition_num == 5
        if only_show_tree_no_modify:
            for tree_page in self.modify_tree_pages:
                tree_page.frozen = True
            self.frozen_pages = self.modify_tree_pages

        n_iterations = 2
        for layout_idx in range(1, len(self.env_wrappers)):
            current_n_iterations = n_iterations if layout_idx > 0 else 1
            self.pages.append(self.env_pages[layout_idx])
            for i in range(current_n_iterations):
                if self.condition_num == 1:  # modify tree
                    self.pages.append(self.modify_tree_pages[layout_idx])
                elif self.condition_num == 2:  # optimization
                    self.pages.append(self.frozen_pages[layout_idx])
                elif self.condition_num == 3:  # reward modification
                    self.pages.append(self.reward_modify_pages[layout_idx])
                elif self.condition_num == 4:  # not intepretable black-box
                    pass
                elif self.condition_num == 5:  # intepretable black-box
                    self.pages.append(self.frozen_pages[layout_idx])
                self.pages.append(self.env_pages[layout_idx])

                if self.condition_num < 4:
                    self.pages.append(self.initial_tree_show_pages[layout_idx])
                    self.pages.append(self.next_tree_show_pages[layout_idx])
                    self.pages.append(self.two_choices_pages[layout_idx])
                if layout_idx > 0 and not self.disable_surveys:
                    self.pages.append(self.survey_page)
            if layout_idx > 0 and not self.disable_surveys:
                self.pages.append(self.survey_qual)
        self.pages.append(self.thank_you_page)

        self.previous_domain = 0
        self.current_domain = 0
        self.current_iteration = 0

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

            if self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage':
                if self.settings.zoom != self.settings.old_zoom:
                    process_zoom(self.screen, self.settings)
            pygame.display.update()
            clock.tick(30)

            if not self.saved_first_tree and \
                    self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage':
                self.save_initial_tree()

            if not self.showed_nasa_tlx and self.pages[self.current_page].__class__.__name__ == 'GUIPageCenterText' \
                    and self.pages[self.current_page].nasa_tlx:
                self.showed_nasa_tlx = True
                run_gui(self.user_id, self.condition_num, self.current_domain)

    def save_rewards_for_domain(self, domain_idx):
        folder = os.path.join(self.data_folder, self.domain_names[domain_idx])
        filepath = os.path.join(folder, 'rewards.txt')
        with open(filepath, 'w') as f:
            tree_page = self.modify_tree_pages[domain_idx]
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
        folder = os.path.join(self.data_folder, self.domain_names[self.current_domain],
                              'iteration_' + str(self.current_iteration))
        if not os.path.exists(folder):
            os.makedirs(folder)
        imagepath = os.path.join(folder, filename)
        pygame.image.save(self.screen, imagepath)
        # TODO: also save the tree as a pytorch model

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
        self.two_choices_pages[domain_idx].loaded_images = False
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

        # TODO: properly add this code where needed
        # if self.pages[self.current_page].__class__.__name__ == 'SurveyPage' and self.condition_num == 2:
        #     data_file = self.env_wrappers[self.current_domain].latest_save_file
        #     self.env_wrappers[self.current_domain].robot_policy.finetune_robot_idct_policy(recent_data_file=data_file)
        #     self.env_wrappers[self.current_domain].robot_policy.translate_recent_data_to_labels(recent_data_loc=data_file)
        #     self.env_wrappers[self.current_domain].robot_policy.finetune_intent_model()
        #     self.env_wrappers[self.current_domain].human_policy.translate_recent_data_to_labels(recent_data_loc=data_file)
        #     self.env_wrappers[self.current_domain].human_policy.finetune_human_ppo_policy()

        self.pages[self.current_page].hide()
        self.current_page += 1

        if type(self.pages[self.current_page]) == DecisionTreeCreationPage:
            self.new_tree_page(domain_idx=self.current_domain)

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
        modify_tree_page = self.modify_tree_pages[self.current_domain]
        initial_policy = modify_tree_page.decision_tree_history[0]
        modify_tree_page.reset_initial_policy(initial_policy)
        self.next_page()

    def update_prior_policy(self, tree_page):
        final_policy = tree_page.decision_tree_history[-1]
        path = tree_page.env_wrapper.initial_policy_path
        with open(path, 'wb') as outp:
            pickle.dump(final_policy, outp, pickle.HIGHEST_PROTOCOL)

    def pick_final_policy(self):
        tree_page = self.modify_tree_pages[self.current_domain]
        final_policy = tree_page.decision_tree_history[-1]
        tree_page.reset_initial_policy(final_policy)
        if tree_page.env_wrapper.save_chosen_as_prior:
            self.update_prior_policy(tree_page)
        self.next_page()
