import copy
import os
import shutil
import time

import pickle
import pygame
import torch
from ipm.gui.experiment_gui_utils import SettingsWrapper, get_next_user_id, process_zoom
from ipm.gui.nasa_tlx import run_gui
from ipm.gui.pages import GUIPageCenterText, OvercookedPage, \
    DecisionTreeCreationPage, GUIPageWithTwoTreeChoices, GUIPageWithImage, \
    GUIPageWithTextAndURL, GUIPageWithSingleTree, EnvRewardModificationPage
from ipm.models.bc_agent import StayAgent
from ipm.models.decision_tree import sparse_ddt_to_decision_tree, DecisionTree
from ipm.overcooked.overcooked_envs import OvercookedPlayWithFixedPartner
from models.human_model import HumanModel
from models.robot_model import RobotModel
from models.fcp_model import FCPModel
from stable_baselines3 import PPO


class EnvWrapper:
    def __init__(self, layout, data_folder, hp_config, condition, domain_idx):
        # wrapping this up in a class so that we can easily change the reward function
        # this acts like a pointer
        self.multipliers = [1, 1, 1]
        self.ego_idx = 0
        self.alt_idx = 1
        self.layout = layout
        self.domain_idx = domain_idx
        self.data_folder = data_folder
        self.hp_config = hp_config
        self.rewards = []
        self.initial_reward = None
        self.modified_reward = None
        self.save_chosen_as_prior = False
        self.latest_save_file = None
        self.current_iteration = 0
        self.condition = condition

        dummy_env = OvercookedPlayWithFixedPartner(partner=StayAgent(), layout_name=layout,
                                                   reduced_state_space_ego=True, reduced_state_space_alt=True,
                                                   use_skills_ego=True, use_skills_alt=True)

        self.initial_warm_start_path = os.path.join('data', 'warm_start' + '.tar')
        self.initial_fcp_path = os.path.join('data', 'fcp' + '.tar')
        self.initial_random_policy_path = os.path.join('data', 'random' + '.tar')

        model = PPO("MlpPolicy", dummy_env)
        # weights = torch.load(self.initial_warm_start_path)
        # model.policy.load_state_dict(weights['ego_state_dict'])
        human_ppo_policy = model.policy
        self.human_policy = HumanModel(layout, human_ppo_policy)

        input_dim = dummy_env.observation_space.shape[0]
        output_dim = dummy_env.n_actions_alt

        if 'warm_start' in self.condition:
            robot_path = self.initial_warm_start_path
        else:
            robot_path = self.initial_random_policy_path

        if self.layout == 'tutorial':
            robot_path = '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/data/prior_tree_policies/tutorial.tar'
        self.robot_policy = RobotModel(layout=layout,
                                       idct_policy_filepath=robot_path,
                                       human_policy=self.human_policy,
                                       input_dim=input_dim,
                                       output_dim=output_dim,
                                       randomize_initial_idct=hp_config.rpo_random_initial_idct,
                                       only_optimize_leaves=hp_config.rpo_rl_only_optimize_leaves)

        self.current_policy, tree_info = sparse_ddt_to_decision_tree(self.robot_policy.robot_idct_policy,
                                                                     self.robot_policy.env)


        if self.layout == 'tutorial':
            self.fcp_policy = FCPModel(dummy_env=dummy_env, filepath='/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/data/fcp/tutorial.tar')
        else:
            self.fcp_policy = FCPModel(dummy_env=dummy_env, filepath=self.initial_fcp_path)


    def initialize_env(self):
        # we keep track of the reward function that may change
        self.robot_policy.env.set_env(placing_in_pot_multiplier=self.multipliers[0],
                                      dish_pickup_multiplier=self.multipliers[1],
                                      soup_pickup_multiplier=self.multipliers[2])


class FollowUpExperiment:
    def __init__(self, condition: str, conditions: list, disable_surveys: bool = False, hp_config=None):
        self.user_id = get_next_user_id()
        print('#############################################')
        print('User ID: ', self.user_id)
        print('#############################################')
        self.condition = condition
        self.condition_num = conditions.index(condition) + 1
        self.data_folder = os.path.join('data',
                                        'experiments',
                                        condition,
                                        'user_' + str(self.user_id))
        self.hp_config = hp_config
        self.domain_names = ['tutorial', 'two_rooms_narrow']
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
        main_page = GUIPageCenterText(self.screen,
                                      'Welcome to our experiment. Are you ready to proceed? (Press next when signed consent form)',
                                      24,
                                      bottom_left_button=False, bottom_right_button=True,
                                      bottom_right_fn=self.next_page)

        # proceed_page = GUIPageCenterText(self.screen, 'Are you ready to proceed? (Press next when signed consent form)',
        #                                  24,
        #                                  bottom_left_button=False, bottom_right_button=True,
        #                                  bottom_left_fn=None, bottom_right_fn=self.next_page)

        survey_urls = ['https://gatech.co1.qualtrics.com/jfe/form/SV_5gXCgThPlDGGlQG']

        presurveys_page = GUIPageWithTextAndURL(screen=self.screen,
                                                text='Please take these surveys so that we have more info about your background and personality.',
                                                urls=survey_urls,
                                                font_size=24,
                                                bottom_left_button=False, bottom_right_button=True,
                                                bottom_left_fn=False, bottom_right_fn=self.next_page)

        self.pages.append(main_page)
        if not self.disable_surveys:
            self.pages.append(presurveys_page)
        # self.pages.append(proceed_page)

        task_definition_page = GUIPageWithImage(self.screen, 'Experiment Overview', 'text/task_definition_alt.png',
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=False)

        self.pages.append(task_definition_page)

        oc_tutorial_page = GUIPageWithImage(self.screen, 'Overcooked Gameplay Overview', 'OvercookedTutorial.png',
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(oc_tutorial_page)
        control_tutorial_page = GUIPageWithImage(self.screen, 'Controls', 'text/controls.png',
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(control_tutorial_page)

        tutorial_bad_reminder = GUIPageCenterText(self.screen, 'You will now be playing a tutorial. Within this scenario, the AI will be unhelpful.'
                                                         , 36,
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(tutorial_bad_reminder)
        dt_tutorial_page = GUIPageWithImage(self.screen, ' ',
                                            'DTTutorial_updated_2.png',
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(dt_tutorial_page)

        proceed_dt_page = GUIPageCenterText(self.screen, 'You will now play a practice round with your teammate. '
                                                         'Afterwards, you may modify it as you wish.', 36,
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(proceed_dt_page)

        proceed_text = "Thank you for playing with our tutorial agent. I hope you have become familiar with the mechanics of Overcooked. Now the main experiment will begin. You will be teaming with a new AI teammate in a new domain. Similar to before, you will have a chance to perform a policy intervention by directly modifying your AI Teammate's behavorial policy. After each gameplay, you will have a chance to decide which policy to team with."
        proceed_image = 'text/transition_tutorial_tree.png'
        proceed_image_alt = 'text/transition_tutorial_tree_alt.png'
        transition_1_2_image = 'text/transition_tree_1_2.png'
        transition_2_3_image = 'text/transition_tree_2_3.png'

        # proceed_page = GUIPageCenterText(self.screen, proceed_text,
        #                                  24,
        #                                  bottom_left_button=False, bottom_right_button=True,
        #                                  bottom_left_fn=None, bottom_right_fn=self.next_page, alt_display=True)
        self.tutorial_transition = GUIPageWithImage(self.screen, ' ', proceed_image,
                                                    bottom_left_button=False, bottom_right_button=True,
                                                    bottom_left_fn=None, bottom_right_fn=self.next_page,
                                                    wide_image=True)
        self.tutorial_transition_alt = GUIPageWithImage(self.screen, ' ', proceed_image_alt,
                                                    bottom_left_button=False, bottom_right_button=True,
                                                    bottom_left_fn=None, bottom_right_fn=self.next_page,
                                                    wide_image=True)
        self.reward_explanation = GUIPageWithImage(self.screen,
                                                   'Before we begin, here is a description of how to increase your score.',
                                                   'text/reward_explanation.png',
                                                   bottom_left_button=False, bottom_right_button=True,
                                                   bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=True)
        self.domain_explanation = GUIPageWithImage(self.screen,
                                                   ' ',
                                                   'forced_coordination_tutorial.jpg',
                                                   bottom_left_button=False, bottom_right_button=True,
                                                   bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=False)

        reward_img = 'text/reward_two_rooms_narrow_alt.png'
        self.reward_explanation_2 = GUIPageWithImage(self.screen, 'Domain 2 Scoring Scheme', reward_img,
                                      bottom_left_button=False, bottom_right_button=True,
                                      bottom_left_fn=None, bottom_right_fn=self.next_page,
                                      wide_image=True)
        self.domain_explanation_2 = GUIPageWithImage(self.screen,
                                                   'Domain 2 Description',
                                                   'two_rooms_narrow_tutorial.jpg',
                                                   bottom_left_button=False, bottom_right_button=True,
                                                   bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=False)

        self.tree_mod_intro = GUIPageWithImage(self.screen,
                                               "Here is a description of controls to modify the AI's behavior.",
                                               'text/tree_tutorial_text.png',
                                               bottom_left_button=False, bottom_right_button=True,
                                               bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=False)
        self.transition_1_2 = GUIPageWithImage(self.screen, ' ', transition_1_2_image,
                                               bottom_left_button=False, bottom_right_button=True,
                                               bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=True)
        # self.transition_2_3 = GUIPageWithImage(self.screen, ' ', transition_2_3_image,
        #                                        bottom_left_button=False, bottom_right_button=True,
        #                                        bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=True)

        # self.pages.append(proceed_page)

    def setup_survey_misc_pages(self):
        text = 'Please take the survey. Press next when finished to begin the next interaction.'
        self.survey_page = GUIPageCenterText(self.screen, text, 24,
                                             bottom_left_button=False, bottom_right_button=True,
                                             bottom_left_fn=False, bottom_right_fn=self.next_page,
                                             nasa_tlx=True, is_survey_page=True)

        survey_urls = ['https://gatech.co1.qualtrics.com/jfe/form/SV_3Dku0KGtuDQbGiW']
        survey_urls_alt = ['https://gatech.co1.qualtrics.com/jfe/form/SV_40XOPAB4R0LQrX0',
                           'https://gatech.co1.qualtrics.com/jfe/form/SV_3Dku0KGtuDQbGiW']

        self.survey_qual = GUIPageWithTextAndURL(screen=self.screen,
                                                 text='Please take the qualtrics survey provided by the researcher.',
                                                 urls=survey_urls,
                                                 font_size=24,
                                                 bottom_left_button=False, bottom_right_button=True,
                                                 bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.survey_qual_alt = GUIPageWithTextAndURL(screen=self.screen,
                                                 text='Please take the qualtrics survey provided by the researcher.',
                                                 urls=survey_urls_alt,
                                                 font_size=24,
                                                 bottom_left_button=False, bottom_right_button=True,
                                                 bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.thank_you_page = GUIPageCenterText(self.screen, 'Thank you for participating in our study', 24,
                                                bottom_left_button=False, bottom_right_button=False,
                                                bottom_left_fn=None, bottom_right_fn=None,
                                                nasa_tlx=False)

    def setup_main_pages(self):
        self.modify_tree_pages = []
        self.env_pages = []
        self.fcp_env_pages = []
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

            env_page_fcp = OvercookedPage(self.screen, env_wrapper, tree_page,
                                      layout=env_wrapper.layout, text=' ',
                                      font_size=24,
                                      bottom_left_button=False, bottom_right_button=True,
                                      bottom_left_fn=None, bottom_right_fn=self.next_page,
                                      play_fcp=True)
            self.fcp_env_pages.append(env_page_fcp)

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
                                                           bottom_left_button=False,
                                                           bottom_right_button=True,
                                                           bottom_left_fn=None,
                                                           bottom_right_fn=self.next_page)
            self.initial_tree_show_pages.append(initial_tree_show_page)

            next_tree_show_page = GUIPageWithSingleTree(False, self.screen, tree_page=tree_page,
                                                        env_wrapper=env_wrapper,
                                                        font_size=24,
                                                        bottom_left_button=False,
                                                        bottom_right_button=True,
                                                        bottom_left_fn=None,
                                                        bottom_right_fn=self.next_page)
            self.next_tree_show_pages.append(next_tree_show_page)

            env_reward_modification_page = EnvRewardModificationPage(env_wrapper, screen=self.screen,
                                                                     settings=self.settings,
                                                                     X=self.settings.width, Y=self.settings.height,
                                                                     font_size=24,
                                                                     bottom_left_button=False,
                                                                     bottom_right_button=True,
                                                                     bottom_left_fn=None,
                                                                     bottom_right_fn=self.next_page)
            self.reward_modify_pages.append(env_reward_modification_page)

    def setup_pages(self):

        self.pages = []
        self.current_page = 0

        self.add_preliminary_pages()
        self.setup_survey_misc_pages()

        self.env_wrappers = [
            EnvWrapper(layout=layout, data_folder=self.data_folder, hp_config=self.hp_config, domain_idx=i,
                       condition=self.condition) for i, layout in enumerate(self.domain_names)]
        self.setup_main_pages()

        n_iterations = 3
        for layout_idx in range(len(self.env_wrappers)):
            is_tutorial = layout_idx == 0
            current_n_iterations = n_iterations if not is_tutorial else 1
            # TODO: can we do this better?
            if self.domain_names[layout_idx] == 'two_rooms_narrow':
                self.pages.append(self.domain_explanation_2)
                self.pages.append(self.reward_explanation_2)

            if self.condition == 'fcp_then_warm_start' or self.condition == 'fcp_then_scratch':
                self.pages.append(self.fcp_env_pages[layout_idx])
            else:
                self.pages.append(self.env_pages[layout_idx])

            for i in range(current_n_iterations):
                if (self.condition == 'fcp_then_warm_start' or self.condition == 'fcp_then_scratch') and not is_tutorial:  # modify tree
                    transition_img = 'text/transition_fcp_game.png'
                    transition = GUIPageWithImage(self.screen, ' ', transition_img,
                                                  bottom_left_button=False, bottom_right_button=True,
                                                  bottom_left_fn=None, bottom_right_fn=self.next_page,
                                                  wide_image=True)
                    self.pages.append(transition)
                    self.pages.append(self.fcp_env_pages[layout_idx])
                else:
                    self.pages.append(self.tree_mod_intro)
                    self.pages.append(self.modify_tree_pages[layout_idx])
                    self.pages.append(self.env_pages[layout_idx])
                    self.pages.append(self.initial_tree_show_pages[layout_idx])
                    self.pages.append(self.next_tree_show_pages[layout_idx])
                #
                # if self.condition_num == 1:  # choose between two policies for first 3 conditions
                #     self.pages.append(self.initial_tree_show_pages[layout_idx])
                #     self.pages.append(self.next_tree_show_pages[layout_idx])
                    # self.pages.append(self.two_choices_pages[layout_idx])
            if not is_tutorial:
                transition_img = 'text/transition_to_fcp.png'
                transition = GUIPageWithImage(self.screen, ' ', transition_img,
                                              bottom_left_button=False, bottom_right_button=True,
                                              bottom_left_fn=None, bottom_right_fn=self.next_page,
                                              wide_image=True)
                self.pages.append(transition)

            if not is_tutorial and not self.disable_surveys:
                self.pages.append(self.survey_page)
                self.pages.append(self.survey_qual)
            if is_tutorial:
                if self.condition == 'fcp_then_warm_start' or self.condition == 'fcp_then_scratch':
                    self.pages.append(self.tutorial_transition_alt)
                else:
                    self.pages.append(self.tutorial_transition)
                # self.pages.append(self.reward_explanation)
            else:
                if self.condition == 'fcp_then_warm_start' or self.condition == 'fcp_then_scratch':
                    self.pages.append(self.env_pages[layout_idx])
                else:
                    self.pages.append(self.fcp_env_pages[layout_idx])

                for i in range(current_n_iterations):
                    if self.condition == 'fcp_then_warm_start' or self.condition == 'fcp_then_scratch':  # modify tree
                        self.pages.append(self.tree_mod_intro)
                        self.pages.append(self.modify_tree_pages[layout_idx])
                        self.pages.append(self.env_pages[layout_idx])
                    else:
                        transition_img = 'text/transition_fcp_game.png'
                        transition = GUIPageWithImage(self.screen, ' ', transition_img,
                                                      bottom_left_button=False, bottom_right_button=True,
                                                      bottom_left_fn=None, bottom_right_fn=self.next_page,
                                                      wide_image=True)
                        self.pages.append(transition)
                        self.pages.append(self.fcp_env_pages[layout_idx])

                if not is_tutorial and not self.disable_surveys:
                    self.pages.append(self.survey_page)
                    self.pages.append(self.survey_qual_alt)
                # if layout_idx == 1:
                #     self.pages.append(self.transition_1_2)
                # elif layout_idx == 2:
                #     pass
                    # self.pages.append(self.transition_2_3)
                # else:
                #     pass
        self.pages.append(self.thank_you_page)

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

                # UNCOMMENT IF WE WANT ZOOM IN FOR TWO TREE CHOICES PAGE!
                # if self.pages[self.current_page].__class__.__name__ == 'GUIPageWithTwoTreeChoices':
                #     if event.type == pygame.MOUSEBUTTONDOWN:
                #         if event.button == 4:
                #             self.settings.zoom_in()
                #         elif event.button == 5:
                #             self.settings.zoom_out()
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
                self.save_initial_tree_img()

            if not self.showed_nasa_tlx and self.pages[self.current_page].__class__.__name__ == 'GUIPageCenterText' \
                    and self.pages[self.current_page].nasa_tlx:
                self.showed_nasa_tlx = True
                self.run_optimization()  # we somehow need to allow the users to be able to do the surveys while we optimize..
                run_gui(self.user_id, self.condition_num, self.current_domain)

    def run_optimization(self):
        data_file = self.env_wrappers[self.current_domain].latest_save_file
        is_optimization_condition = False
        #
        # if self.condition_num != 6:
        #     self.env_wrappers[self.current_domain].robot_policy.translate_recent_data_to_labels(
        #         recent_data_loc=data_file)
        #     self.env_wrappers[self.current_domain].robot_policy.finetune_intent_model(
        #         learning_rate=self.hp_config.ipo_lr,
        #         n_epochs=self.hp_config.ipo_n_epochs)

        if is_optimization_condition:
            user_interacted = self.env_wrappers[self.current_domain].human_policy.translate_recent_data_to_labels(
                recent_data_loc=data_file)
            if user_interacted:
                self.env_wrappers[self.current_domain].human_policy.finetune_human_ppo_policy(
                    learning_rate=self.hp_config.hpo_lr,
                    n_epochs=self.hp_config.hpo_n_epochs)
            else:
                print('no actions in dataset so no optimization :(')

            if self.hp_config.rpo_ga and self.hp_config.rpo_rl:
                algorithm_choice = 'ga+rl'
            elif self.hp_config.rpo_ga:
                algorithm_choice = 'ga'
            elif self.hp_config.rpo_rl:
                algorithm_choice = 'rl'
            else:
                raise ValueError('Invalid rpo algorithm choice')

            # TODO: add saving of robot policy, remove hardcode of path
            torch.save(
                {'alt_state_dict': self.env_wrappers[self.current_domain].robot_policy.robot_idct_policy.state_dict(),
                 'human_ppo_policy': self.env_wrappers[self.current_domain].human_policy.human_ppo_policy.state_dict()},
                'pre_robot_update.tar')

            # self.env_wrappers[self.current_domain].robot_policy.finetune_robot_idct_policy(
            #     recent_data_file=data_file,
            #     rl_n_steps=self.hp_config.rpo_rl_n_steps,
            #     rl_learning_rate=self.hp_config.rpo_rl_lr,
            #     algorithm_choice=algorithm_choice,
            #     ga_depth=self.hp_config.rpo_ga_depth,
            #     ga_n_gens=self.hp_config.rpo_ga_n_gens,
            #     ga_n_pop=self.hp_config.rpo_ga_n_pop,
            #     ga_n_parents_mating=self.hp_config.rpo_ga_n_parents_mating,
            #     ga_crossover_prob=self.hp_config.rpo_ga_crossover_prob,
            #     ga_crossover_type=self.hp_config.rpo_ga_crossover_type,
            #     ga_mutation_prob=self.hp_config.rpo_ga_mutation_prob,
            #     ga_mutation_type=self.hp_config.rpo_ga_mutation_type)

            self.env_wrappers[self.current_domain].robot_policy.finetune_robot_idct_policy_parallel()

            self.env_wrappers[self.current_domain].current_policy, tree_info = sparse_ddt_to_decision_tree(
                self.env_wrappers[self.current_domain].robot_policy.robot_idct_policy,
                self.env_wrappers[self.current_domain].robot_policy.env)

            self.current_tree_copy = copy.deepcopy(self.env_wrappers[self.current_domain].current_policy)
            self.frozen_pages[self.current_domain].decision_tree_history += [self.current_tree_copy]
            self.frozen_pages[self.current_domain].current_policy = self.env_wrappers[
                self.current_domain].current_policy



    def save_rewards_for_domain(self, domain_idx):
        folder = os.path.join(self.data_folder, self.domain_names[domain_idx])
        filepath = os.path.join(folder, 'rewards.txt')
        with open(filepath, 'w') as f:
            try:
                tree_page = self.modify_tree_pages[domain_idx]
                f.write(str(tree_page.env_wrapper.rewards))
            except IndexError:
                print('something wrong with condition', self.condition_num, 'domain', domain_idx)

    def next_domain(self):
        # save rewards to file
        self.save_rewards_for_domain(domain_idx=self.current_domain)
        self.current_domain += 1
        new_folder = os.path.join(self.data_folder, self.domain_names[self.current_domain])
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        self.current_iteration = 0

    def save_tree_img(self, initial=True):
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

    def save_initial_tree_img(self):
        self.save_tree_img(initial=True)
        self.saved_first_tree = True

    def save_final_tree_img(self):
        self.save_tree_img(initial=False)

    def new_tree_page(self, domain_idx):
        if domain_idx != self.current_domain:
            self.next_domain()
        else:
            self.current_iteration += 1
            self.env_wrappers[self.current_domain].current_iteration += 1
        self.two_choices_pages[domain_idx].loaded_images = False
        self.initial_tree_show_pages[domain_idx].loaded_images = False
        self.next_tree_show_pages[domain_idx].loaded_images = False
        self.saved_first_tree = False

    def save_times(self):
        output_file = os.path.join(self.data_folder, 'times.csv')
        # save times and pages names to csv
        with open(output_file, 'w') as outp:
            outp.write('page,time\n')
            for i in range(len(self.pages_names)):
                outp.write(f'{self.pages_names[i]}, {self.times[i]}\n')


    def simple_function_to_check_tree_leaves(self, tree):
        """
        Start at root and move down until you hit each leaf. If any leaf not summing to 1, print it out.
        Args:
            tree:

        Returns:

        """

        for i in range(100):
            found_leaf = False
            current_node = tree.root
            while not found_leaf:
                # randomly sample left or right
                print('----------------------------')
                import numpy as np
                if np.random.rand() < 0.5:
                    current_node = current_node.left
                    print('left')
                else:
                    current_node = current_node.right
                    print('right')
                if current_node.__class__.__name__ == 'LeafNode':
                    print('found leaf node')
                    found_leaf = True
                    total_sum = sum(current_node.action.values)
                    print('leaf summing to 1')
                    action_names = ['Wait',
                                    'Get Onion from D', 'Get Onion from C',
                                    'Get Dish from D', 'Get Dish from C',
                                    'Get Soup from Pot', 'Get Soup from C',
                                    'Serve Soup', 'Bring To Pot', 'Place on Counter']
                    action_names += ['Get Tomato from D', 'Get Tomato from C']
                    print(current_node.action.values)
                    b = current_node.action.indices
                    print(action_names[b[0]], action_names[b[1]], action_names[b[2]])
                    if total_sum != 1:
                        print('leaf not summing to 1')
                        return False
        return True



    def next_page(self):
        # first check if it is a decision tree creation page
        # if so, then make sure the probabilities all sum up to 1 for each leaf
        if self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage':
            # for action_item in self.pages[self.current_page].gui_action_items:
            #     total_sum = float(action_item.value) + float(action_item.value1) + float(action_item.value2)
            #     if total_sum != 1:
            #         print(action_item.position)
            #         return
            leaves_good = self.simple_function_to_check_tree_leaves(self.pages[self.current_page].current_policy)
            if not leaves_good:
                return
        # record time spent in prior page
        self.times.append(time.time() - self.page_start_time)
        self.page_start_time = time.time()

        # save final tree if the prior page is of type DecisionTreeCreationPage
        if self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage':
            self.save_final_tree_img()

        self.pages[self.current_page].hide()
        self.current_page += 1

        if type(self.pages[self.current_page]).__name__ == 'DecisionTreeCreationPage':
            idx = self.pages[self.current_page].env_wrapper.domain_idx
            self.new_tree_page(domain_idx=idx)

        if self.current_page == len(self.pages) - 1:
            # this means you are on final page
            self.save_times()
            self.save_rewards_rohan_way()
            self.save_rewards_for_domain(domain_idx=self.current_domain)
            self.pages[self.current_page].show()
        else:
            self.pages_names.append(self.pages[self.current_page].__class__.__name__)
            self.showed_nasa_tlx = False
            self.saved_first_tree = False
            self.pages[self.current_page].show()

    def save_rewards_rohan_way(self):
        """
        Saving rewards as torch file
        Returns:

        """
        rewards_dict = {}
        for i in range(len(self.env_wrappers)):
            env_wrapper = self.env_wrappers[i]
            rewards_dict[self.domain_names[i]] = env_wrapper.rewards
            print(self.domain_names[i])
        torch.save(rewards_dict, os.path.join(self.data_folder, 'rewards.pt'))

    def previous_page(self):
        self.pages[self.current_page].hide()
        self.current_page -= 1
        self.pages[self.current_page].show()

    def pick_initial_policy(self):
        modify_tree_page = self.modify_tree_pages[self.current_domain]
        initial_policy = modify_tree_page.decision_tree_history[0]
        initial_reward = modify_tree_page.env_wrapper.rewards[0]

        modify_tree_page.env_wrapper.initial_reward = initial_reward
        modify_tree_page.env_wrapper.modified_reward = None

        modify_tree_page.env_wrapper.rewards.append(initial_reward)
        # remove first reward
        modify_tree_page.env_wrapper.rewards = modify_tree_page.env_wrapper.rewards[1:]
        modify_tree_page.reset_initial_policy(initial_policy)
        self.next_page()

    def update_prior_policy(self, tree_page):
        final_policy = tree_page.decision_tree_history[-1]
        path = tree_page.env_wrapper.initial_warm_start_path
        if type(final_policy).__name__ == 'DecisionTree':
            with open(path, 'wb') as outp:
                pickle.dump(final_policy, outp, pickle.HIGHEST_PROTOCOL)
        else:
            # use pytorch to write it out
            torch.save(final_policy, path)

    def pick_final_policy(self):
        tree_page = self.modify_tree_pages[self.current_domain]

        tree_page.env_wrapper.initial_reward = tree_page.env_wrapper.modified_reward
        tree_page.env_wrapper.modified_reward = None

        final_policy = tree_page.decision_tree_history[-1]
        tree_page.reset_initial_policy(final_policy)

        if tree_page.env_wrapper.save_chosen_as_prior:
            self.update_prior_policy(tree_page)

        old_filename = 'final_tree.png'
        new_filename = 'initial_tree.png'
        folder = os.path.join(self.data_folder, self.domain_names[self.current_domain],
                              'iteration_' + str(self.current_iteration))
        imagepath = os.path.join(folder, old_filename)
        new_imagepath = os.path.join(folder, new_filename)
        shutil.copyfile(imagepath, new_imagepath)

        self.next_page()