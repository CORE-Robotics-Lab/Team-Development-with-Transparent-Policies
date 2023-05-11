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
    def __init__(self, layout, data_folder, hp_config, condition_num, domain_idx):
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
        self.condition_num = condition_num

        dummy_env = OvercookedPlayWithFixedPartner(partner=StayAgent(), layout_name=layout,
                                                   behavioral_model='dummy',
                                                   reduced_state_space_ego=True, reduced_state_space_alt=True,
                                                   use_skills_ego=True, use_skills_alt=True)

        self.initial_policy_path = os.path.join('data', 'prior_tree_policies', layout + '.tar')
        self.initial_fcp_path = os.path.join('data', 'fcp', layout + '.tar')
        intent_model_path = os.path.join('data', 'intent_models', layout + '.pt')

        model = PPO("MlpPolicy", dummy_env)
        weights = torch.load(self.initial_policy_path)
        model.policy.load_state_dict(weights['ego_state_dict'])
        human_ppo_policy = model.policy
        self.human_policy = HumanModel(layout, human_ppo_policy)

        input_dim = dummy_env.observation_space.shape[0]
        output_dim = dummy_env.n_actions_alt

        if condition_num <= 5:
            self.robot_policy = RobotModel(layout=layout,
                                           idct_policy_filepath=self.initial_policy_path,
                                           human_policy=self.human_policy,
                                           intent_model_filepath=intent_model_path,
                                           input_dim=input_dim,
                                           output_dim=output_dim,
                                           randomize_initial_idct=hp_config.rpo_random_initial_idct,
                                           only_optimize_leaves=hp_config.rpo_rl_only_optimize_leaves)

            self.current_policy, tree_info = sparse_ddt_to_decision_tree(self.robot_policy.robot_idct_policy,
                                                                         self.robot_policy.env)
            self.intent_model = self.robot_policy.intent_model
        else:
            # the first stuff is just loaded in so we can borrow the intent model
            self.robot_policy = RobotModel(layout=layout,
                                           idct_policy_filepath=self.initial_policy_path,
                                           human_policy=self.human_policy,
                                           intent_model_filepath=intent_model_path,
                                           input_dim=input_dim,
                                           output_dim=output_dim,
                                           randomize_initial_idct=hp_config.rpo_random_initial_idct,
                                           only_optimize_leaves=hp_config.rpo_rl_only_optimize_leaves)

            self.current_policy, tree_info = sparse_ddt_to_decision_tree(self.robot_policy.robot_idct_policy,
                                                                         self.robot_policy.env)
            self.intent_model = self.robot_policy.intent_model

            self.robot_policy = FCPModel(dummy_env=dummy_env, filepath=self.initial_fcp_path)
            self.current_policy = self.robot_policy


    def initialize_env(self):
        # we keep track of the reward function that may change
        self.robot_policy.env.set_env(placing_in_pot_multiplier=self.multipliers[0],
                                      dish_pickup_multiplier=self.multipliers[1],
                                      soup_pickup_multiplier=self.multipliers[2])


class MainExperiment:
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
        main_page = GUIPageCenterText(self.screen,
                                      'Welcome to our experiment. Are you ready to proceed? (Press next when signed consent form)',
                                      24,
                                      bottom_left_button=False, bottom_right_button=True,
                                      bottom_right_fn=self.next_page)

        # proceed_page = GUIPageCenterText(self.screen, 'Are you ready to proceed? (Press next when signed consent form)',
        #                                  24,
        #                                  bottom_left_button=False, bottom_right_button=True,
        #                                  bottom_left_fn=None, bottom_right_fn=self.next_page)

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
        # self.pages.append(proceed_page)

        oc_tutorial_page = GUIPageWithImage(self.screen, 'Overcooked Gameplay Overview', 'OvercookedTutorial.png',
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=None, bottom_right_fn=self.next_page)

        self.pages.append(oc_tutorial_page)
        if self.condition_num == 1:
            dt_tutorial_page = GUIPageWithImage(self.screen, 'Decision Tree Modification Overview', 'DTTutorial_updated.png',
                                                bottom_left_button=False, bottom_right_button=True,
                                                bottom_left_fn=None, bottom_right_fn=self.next_page)

            self.pages.append(dt_tutorial_page)

            proceed_dt_page = GUIPageCenterText(self.screen, 'You will now play a practice round with your teammate. '
                                                             'Afterwards, you may modify it as you wish.', 36,
                                                bottom_left_button=False, bottom_right_button=True,
                                                bottom_left_fn=None, bottom_right_fn=self.next_page)

            self.pages.append(proceed_dt_page)
        elif self.condition_num == 2:
            dt_tutorial_page = GUIPageWithImage(self.screen, 'Decision Tree Modification Overview',
                                                'nonedit_DTTutorial.png',
                                                bottom_left_button=False, bottom_right_button=True,
                                                bottom_left_fn=None, bottom_right_fn=self.next_page)

            self.pages.append(dt_tutorial_page)

            proceed_dt_page = GUIPageCenterText(self.screen, 'You will now play a practice round with your teammate. '
                                                             'Afterwards, the agent will optimize itself to be a better teammate. As this is a tutorial, the agent will optimize itself for a very short period of time.', 24,
                                                bottom_left_button=False, bottom_right_button=True,
                                                bottom_left_fn=None, bottom_right_fn=self.next_page, alt_display=True)
            self.pages.append(proceed_dt_page)

        elif self.condition_num == 3:
            dt_tutorial_page = GUIPageWithImage(self.screen, 'Decision Tree Modification Overview',
                                                'nonedit_DTTutorial.png',
                                                bottom_left_button=False, bottom_right_button=True,
                                                bottom_left_fn=None, bottom_right_fn=self.next_page)

            self.pages.append(dt_tutorial_page)

            proceed_dt_page = GUIPageCenterText(self.screen, "You will now play a practice round with your teammate. "
                                                             "Afterwards, you may modify your teammate's objectives as you wish and the agent will optimize itself to be a better teammate. As this is a tutorial, the agent will optimize itself for a very short period of time.", 24,
                                                bottom_left_button=False, bottom_right_button=True,
                                                bottom_left_fn=None, bottom_right_fn=self.next_page, alt_display=True)
            self.pages.append(proceed_dt_page)
        elif self.condition_num == 5:
            dt_tutorial_page = GUIPageWithImage(self.screen, 'Decision Tree Overview',
                                                'nonedit_DTTutorial.png',
                                                bottom_left_button=False, bottom_right_button=True,
                                                bottom_left_fn=None, bottom_right_fn=self.next_page)

            self.pages.append(dt_tutorial_page)
            proceed_dt_page = GUIPageCenterText(self.screen, "You will now play a practice round with your teammate. "
                                                             "Afterwards, you can visualize your teammate's policy.", 34,
                                                bottom_left_button=False, bottom_right_button=True,
                                                bottom_left_fn=None, bottom_right_fn=self.next_page)

            self.pages.append(proceed_dt_page)
        elif self.condition_num == 6:

            proceed_dt_page = GUIPageCenterText(self.screen, "You will now play a practice round with your teammate. ",
                                                36,
                                                bottom_left_button=False, bottom_right_button=True,
                                                bottom_left_fn=None, bottom_right_fn=self.next_page)

            self.pages.append(proceed_dt_page)

        else:
            raise NotImplementedError

        if self.condition == 'human_modifies_tree':
            proceed_text = "Thank you for playing with our tutorial agent. I hope you have become familiar with the mechanics of Overcooked. Now the main experiment will begin. You will be teaming with a new AI teammate in a new domain. Similar to before, you will have a chance to perform a policy intervention by directly modifying your AI Teammate's behavorial policy. After each gameplay, you will have a chance to decide which policy to team with."
            proceed_image = 'text/transition_tutorial_tree.png'
            transition_1_2_image = 'text/transition_tree_1_2.png'
            transition_2_3_image = 'text/transition_tree_2_3.png'
        elif self.condition == 'optimization':
            proceed_text = "Thank you for playing with our tutorial agent. I hope you have become familiar with the mechanics of Overcooked. Now the main experiment will begin. You will be teaming with a new AI teammate in a new domain. Similar to before, the agent will optimize itself to better support you as a teammate. As before, you will also have a chance to decide which policy to team with."
            proceed_image = 'text/transition_optimization.png'
            transition_1_2_image = 'text/transition_optimization_1_2.png'
            transition_2_3_image = 'text/transition_optimization_2_3.png'
        elif self.condition == 'optimization_while_modifying_reward':
            proceed_text = "Thank you for playing with our tutorial agent. I hope you have become familiar with the mechanics of Overcooked. Now the main experiment will begin. You will be teaming with a new AI teammate in a new domain. Similar to before, the agent will optimize itself to better support you as a teammate and you can help it by specifying certain objectives you would like prioritized. As before, you will also have a chance to decide which policy to team with."
            proceed_image = 'text/transition_reward_mod.png'
            transition_1_2_image = 'text/transition_reward_mod_1_2.png'
            transition_2_3_image = 'text/transition_reward_mod_2_3.png'
        elif self.condition == 'no_modification_interpretable':
            proceed_text = "Thank you for playing with our tutorial agent. I hope you have become familiar with the mechanics of Overcooked. Now the main experiment will begin. You will be teaming with a new AI teammate in a new domain. Similar to before, you will be able to view your AI teammate's policy after interacting with it."
            proceed_image = 'text/transition_interpretable_nomod.png'
            transition_1_2_image = 'text/transition_interpretable_nomod_1_2.png'
            transition_2_3_image = 'text/transition_interpretable_nomod_2_3.png'
        else:
            proceed_text = "Thank you for playing with our tutorial agent. I hope you have become familiar with the mechanics of Overcooked. Now the main experiment will begin. You will be teaming with a new AI teammate in a new domain."
            proceed_image = 'text/transition_nomod.png'
            transition_1_2_image = 'text/transition_nomod_1_2.png'
            transition_2_3_image = 'text/transition_nomod_2_3.png'

        # proceed_page = GUIPageCenterText(self.screen, proceed_text,
        #                                  24,
        #                                  bottom_left_button=False, bottom_right_button=True,
        #                                  bottom_left_fn=None, bottom_right_fn=self.next_page, alt_display=True)
        self.tutorial_transition = GUIPageWithImage(self.screen, ' ', proceed_image,
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=True)
        self.reward_explanation = GUIPageWithImage(self.screen, 'Before we begin, here is a description of how to increase your score.', 'text/reward_explanation.png',
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=True)
        self.tree_mod_intro = GUIPageWithImage(self.screen,
                                                   "Here is a description of controls to modify the AI's behavior.",
                                                   'text/tree_tutorial_text.png',
                                                   bottom_left_button=False, bottom_right_button=True,
                                                   bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=True)
        self.transition_1_2 = GUIPageWithImage(self.screen, ' ', transition_1_2_image,
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=True)
        self.transition_2_3 = GUIPageWithImage(self.screen, ' ', transition_2_3_image,
                                            bottom_left_button=False, bottom_right_button=True,
                                            bottom_left_fn=None, bottom_right_fn=self.next_page, wide_image=True)
        if self.condition_num == 5:
            self.transition_2_3.scaling = .1
        # self.pages.append(proceed_page)

    def setup_survey_misc_pages(self):
        if self.condition_num == 2 or self.condition_num == 3:
            text = 'Please take the survey. While you take it, your teammate will take 5 minutes to optimize itself.'
        else:
            text = 'Please take the survey. Press next when finished.'
        self.survey_page = GUIPageCenterText(self.screen, text, 24,
                                             bottom_left_button=False, bottom_right_button=True,
                                             bottom_left_fn=False, bottom_right_fn=self.next_page,
                                             nasa_tlx=True, is_survey_page=True)

        survey_urls = ['https://gatech.co1.qualtrics.com/jfe/form/SV_bCIZ8mjqcOtKveS',
                       'https://gatech.co1.qualtrics.com/jfe/form/SV_ezZAMpSbcQ3Vx9s',
                       'https://gatech.co1.qualtrics.com/jfe/form/SV_3gCgLUCf2sRNafA']

        self.survey_qual = GUIPageWithTextAndURL(screen=self.screen,
                                                 text='Please take the qualtrics survey provided by the researcher.',
                                                 urls=survey_urls,
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
        self.two_choices_pages = []
        self.initial_tree_show_pages = []
        self.next_tree_show_pages = []
        self.reward_modify_pages = []
        for i, env_wrapper in enumerate(self.env_wrappers):
            if self.condition_num != 6:
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
                                                                         bottom_left_button=False, bottom_right_button=True,
                                                                         bottom_left_fn=None,
                                                                         bottom_right_fn=self.next_page)
                self.reward_modify_pages.append(env_reward_modification_page)
            else:
                env_page = OvercookedPage(self.screen, env_wrapper, tree_page=None,
                                          layout=env_wrapper.layout, text=' ',
                                          font_size=24,
                                          bottom_left_button=False, bottom_right_button=True,
                                          bottom_left_fn=None, bottom_right_fn=self.next_page)
                self.env_pages.append(env_page)

    def setup_pages(self):

        self.pages = []
        self.current_page = 0

        self.add_preliminary_pages()
        self.setup_survey_misc_pages()

        self.env_wrappers = [EnvWrapper(layout=layout, data_folder=self.data_folder, hp_config=self.hp_config, domain_idx=i,
                                        condition_num=self.condition_num) for i, layout in enumerate(self.domain_names)]
        self.setup_main_pages()

        only_show_tree_no_modify = self.condition_num == 2 or self.condition_num == 3 or self.condition_num == 5
        if only_show_tree_no_modify:
            for tree_page in self.modify_tree_pages:
                tree_page.frozen = True
            self.frozen_pages = self.modify_tree_pages

        n_iterations = 3
        for layout_idx in range(len(self.env_wrappers)):
            is_tutorial = layout_idx == 0
            current_n_iterations = n_iterations if not is_tutorial else 1
            self.pages.append(self.env_pages[layout_idx])
            for i in range(current_n_iterations):
                if self.condition_num == 1:  # modify tree
                    self.pages.append(self.tree_mod_intro)
                    self.pages.append(self.modify_tree_pages[layout_idx])
                elif self.condition_num == 2:  # optimization, show policy u played with
                    self.pages.append(self.frozen_pages[layout_idx])
                elif self.condition_num == 3:  # reward modification
                    self.pages.append(self.frozen_pages[layout_idx])  # optimization, show policy u played with
                    self.pages.append(self.reward_modify_pages[layout_idx])  # show reward modification page
                elif self.condition_num == 4:  # not intepretable black-box
                    pass
                elif self.condition_num == 5:  # intepretable black-box
                    self.pages.append(self.frozen_pages[layout_idx])
                elif self.condition_num == 6:  # do nothing for fcp (black-box)
                    pass
                else:
                    raise NotImplementedError("Condition number {} not implemented".format(self.condition_num))

                if not is_tutorial and not self.disable_surveys:
                    self.pages.append(self.survey_page)

                self.pages.append(self.env_pages[layout_idx])

                # for optimization conditions, show policy again which may be optimized
                if self.condition_num == 2:  # optimization, show policy u played with
                    self.pages.append(self.frozen_pages[layout_idx])
                elif self.condition_num == 3:  # reward modification
                    self.pages.append(self.frozen_pages[layout_idx])  # optimization, show policy u played with
                    # self.pages.append(self.reward_modify_pages[layout_idx])  # show reward modification page

                if self.condition_num < 4:  # choose between two policies for first 3 conditions
                    self.pages.append(self.initial_tree_show_pages[layout_idx])
                    self.pages.append(self.next_tree_show_pages[layout_idx])
                    self.pages.append(self.two_choices_pages[layout_idx])
            if not is_tutorial and not self.disable_surveys:
                self.pages.append(self.survey_qual)
            if is_tutorial:
                self.pages.append(self.tutorial_transition)
                self.pages.append(self.reward_explanation)
            else:
                if layout_idx == 1:
                    self.pages.append(self.transition_1_2)
                elif layout_idx == 2:
                    self.pages.append(self.transition_2_3)
                else:
                    pass
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
            elif self.saved_first_tree and \
                    self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage' and \
                    (self.condition_num == 2 or self.condition_num == 3):
                self.save_final_tree_img()

            if not self.showed_nasa_tlx and self.pages[self.current_page].__class__.__name__ == 'GUIPageCenterText' \
                    and self.pages[self.current_page].nasa_tlx:
                self.showed_nasa_tlx = True
                self.run_optimization()  # we somehow need to allow the users to be able to do the surveys while we optimize..
                run_gui(self.user_id, self.condition_num, self.current_domain)

    def run_optimization(self):
        data_file = self.env_wrappers[self.current_domain].latest_save_file
        is_optimization_condition = self.condition_num == 2 or self.condition_num == 3
        #
        # if self.condition_num != 6:
        #     self.env_wrappers[self.current_domain].robot_policy.translate_recent_data_to_labels(
        #         recent_data_loc=data_file)
        #     self.env_wrappers[self.current_domain].robot_policy.finetune_intent_model(
        #         learning_rate=self.hp_config.ipo_lr,
        #         n_epochs=self.hp_config.ipo_n_epochs)

        if is_optimization_condition:
            self.env_wrappers[self.current_domain].human_policy.translate_recent_data_to_labels(
                recent_data_loc=data_file)
            self.env_wrappers[self.current_domain].human_policy.finetune_human_ppo_policy(
                learning_rate=self.hp_config.hpo_lr,
                n_epochs=self.hp_config.hpo_n_epochs)


            if self.hp_config.rpo_ga and self.hp_config.rpo_rl:
                algorithm_choice = 'ga+rl'
            elif self.hp_config.rpo_ga:
                algorithm_choice = 'ga'
            elif self.hp_config.rpo_rl:
                algorithm_choice = 'rl'
            else:
                raise ValueError('Invalid rpo algorithm choice')

            # TODO: add saving of robot policy, remove hardcode of path
            torch.save({'alt_state_dict': self.env_wrappers[self.current_domain].robot_policy.robot_idct_policy.state_dict(),
                        'robot_intent_model': self.env_wrappers[self.current_domain].robot_policy.intent_model.state_dict(),
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

    def save_rewards_for_domain(self, domain_idx):
        folder = os.path.join(self.data_folder, self.domain_names[domain_idx])
        filepath = os.path.join(folder, 'rewards.txt')
        with open(filepath, 'w') as f:
            tree_page = self.modify_tree_pages[domain_idx]
            f.write(str(tree_page.env_wrapper.rewards))

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

    def next_page(self):
        # first check if it is a decision tree creation page
        # if so, then make sure the probabilities all sum up to 1 for each leaf
        if self.pages[self.current_page].__class__.__name__ == 'DecisionTreeCreationPage':
            for action_item in self.pages[self.current_page].gui_action_items:
                total_sum = float(action_item.value) + float(action_item.value1) + float(action_item.value2)
                if total_sum != 1:
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
        path = tree_page.env_wrapper.initial_policy_path
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
