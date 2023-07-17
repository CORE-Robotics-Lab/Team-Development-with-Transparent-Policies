import pickle
import os
import csv
import numpy as np
import re
import matplotlib.pyplot as plt


# TODOS
# TODO: add not included stuff



def find_male_female_split(participant_data):
    male = 0
    female = 0
    other = 0
    for i in participant_data:
        if participant_data[i]['gender'] == 'Male':
            male += 1
        elif participant_data[i]['gender'] == 'Female':
            female += 1
        else:
            print(participant_data[i]['gender'])
            other += 1
    return male, female, other


def find_min_max_age(participant_data):
    min_age = 1000
    max_age = 0
    ages = []
    for i in participant_data:
        if int(participant_data[i]['age']) <= min_age:
            min_age = int(participant_data[i]['age'])
        if int(participant_data[i]['age']) >= max_age:
            max_age = int(participant_data[i]['age'])
        ages.append(int(participant_data[i]['age']))
    return min_age, max_age, np.mean(ages), np.std(ages)


def participant_data_checker(participant_data):
    pass


def count_each_condition(participant_data):
    conditions = {'Human Modifies Tree': 0,
                  'Optimization': 0,
                  'No modification (Black-Box)': 0,
                  'No modification (Interpretable)': 0,
                  'FCP': 0}
    condition_mapper = {'Human Modifies Tree': [],
                        'Optimization': [],
                        'No modification (Black-Box)': [],
                        'No modification (Interpretable)': [],
                        'FCP': []}


    for i in participant_data:
        if 'Human Modifies Tree' in participant_data[i].keys():
            conditions['Human Modifies Tree'] += 1
            condition_mapper['Human Modifies Tree'].append(i)
        elif 'Optimization' in participant_data[i].keys():
            conditions['Optimization'] += 1
            condition_mapper['Optimization'].append(i)
        elif 'No modification (Black-Box)' in participant_data[i].keys():
            conditions['No modification (Black-Box)'] += 1
            condition_mapper['No modification (Black-Box)'].append(i)
        elif 'No modification (Interpretable)' in participant_data[i].keys():
            conditions['No modification (Interpretable)'] += 1
            condition_mapper['No modification (Interpretable)'].append(i)
        elif 'FCP' in participant_data[i].keys():
            conditions['FCP'] += 1
            condition_mapper['FCP'].append(i)
        else:
            print(i, 'error')

    return conditions, condition_mapper


def get_participant_condition(participant_data, participant_id):
    conditions = {'Human Modifies Tree': 0,
                  'Optimization': 0,
                  'No modification (Black-Box)': 0,
                  'No modification (Interpretable)': 0,
                  'FCP': 0}

    for i in participant_data:
        if i == participant_id:
            if 'Human Modifies Tree' in participant_data[i].keys():
                return 'Human Modifies Tree'
            elif 'Optimization' in participant_data[i].keys():
                return 'Optimization'
            elif 'No modification (Black-Box)' in participant_data[i].keys():
                return 'No modification (Black-Box)'
            elif 'No modification (Interpretable)' in participant_data[i].keys():
                return 'No modification (Interpretable)'
            elif 'FCP' in participant_data[i].keys():
                return 'FCP'
            else:
                print(i, 'error in getting participant condtion')
                return None
        else:
            continue


# participants that must be ignored
not_included = [0]
errored_out = ['12', '28', '4']  # maybe add 7-9 and 11
intuitive_mapping = {}
intuitive_mapping_real = {}
# pre-experiment survey
with open(
        '/home/rohanpaleja/Downloads/combined_pre_data/combined_pre_data.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    participant_data = {}
    line_count = 0
    for e, i in enumerate(csv_reader):
        print(i)
        if e <= 1:
            if e == 0:
                # create some intutitive mapping for all the Q's
                for j, k in enumerate(i.keys()):
                    if k[0] == 'Q':
                        intuitive_mapping[k] = i[k]
                continue
            else:
                continue

        if i['DistributionChannel'] == 'preview':
            continue
        if int(i['Q1']) in not_included or i['Q1'] in errored_out:
            continue
        else:
            participant_data[i['Q1']] = {}
            participant_data[i['Q1']]['gender'] = i['Q2']
            participant_data[i['Q1']]['age'] = i['Q3']
            participant_data[i['Q1']]['major'] = i['Q9']
            participant_data[i['Q1']]['gaming_familiarity'] = i['Q4_1']
            participant_data[i['Q1']]['dt_familiarity'] = i['Q4_2']
            participant_data[i['Q1']]['weekly_hours_videogames'] = i['Q5']

            # personality information
            mapping = {'Very Strongly disagree': 1, 'Very Strongly Disagree': 1, 'Strongly Disagree': 2, 'Disagree': 3,
                       'Neither Agree nor Disagree': 4,
                       'Agree': 5, 'Strongly Agree': 6, 'Very Strongly Agree': 7}
            try:
                participant_data[i['Q1']]['E'] = 20 + mapping[i['Q12_1']] - mapping[i['Q12_6']] + mapping[
                    i['Q12_11']] - mapping[
                                                     i['Q12_16']] + \
                                                 mapping[i['Q12_21']] - mapping[i['Q12_26']] + mapping[
                                                     i['Q12_31']] - mapping[
                                                     i['Q12_36']] + \
                                                 mapping[i['Q12_41']] - mapping[i['Q12_46']]

                participant_data[i['Q1']]['A'] = 14 - mapping[i['Q12_2']] + mapping[i['Q12_7']] - mapping[
                    i['Q12_12']] + mapping[
                                                     i['Q12_17']] - \
                                                 mapping[i['Q12_22']] + mapping[i['Q12_27']] - mapping[
                                                     i['Q12_32']] + mapping[
                                                     i['Q12_37']] + \
                                                 mapping[i['Q12_42']] + mapping[i['Q12_47']]

                participant_data[i['Q1']]['C'] = 14 + mapping[i['Q12_3']] - mapping[i['Q12_8']] + mapping[
                    i['Q12_13']] - mapping[
                                                     i['Q12_18']] + \
                                                 mapping[i['Q12_23']] - mapping[i['Q12_28']] + mapping[
                                                     i['Q12_33']] - mapping[
                                                     i['Q12_38']] + \
                                                 mapping[i['Q12_43']] + mapping[i['Q12_48']]

                participant_data[i['Q1']]['N'] = 38 - mapping[i['Q12_4']] + mapping[i['Q12_9']] - mapping[
                    i['Q12_14']] + mapping[
                                                     i['Q12_19']] - \
                                                 mapping[i['Q12_24']] - mapping[i['Q12_29']] - mapping[
                                                     i['Q12_34']] - mapping[
                                                     i['Q12_39']] - \
                                                 mapping[i['Q12_44']] - mapping[i['Q12_49']]

                participant_data[i['Q1']]['O'] = 8 + mapping[i['Q12_5']] - mapping[i['Q12_10']] + mapping[
                    i['Q12_15']] - mapping[
                                                     i['Q12_20']] + \
                                                 mapping[i['Q12_25']] - mapping[i['Q12_30']] + mapping[
                                                     i['Q12_35']] + mapping[
                                                     i['Q12_40']] + \
                                                 mapping[i['Q12_45']] + mapping[i['Q12_50']]
            except:
                print('Personality Error occured in i for ', i['Q1'])
                continue

total_participants = len(participant_data.keys())
male, female, other = find_male_female_split(participant_data)
min_age, max_age, age_mean, age_std = find_min_max_age(participant_data)
print('Male is ', male, '/', total_participants)
print('Female is ', female, '/', total_participants)
print('Other is ', other, '/', total_participants)
print('Min age is ', min_age, '. Max age is ', max_age, '. Mean age is ', age_mean, '. Std age is ', age_std)

# post-experiment subjective survey

# first check since some variables are double named
with open(
        '/home/rohanpaleja/Downloads/combined_post_data/combined_post_data.csv') as csv_file:
    for line in csv_file.readlines():
        all_vars = line.split(',')
        break
    import collections

    # if this prints anything, go in the csv and fix manually
    print([item for item, count in collections.Counter(all_vars).items() if count > 1])

with open(
        '/home/rohanpaleja/Downloads/combined_post_data/combined_post_data.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for e, i in enumerate(csv_reader):
        # print(i)
        if i['DistributionChannel'] == 'preview':
            continue
        if e <= 1:
            continue
        else:
            mapping = {'Strongly disagree': 1, 'Disagree': 2, 'Somewhat disagree': 3, 'Neither agree nor disagree': 4,
                       'Somewhat agree': 5, 'Agree': 6, 'Strongly agree': 7}
            participant_id = i['Q16']

            if int(participant_id) in not_included or participant_id in errored_out:
                continue
            # if participant_id == '18':
            #     print('hi')

            # condition

            condition_mapping = {'A': 'Human Modifies Tree',
                                 'B': 'Optimization',
                                 'C': 'No modification (Black-Box)',
                                 'D': 'No modification (Interpretable)',
                                 'E': 'FCP'}

            condition = condition_mapping[i['Q4']]
            print('Processing participant ', participant_id + ' in condition ', condition)
            try:
                len(participant_data[participant_id][condition])
            except KeyError:
                participant_data[participant_id][condition] = {}
                if i['Q5'] == '1':
                    participant_data[participant_id]['domain_ordering'] = 'normal'
                else:
                    participant_data[participant_id]['domain_ordering'] = 'reversed'

            # domain
            domain = 'domain_' + i['Q5']

            participant_data[i['Q16']][condition][domain] = {}

            # fluency
            participant_data[participant_id][condition][domain]['fluency'] = mapping[i['Q1_1']] + mapping[i['Q1_2']] + \
                                                                             mapping[i['Q1_3']]

            participant_data[participant_id][condition][domain]['robot_contribution'] = 14 + -mapping[i['Q1_4']] + \
                                                                                        mapping[
                                                                                            i['Q1_5']] - mapping[
                                                                                            i['Q1_6']] + mapping[
                                                                                            i['Q1_7']]

            participant_data[participant_id][condition][domain]['trust'] = mapping[i['Q1_8']] + mapping[i['Q1_9']]

            participant_data[participant_id][condition][domain]['positive_teammate_traits'] = mapping[i['Q1_10']] + \
                                                                                              mapping[
                                                                                                  i['Q1_9']] + mapping[
                                                                                                  i['Q1_11']]

            participant_data[participant_id][condition][domain]['improvement'] = mapping[i['Q1_12']] + mapping[
                i['Q1_2']]

            participant_data[participant_id][condition][domain]['working_alliance'] = -mapping[i['Q1_13']] + mapping[
                i['Q1_14']] + mapping[i['Q1_15']] + mapping[i['Q1_16']] + mapping[i['Q1_17']] + mapping[i['Q1_18']] + \
                                                                                      mapping[i['Q1_19']]

            participant_data[participant_id][condition][domain]['goal'] = mapping[i['Q1_20']] - mapping[i['Q1_21']] + \
                                                                          mapping[
                                                                              i['Q1_22']] + mapping[i['Q1_19']]

            # participant_data[i['Q3']][condition]['working_alliance2'] = participant_data[i['Q3']][condition]['goal'] + \
            #                                                                participant_data[i['Q3']][condition][
            #                                                                    'working_alliance']
            try:
                participant_data[participant_id][condition][domain]['tool_vs_teammate'] = int(i['Q5_11'])
            except ValueError:
                # no click means full machine
                participant_data[participant_id][condition][domain]['tool_vs_teammate'] = 0

            participant_data[participant_id][condition][domain]['likability'] = mapping[i['Q5_1']] + mapping[
                i['Q5_2']] + \
                                                                                mapping[i['Q5_3']] + \
                                                                                mapping[i['Q5_4']] + mapping[i['Q5_5']]

condition_counts, condition_mapper = count_each_condition(participant_data)
print(condition_counts)

# objective results

# go through each participant and add their rewards, times, and nasatlx
for participant in participant_data.keys():
    # later people had .pt data

    # locate some files
    import os

    # get participant condition
    condition = get_participant_condition(participant_data, participant)

    if participant == '9':
        print('hi')

    if condition is None:
        continue

    # relate condition to file names
    if condition == 'Human Modifies Tree':
        file_condition = 'human_modifies_tree'
    elif condition == 'Optimization':
        file_condition = 'optimization'
    elif condition == 'No modification (Black-Box)':
        file_condition = 'no_modification_bb'
    elif condition == 'No modification (Interpretable)':
        file_condition = 'no_modification_interpretable'
    elif condition == 'FCP':
        file_condition = 'fcp'

    rootdir = (
            '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/data/experiments/' + file_condition + '/user_' + participant + '/')
    if participant == '9':
        rootdir = (
                '/home/rohanpaleja/PycharmProjects/ipm/ipm/bin/data/experiments/' + file_condition + '/user_10' + '/')
    import os
    from glob import glob

    result = [y for x in os.walk(rootdir) for y in glob(os.path.join(x[0], '*.txt'))]
    # print('result is', result)
    # do fc first
    for j in result:
        # handle reward and workload separately
        if participant == '12':
            continue
        if participant == '17':
            import torch

            reward_data = torch.load(rootdir + 'user_18/rewards.pt')

            participant_data[participant][condition]['domain_1']['rewards'] = [391, 405, 465, 468]
            participant_data[participant][condition]['domain_2']['rewards'] = reward_data['two_rooms_narrow']

            import numpy as np

            participant_data[participant][condition]['domain_1']['workload'] = [np.sum([10, 5, 35, 35, 15, 10]),
                                                                                np.sum([25, 0, 25, 30, 30, 0]),
                                                                                np.sum([20, 0, 10, 20, 15, 0]),
                                                                                np.sum([20, 0, 15, 25, 15, 10])]
            participant_data[participant][condition]['domain_2']['workload'] = [np.sum([20, 0, 0, 40, 15, 40]),
                                                                                np.sum([35, 0, 0, 35, 0, 40]),
                                                                                np.sum([20, 0, 0, 40, 15, 60]),
                                                                                np.sum([25, 0, 0, 45, 10, 65])]
            continue

        if participant == '3':
            # manual entry
            participant_data[participant][condition]['domain_1']['rewards'] = [397, 471, 471, 474]
            participant_data[participant][condition]['domain_2']['rewards'] = [285, 300, 303, 303]

        if participant == '1':
            participant_data[participant][condition]['domain_1']['rewards'] = [299, 305, 305, 302]
            participant_data[participant][condition]['domain_2']['rewards'] = [111, 153, 153, 153]

        if int(participant) < 5 or int(participant) == 16:
            if 'forced_coordination' in j and 'rewards' in j:
                domain_rewards = []
                with open(j) as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for lines in csv_reader:
                        for k in lines:
                            domain_rewards.append(float(k.strip('][').split(',')[0]))

                participant_data[participant][condition]['domain_1']['rewards'] = domain_rewards

            elif 'two_rooms_narrow' in j and 'rewards' in j:
                domain_rewards = []
                with open(j) as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for lines in csv_reader:
                        for k in lines:
                            domain_rewards.append(float(k.strip('][').split(',')[0]))

                participant_data[participant][condition]['domain_2']['rewards'] = domain_rewards

                if participant == '16':
                    participant_data[participant][condition]['domain_1']['rewards'] = [246, 317, 317, 317]

        else:
            # there is a pt
            import torch

            reward_data = torch.load(rootdir + 'rewards.pt')

            # TODO: add support for tutorial everywhere
            # participant_data[participant][condition]['domain_0']['rewards'] = reward_data['forced_coordination']
            participant_data[participant][condition]['domain_1']['rewards'] = reward_data['forced_coordination']
            participant_data[participant][condition]['domain_2']['rewards'] = reward_data['two_rooms_narrow']

        if 'forced_coordination' in j and 'rewards' not in j and condition != 'FCP':
            domain_workloads = []
            with open(j) as csv_file:
                csv_reader = csv.reader(csv_file)
                for line in csv_reader:
                    domain_workloads.append(int(line[3]) + int(line[4]) + int(line[5]) + int(line[7]) + int(line[8]))

            participant_data[participant][condition]['domain_1']['workload'] = domain_workloads

        elif 'two_rooms_narrow' in j and 'rewards' not in j and condition not in ['No modification (Black-Box)', 'FCP']:
            domain_workloads = []
            with open(j) as csv_file:
                csv_reader = csv.reader(csv_file)
                for line in csv_reader:
                    domain_workloads.append(int(line[3]) + int(line[4]) + int(line[5]) + int(line[7]) + int(line[8]))

            participant_data[participant][condition]['domain_2']['workload'] = domain_workloads
            if len(domain_workloads) != 4:
                print(j, 'problem')
        elif 'rewards' not in j and condition in ['No modification (Black-Box)', 'FCP']:
            domain_workloads = []
            with open(j) as csv_file:
                csv_reader = csv.reader(csv_file)
                for line in csv_reader:
                    domain_workloads.append(int(line[3]) + int(line[4]) + int(line[5]) + int(line[7]) + int(line[8]))

            if len(domain_workloads) != 8:
                print('problem')
            if participant_data[participant]['domain_ordering'] == 'normal':
                participant_data[participant][condition]['domain_1']['workload'] = domain_workloads[0:4]
                participant_data[participant][condition]['domain_2']['workload'] = domain_workloads[4:8]
            else:
                participant_data[participant][condition]['domain_1']['workload'] = domain_workloads[4:8]
                participant_data[participant][condition]['domain_2']['workload'] = domain_workloads[0:4]

print('hi')


# start coding plotters
def get_all_rewards(participant_data, domain, condition, it):
    rewards = []

    conditions = {1: 'Human Modifies Tree',
                  2: 'Optimization',
                  3: 'No modification (Black-Box)',
                  4: 'No modification (Interpretable)',
                  5: 'FCP'}
    condition = conditions[condition]

    for i in participant_data:
        if i in errored_out:
            continue
        if condition in participant_data[i].keys():
            try:
                # print('Participant ', i, 'in condition ', condition, 'has rewards',
                #       participant_data[i][condition][domain]['rewards'])
                if type(it) is int:
                    rewards.append(participant_data[i][condition][domain]['rewards'][it])
                    if participant_data[i][condition][domain]['rewards'][it] < 100:
                        print(i, condition, domain, it, participant_data[i][condition][domain]['rewards'][it])
                else:
                    rewards.append(max(participant_data[i][condition][domain]['rewards']))
                    if max(participant_data[i][condition][domain]['rewards']) == 477 and condition == 'FCP':
                        print(i, 'who did best in FCP d1')
                    if max(participant_data[i][condition][domain]['rewards']) == 325 and condition == 'FCP':
                        print(i, 'who did best in FCP d2')
                    if max(participant_data[i][condition][domain]['rewards']) == 480:
                        print(i, 'who did best modifying tree d1')
                    if max(participant_data[i][condition][domain]['rewards']) == 256:
                        print(i, 'who did best modifying tree d2')

            except KeyError:
                print('kk')

    return rewards


def get_all_workloads(participant_data, domain, condition, it):
    workloads = []

    conditions = {1: 'Human Modifies Tree',
                  2: 'Optimization',
                  3: 'No modification (Black-Box)',
                  4: 'No modification (Interpretable)',
                  5: 'FCP'}
    condition = conditions[condition]

    for i in participant_data:
        if i in errored_out:
            continue
        if condition in participant_data[i].keys():
            try:
                # print('Participant ', i, 'in condition ', condition, 'has rewards',
                #       participant_data[i][condition][domain]['rewards'])
                if type(it) is int:
                    workloads.append(participant_data[i][condition][domain]['workload'][it])
                else:
                    workloads.append(max(participant_data[i][condition][domain]['workload']))
                    # if max(participant_data[i][condition][domain]['domain_workload']) == 477 and condition == 'FCP':
                    #     print(i, 'who did best in FCP d1')
                    # if max(participant_data[i][condition][domain]['domain_workload']) == 325 and condition == 'FCP':
                    #     print(i, 'who did best in FCP d2')
                    # if max(participant_data[i][condition][domain]['domain_workload']) == 480:
                    #     print(i, 'who did best modifying tree d1')
                    # if max(participant_data[i][condition][domain]['rewards']) == 256:
                    #     print(i, 'who did best modifying tree d2')
            except IndexError:
                print('kk', i, condition, domain)
            except KeyError:
                print('kk', i, condition, domain)

    return workloads


def generate_word_cloud(wanted_condition, wanted_domain):
    with open(
            '/home/rohanpaleja/Downloads/combined_post_data/combined_post_data.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        words = []
        for e, i in enumerate(csv_reader):
            # print(i)
            if i['DistributionChannel'] == 'preview':
                continue
            if e <= 1:
                continue
            else:
                participant_id = i['Q16']

                if int(participant_id) in not_included or participant_id in errored_out:
                    continue

                condition_mapping = {'A': 'Human Modifies Tree',
                                     'B': 'Optimization',
                                     'C': 'No modification (Black-Box)',
                                     'D': 'No modification (Interpretable)',
                                     'E': 'FCP'}

                condition = condition_mapping[i['Q4']]
                # domain
                domain = 'domain_' + i['Q5']

                if participant_id == '2' and condition == wanted_condition:
                    # word data lost for p1, p4, p5, p7, p8, p10, p11, p12
                    if wanted_domain == 'domain_1':
                        words.append('predictable')
                        words.append('algorithmic')
                        words.append('typical')
                    else:
                        words.append('predictable')
                        words.append('focused')
                        words.append('bad')

                elif participant_id == '3' and condition == wanted_condition:
                    # word data lost for p1
                    if wanted_domain == 'domain_1':
                        words.append('intelligent')
                        words.append('adaptive')
                        words.append('predictive')
                    else:
                        words.append('individual')
                        words.append('intelligent')
                        words.append('smooth')

                elif participant_id == '6' and condition == wanted_condition:
                    # word data lost for p1
                    if wanted_domain == 'domain_1':
                        words.append('efficient')
                        words.append('collaborative')
                        words.append('obedient')
                    else:
                        words.append('unintelligent')
                        words.append('difficult')
                        words.append('slow')
                elif participant_id == '9' and condition == wanted_condition:
                    # word data lost for p1
                    if wanted_domain == 'domain_1':
                        words.append('efficient')
                        words.append('intelligent')
                        words.append('helpful')
                    else:
                        words.append('intelligent')
                        words.append('helpful')
                        words.append('good teammate')
                elif participant_id == '13' and condition == wanted_condition:
                    # word data lost for p1
                    if wanted_domain == 'domain_1':
                        words.append('consistent')
                        words.append('useful')
                        words.append('dependable')
                    else:
                        words.append('stategic')
                        words.append('predictive')
                        words.append('rigid')
                elif participant_id == '14' and condition == wanted_condition:
                    # word data lost for p1
                    if wanted_domain == 'domain_1':
                        words.append('probabilistic')
                        words.append('programmatic')
                        words.append('suboptimal')
                    else:
                        words.append('inadequate')
                        words.append('stupid')
                        words.append('idiot sandwich')
                else:
                    if condition == wanted_condition and domain == wanted_domain:
                        words.append(i['Q4_1_TEXT'])
                        words.append(i['Q4_2_TEXT'])
                        words.append(i['Q4_3_TEXT'])
        from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

        # print(words)
        unique_string = (" ").join(words)

        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sentiment = SentimentIntensityAnalyzer()
        sent_1 = sentiment.polarity_scores(unique_string)
        print(sent_1, 'for condition', wanted_condition, 'and domain', wanted_domain)

        wordcloud = WordCloud(width=1000, height=500).generate(unique_string)
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud)
        plt.title('Word Cloud for ' + wanted_condition + ' ' + wanted_domain, fontsize=25)
        plt.axis("off")
        plt.show()




def check_individual_data_completion(participant_data):
    # workload of length 4 per domain
    # rewards of length 4 per domain
    pass


# function to add value labels
# function to add value labels
def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i] // 2, y[i], ha='center')


def plot_change_in_rewards(participant_data, domain, iteration=None, save=False, show=False):
    """

    Args:
        participant_data:
        domain:
        save:
        show:

    Returns:

    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    # plt.rcParams['axes.linewidth'] = 2
    # matplotlib.rcParams.update({'font.size': 18})
    # fig, ax = plt.subplots(figsize=(15, 10))
    # index = np.arange(1)
    # bar_width = 0.17
    # opacity = 0.8
    # error_config = {'ecolor': '0.3'}
    if domain == 'both':


        if type(iteration) is int:
            data_1 = [get_all_rewards(participant_data, 'domain_1', condition=1, it=iteration ),
                      get_all_rewards(participant_data, 'domain_1', condition=2, it=iteration ),
                      get_all_rewards(participant_data, 'domain_1', condition=3, it=iteration ),
                      get_all_rewards(participant_data, 'domain_1', condition=4, it=iteration ),
                      get_all_rewards(participant_data, 'domain_1', condition=5, it=iteration )]
            data_2 = [get_all_rewards(participant_data, 'domain_2', condition=1, it=iteration ),
                      get_all_rewards(participant_data, 'domain_2', condition=2, it=iteration ),
                      get_all_rewards(participant_data, 'domain_2', condition=3, it=iteration ),
                      get_all_rewards(participant_data, 'domain_2', condition=4, it=iteration ),
                      get_all_rewards(participant_data, 'domain_2', condition=5, it=iteration )]

            data_11 = [get_all_rewards(participant_data, 'domain_1', condition=1, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_1', condition=2, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_1', condition=3, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_1', condition=4, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_1', condition=5, it=iteration - 1)]
            data_22 = [get_all_rewards(participant_data, 'domain_2', condition=1, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_2', condition=2, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_2', condition=3, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_2', condition=4, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_2', condition=5, it=iteration - 1)]

            domains = ['Forced Coordination', 'Two Rooms Narrow']
            conditions = ['Human \nModifies Tree', 'Optimization', 'No modification\n (Black-Box)',
                          'No modification\n (Interpretable)', 'FCP']

            condition_A_data = [[a - b for a, b in zip(data_1[0], data_11[0])], [a - b for a, b in zip(data_2[0], data_22[0])]]
            condition_B_data = [[a - b for a, b in zip(data_1[1], data_11[1])], [a - b for a, b in zip(data_2[1], data_22[1])]]
            condition_C_data = [[a - b for a, b in zip(data_1[2], data_11[2])], [a - b for a, b in zip(data_2[2], data_22[2])]]
            condition_D_data = [[a - b for a, b in zip(data_1[3], data_11[3])], [a - b for a, b in zip(data_2[3], data_22[3])]]
            condition_E_data = [[a - b for a, b in zip(data_1[4], data_11[4])], [a - b for a, b in zip(data_2[4], data_22[4])]]


            value_condition_A = [np.mean(i) for i in condition_A_data]  # Values for each condition
            errors_condition_A = [np.std(i) / np.sqrt(len(i)) for i in condition_A_data]

            value_condition_B = [np.mean(i) for i in condition_B_data]  # Values for each condition
            errors_condition_B = [np.std(i) / np.sqrt(len(i)) for i in condition_B_data]

            value_condition_C = [np.mean(i) for i in condition_C_data]  # Values for each condition
            errors_condition_C = [np.std(i) / np.sqrt(len(i)) for i in condition_C_data]

            value_condition_D = [np.mean(i) for i in condition_D_data]  # Values for each condition
            errors_condition_D = [np.std(i) / np.sqrt(len(i)) for i in condition_D_data]

            value_condition_E = [np.mean(i) for i in condition_E_data]  # Values for each condition
            errors_condition_E = [np.std(i) / np.sqrt(len(i)) for i in condition_E_data]

            X_axis = np.arange(len(domains))

            plt.figure(figsize=(10, 6))  # Increase the figure size (width=10, height=6)

            plt.bar(X_axis - 0.2, value_condition_A, yerr=errors_condition_A, capsize=4, width=.1,
                    label=conditions[0] + ' n=' + str(len(condition_A_data[0])))
            plt.bar(X_axis - 0.1, value_condition_B, yerr=errors_condition_B, capsize=4, width=.1,
                    label=conditions[1] + ' n=' + str(len(condition_B_data[0])))
            plt.bar(X_axis, value_condition_C, yerr=errors_condition_C, capsize=4, width=.1,
                    label=conditions[2] + ' n=' + str(len(condition_C_data[0])))
            plt.bar(X_axis + 0.1, value_condition_D, yerr=errors_condition_D, capsize=4, width=.1,
                    label=conditions[3] + ' n=' + str(len(condition_D_data[0])))
            plt.bar(X_axis + 0.2, value_condition_E, yerr=errors_condition_E, capsize=4, width=.1,
                    label=conditions[4] + ' n=' + str(len(condition_E_data[0])))

            plt.xlabel('Domains')
            plt.ylabel('Rewards')
            plt.title('Rewards versus Conditions in Iteration ' + str(iteration) + ' with std error bars')
            plt.xticks(X_axis, domains)
            plt.legend()
            # plt.savefig('domain_2_rewards.png')
            if show:
                plt.show()

def plot_all_rewards(participant_data, domain, iteration=None, save=False, show=False):
    """

    Args:
        participant_data:
        domain:
        save:
        show:

    Returns:

    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    # plt.rcParams['axes.linewidth'] = 2
    # matplotlib.rcParams.update({'font.size': 18})
    # fig, ax = plt.subplots(figsize=(15, 10))
    # index = np.arange(1)
    # bar_width = 0.17
    # opacity = 0.8
    # error_config = {'ecolor': '0.3'}
    if domain == 'both':

        if iteration is None:
            rects1 = ax.bar(index, [np.mean(i) for i in data[0]], bar_width,
                            alpha=opacity, color='#FF6475',
                            yerr=[np.std(i) / np.sqrt(len(i)) for i in data[0]], error_kw=error_config,
                            edgecolor='black',
                            label='Iteration 1')

            rects2 = ax.bar(index + 1.1 * bar_width, [np.mean(i) for i in data[1]], bar_width,
                            alpha=opacity, color='#2FF77F',
                            yerr=[np.std(i) / np.sqrt(len(i)) for i in data[1]], error_kw=error_config,
                            edgecolor='black',
                            label='Iteration 2')

            rects3 = ax.bar(index + 2.2 * bar_width, [np.mean(i) for i in data[2]], bar_width,
                            alpha=opacity, color='#2AEDF2',
                            yerr=[np.std(i) / np.sqrt(len(i)) for i in data[2]], error_kw=error_config,
                            edgecolor='black',
                            label='Iteration 3')

            rects4 = ax.bar(index + 3.3 * bar_width, [np.mean(i) for i in data[2]], bar_width,
                            alpha=opacity, color='#2AEDF2',
                            yerr=[np.std(i) / np.sqrt(len(i)) for i in data[2]], error_kw=error_config,
                            edgecolor='black',
                            label='Iteration 4')
        elif type(iteration) is int:
            data_1 = [get_all_rewards(participant_data, 'domain_1', condition=1, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_1', condition=2, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_1', condition=3, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_1', condition=4, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_1', condition=5, it=iteration - 1)]
            data_2 = [get_all_rewards(participant_data, 'domain_2', condition=1, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_2', condition=2, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_2', condition=3, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_2', condition=4, it=iteration - 1),
                      get_all_rewards(participant_data, 'domain_2', condition=5, it=iteration - 1)]

            domains = ['Forced Coordination', 'Two Rooms Narrow']
            conditions = ['Human \nModifies Tree', 'Optimization', 'No modification\n (Black-Box)',
                          'No modification\n (Interpretable)', 'FCP']

            condition_A_data = [data_1[0], data_2[0]]
            condition_B_data = [data_1[1], data_2[1]]
            condition_C_data = [data_1[2], data_2[2]]
            condition_D_data = [data_1[3], data_2[3]]
            condition_E_data = [data_1[4], data_2[4]]

            value_condition_A = [np.mean(i) for i in condition_A_data]  # Values for each condition
            errors_condition_A = [np.std(i) / np.sqrt(len(i)) for i in condition_A_data]

            value_condition_B = [np.mean(i) for i in condition_B_data]  # Values for each condition
            errors_condition_B = [np.std(i) / np.sqrt(len(i)) for i in condition_B_data]

            value_condition_C = [np.mean(i) for i in condition_C_data]  # Values for each condition
            errors_condition_C = [np.std(i) / np.sqrt(len(i)) for i in condition_C_data]

            value_condition_D = [np.mean(i) for i in condition_D_data]  # Values for each condition
            errors_condition_D = [np.std(i) / np.sqrt(len(i)) for i in condition_D_data]

            value_condition_E = [np.mean(i) for i in condition_E_data]  # Values for each condition
            errors_condition_E = [np.std(i) / np.sqrt(len(i)) for i in condition_E_data]

            X_axis = np.arange(len(domains))

            plt.figure(figsize=(10, 6))  # Increase the figure size (width=10, height=6)

            plt.bar(X_axis - 0.2, value_condition_A, yerr=errors_condition_A, capsize=4, width=.1,
                    label=conditions[0] + ' n=' + str(len(condition_A_data[0])))
            plt.bar(X_axis - 0.1, value_condition_B, yerr=errors_condition_B, capsize=4, width=.1,
                    label=conditions[1] + ' n=' + str(len(condition_B_data[0])))
            plt.bar(X_axis, value_condition_C, yerr=errors_condition_C, capsize=4, width=.1,
                    label=conditions[2] + ' n=' + str(len(condition_C_data[0])))
            plt.bar(X_axis + 0.1, value_condition_D, yerr=errors_condition_D, capsize=4, width=.1,
                    label=conditions[3] + ' n=' + str(len(condition_D_data[0])))
            plt.bar(X_axis + 0.2, value_condition_E, yerr=errors_condition_E, capsize=4, width=.1,
                    label=conditions[4] + ' n=' + str(len(condition_E_data[0])))

            plt.xlabel('Domains')
            plt.ylabel('Rewards')
            plt.title('Rewards versus Conditions in Iteration ' + str(iteration) + ' with std error bars')
            plt.xticks(X_axis, domains)
            plt.legend()
            # plt.savefig('domain_2_rewards.png')
            if show:
                plt.show()

        elif iteration == 'm':

            data_1 = [get_all_rewards(participant_data, 'domain_1', condition=1, it='m'),
                      get_all_rewards(participant_data, 'domain_1', condition=2, it='m'),
                      get_all_rewards(participant_data, 'domain_1', condition=3, it='m'),
                      get_all_rewards(participant_data, 'domain_1', condition=4, it='m'),
                      get_all_rewards(participant_data, 'domain_1', condition=5, it='m')]
            data_2 = [get_all_rewards(participant_data, 'domain_2', condition=1, it='m'),
                      get_all_rewards(participant_data, 'domain_2', condition=2, it='m'),
                      get_all_rewards(participant_data, 'domain_2', condition=3, it='m'),
                      get_all_rewards(participant_data, 'domain_2', condition=4, it='m'),
                      get_all_rewards(participant_data, 'domain_2', condition=5, it='m')]

            domains = ['Forced Coordination', 'Two Rooms Narrow']
            conditions = ['Human \nModifies Tree', 'Optimization', 'No modification\n (Black-Box)',
                          'No modification\n (Interpretable)', 'FCP']

            condition_A_data = [data_1[0], data_2[0]]
            condition_B_data = [data_1[1], data_2[1]]
            condition_C_data = [data_1[2], data_2[2]]
            condition_D_data = [data_1[3], data_2[3]]
            condition_E_data = [data_1[4], data_2[4]]

            value_condition_A = [np.mean(i) for i in condition_A_data]  # Values for each condition
            errors_condition_A = [np.std(i) / np.sqrt(len(i)) for i in condition_A_data]

            value_condition_B = [np.mean(i) for i in condition_B_data]  # Values for each condition
            errors_condition_B = [np.std(i) / np.sqrt(len(i)) for i in condition_B_data]

            value_condition_C = [np.mean(i) for i in condition_C_data]  # Values for each condition
            errors_condition_C = [np.std(i) / np.sqrt(len(i)) for i in condition_C_data]

            value_condition_D = [np.mean(i) for i in condition_D_data]  # Values for each condition
            errors_condition_D = [np.std(i) / np.sqrt(len(i)) for i in condition_D_data]

            value_condition_E = [np.mean(i) for i in condition_E_data]  # Values for each condition
            errors_condition_E = [np.std(i) / np.sqrt(len(i)) for i in condition_E_data]

            X_axis = np.arange(len(domains))

            plt.figure(figsize=(10, 6))  # Increase the figure size (width=10, height=6)

            plt.bar(X_axis - 0.2, value_condition_A, yerr=errors_condition_A, capsize=4, width=.1, label=conditions[0])
            plt.bar(X_axis - 0.1, value_condition_B, yerr=errors_condition_B, capsize=4, width=.1, label=conditions[1])
            plt.bar(X_axis, value_condition_C, yerr=errors_condition_C, capsize=4, width=.1, label=conditions[2])
            plt.bar(X_axis + 0.1, value_condition_D, yerr=errors_condition_D, capsize=4, width=.1, label=conditions[3])
            plt.bar(X_axis + 0.2, value_condition_E, yerr=errors_condition_E, capsize=4, width=.1, label=conditions[4])

            plt.xlabel('Domains')
            plt.ylabel('Rewards')
            plt.title('Max Reward versus Condtions with std error bars')
            plt.xticks(X_axis, domains)
            plt.legend()
            # plt.savefig('domain_2_rewards.png')
            if show:
                plt.show()

            print('max performance obtained in each condition', [np.max(i) for i in condition_A_data],
                  [np.max(i) for i in condition_B_data],
                  [np.max(i) for i in condition_C_data],
                  [np.max(i) for i in condition_D_data],
                  [np.max(i) for i in condition_E_data])

    else:
        raise NotImplementedError


def plot_all_workloads(participant_data, domain, iteration=None, save=False, show=False):
    """

    Args:
        participant_data:
        domain:
        save:
        show:

    Returns:

    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    # plt.rcParams['axes.linewidth'] = 2
    # matplotlib.rcParams.update({'font.size': 18})
    # fig, ax = plt.subplots(figsize=(15, 10))
    # index = np.arange(1)
    # bar_width = 0.17
    # opacity = 0.8
    # error_config = {'ecolor': '0.3'}
    if domain == 'both':

        if iteration is None:
            rects1 = ax.bar(index, [np.mean(i) for i in data[0]], bar_width,
                            alpha=opacity, color='#FF6475',
                            yerr=[np.std(i) / np.sqrt(len(i)) for i in data[0]], error_kw=error_config,
                            edgecolor='black',
                            label='Iteration 1')

            rects2 = ax.bar(index + 1.1 * bar_width, [np.mean(i) for i in data[1]], bar_width,
                            alpha=opacity, color='#2FF77F',
                            yerr=[np.std(i) / np.sqrt(len(i)) for i in data[1]], error_kw=error_config,
                            edgecolor='black',
                            label='Iteration 2')

            rects3 = ax.bar(index + 2.2 * bar_width, [np.mean(i) for i in data[2]], bar_width,
                            alpha=opacity, color='#2AEDF2',
                            yerr=[np.std(i) / np.sqrt(len(i)) for i in data[2]], error_kw=error_config,
                            edgecolor='black',
                            label='Iteration 3')

            rects4 = ax.bar(index + 3.3 * bar_width, [np.mean(i) for i in data[2]], bar_width,
                            alpha=opacity, color='#2AEDF2',
                            yerr=[np.std(i) / np.sqrt(len(i)) for i in data[2]], error_kw=error_config,
                            edgecolor='black',
                            label='Iteration 4')
        elif type(iteration) is int:
            data_1 = [get_all_workloads(participant_data, 'domain_1', condition=1, it=iteration - 1),
                      get_all_workloads(participant_data, 'domain_1', condition=2, it=iteration - 1),
                      get_all_workloads(participant_data, 'domain_1', condition=3, it=iteration - 1),
                      get_all_workloads(participant_data, 'domain_1', condition=4, it=iteration - 1),
                      get_all_workloads(participant_data, 'domain_1', condition=5, it=iteration - 1)]
            data_2 = [get_all_workloads(participant_data, 'domain_2', condition=1, it=iteration - 1),
                      get_all_workloads(participant_data, 'domain_2', condition=2, it=iteration - 1),
                      get_all_workloads(participant_data, 'domain_2', condition=3, it=iteration - 1),
                      get_all_workloads(participant_data, 'domain_2', condition=4, it=iteration - 1),
                      get_all_workloads(participant_data, 'domain_2', condition=5, it=iteration - 1)]

            domains = ['Forced Coordination', 'Two Rooms Narrow']
            conditions = ['Human \nModifies Tree', 'Optimization', 'No modification\n (Black-Box)',
                          'No modification\n (Interpretable)', 'FCP']

            condition_A_data = [data_1[0], data_2[0]]
            condition_B_data = [data_1[1], data_2[1]]
            condition_C_data = [data_1[2], data_2[2]]
            condition_D_data = [data_1[3], data_2[3]]
            condition_E_data = [data_1[4], data_2[4]]

            value_condition_A = [np.mean(i) for i in condition_A_data]  # Values for each condition
            errors_condition_A = [np.std(i) / np.sqrt(len(i)) for i in condition_A_data]

            value_condition_B = [np.mean(i) for i in condition_B_data]  # Values for each condition
            errors_condition_B = [np.std(i) / np.sqrt(len(i)) for i in condition_B_data]

            value_condition_C = [np.mean(i) for i in condition_C_data]  # Values for each condition
            errors_condition_C = [np.std(i) / np.sqrt(len(i)) for i in condition_C_data]

            value_condition_D = [np.mean(i) for i in condition_D_data]  # Values for each condition
            errors_condition_D = [np.std(i) / np.sqrt(len(i)) for i in condition_D_data]

            value_condition_E = [np.mean(i) for i in condition_E_data]  # Values for each condition
            errors_condition_E = [np.std(i) / np.sqrt(len(i)) for i in condition_E_data]

            X_axis = np.arange(len(domains))

            plt.figure(figsize=(10, 6))  # Increase the figure size (width=10, height=6)

            plt.bar(X_axis - 0.2, value_condition_A, yerr=errors_condition_A, capsize=4, width=.1,
                    label=conditions[0] + ' n=' + str(len(condition_A_data[0])))
            plt.bar(X_axis - 0.1, value_condition_B, yerr=errors_condition_B, capsize=4, width=.1,
                    label=conditions[1] + ' n=' + str(len(condition_B_data[0])))
            plt.bar(X_axis, value_condition_C, yerr=errors_condition_C, capsize=4, width=.1,
                    label=conditions[2] + ' n=' + str(len(condition_C_data[0])))
            plt.bar(X_axis + 0.1, value_condition_D, yerr=errors_condition_D, capsize=4, width=.1,
                    label=conditions[3] + ' n=' + str(len(condition_D_data[0])))
            plt.bar(X_axis + 0.2, value_condition_E, yerr=errors_condition_E, capsize=4, width=.1,
                    label=conditions[4] + ' n=' + str(len(condition_E_data[0])))

            plt.xlabel('Domains')
            plt.ylabel('Workload')
            plt.title('Workload versus Conditions in Iteration ' + str(iteration) + ' with std error bars')
            plt.xticks(X_axis, domains)
            plt.legend()
            # plt.savefig('domain_2_rewards.png')
            if show:
                plt.show()

        elif iteration == 'm':

            data_1 = [get_all_workloads(participant_data, 'domain_1', condition=1, it='m'),
                      get_all_workloads(participant_data, 'domain_1', condition=2, it='m'),
                      get_all_workloads(participant_data, 'domain_1', condition=3, it='m'),
                      get_all_workloads(participant_data, 'domain_1', condition=4, it='m'),
                      get_all_workloads(participant_data, 'domain_1', condition=5, it='m')]
            data_2 = [get_all_workloads(participant_data, 'domain_2', condition=1, it='m'),
                      get_all_workloads(participant_data, 'domain_2', condition=2, it='m'),
                      get_all_workloads(participant_data, 'domain_2', condition=3, it='m'),
                      get_all_workloads(participant_data, 'domain_2', condition=4, it='m'),
                      get_all_workloads(participant_data, 'domain_2', condition=5, it='m')]

            domains = ['Forced Coordination', 'Two Rooms Narrow']
            conditions = ['Human \nModifies Tree', 'Optimization', 'No modification\n (Black-Box)',
                          'No modification\n (Interpretable)', 'FCP']

            condition_A_data = [data_1[0], data_2[0]]
            condition_B_data = [data_1[1], data_2[1]]
            condition_C_data = [data_1[2], data_2[2]]
            condition_D_data = [data_1[3], data_2[3]]
            condition_E_data = [data_1[4], data_2[4]]

            value_condition_A = [np.mean(i) for i in condition_A_data]  # Values for each condition
            errors_condition_A = [np.std(i) / np.sqrt(len(i)) for i in condition_A_data]

            value_condition_B = [np.mean(i) for i in condition_B_data]  # Values for each condition
            errors_condition_B = [np.std(i) / np.sqrt(len(i)) for i in condition_B_data]

            value_condition_C = [np.mean(i) for i in condition_C_data]  # Values for each condition
            errors_condition_C = [np.std(i) / np.sqrt(len(i)) for i in condition_C_data]

            value_condition_D = [np.mean(i) for i in condition_D_data]  # Values for each condition
            errors_condition_D = [np.std(i) / np.sqrt(len(i)) for i in condition_D_data]

            value_condition_E = [np.mean(i) for i in condition_E_data]  # Values for each condition
            errors_condition_E = [np.std(i) / np.sqrt(len(i)) for i in condition_E_data]

            X_axis = np.arange(len(domains))

            plt.figure(figsize=(10, 6))  # Increase the figure size (width=10, height=6)

            plt.bar(X_axis - 0.2, value_condition_A, yerr=errors_condition_A, capsize=4, width=.1, label=conditions[0])
            plt.bar(X_axis - 0.1, value_condition_B, yerr=errors_condition_B, capsize=4, width=.1, label=conditions[1])
            plt.bar(X_axis, value_condition_C, yerr=errors_condition_C, capsize=4, width=.1, label=conditions[2])
            plt.bar(X_axis + 0.1, value_condition_D, yerr=errors_condition_D, capsize=4, width=.1, label=conditions[3])
            plt.bar(X_axis + 0.2, value_condition_E, yerr=errors_condition_E, capsize=4, width=.1, label=conditions[4])

            plt.xlabel('Domains')
            plt.ylabel('Workload Scores')
            plt.title('Max Workload versus Condtions with std error bars')
            plt.xticks(X_axis, domains)
            plt.legend()
            # plt.savefig('domain_2_rewards.png')
            if show:
                plt.show()

            print('max performance obtained in each condition', [np.max(i) for i in condition_A_data],
                  [np.max(i) for i in condition_B_data],
                  [np.max(i) for i in condition_C_data],
                  [np.max(i) for i in condition_D_data],
                  [np.max(i) for i in condition_E_data])

    else:
        raise NotImplementedError


# plot_all_rewards(participant_data, domain='domain_1',iteration=1, save=False, show=True)
# plot_all_rewards(participant_data, domain='domain_1',iteration=2, save=False, show=True)
# plot_all_rewards(participant_data, domain='domain_1',iteration=3, save=False, show=True)
# plot_all_rewards(participant_data, domain='domain_1',iteration=4, save=False, show=True)
# plot_all_rewards(participant_data, domain='domain_1',iteration='m', save=False, show=True)
#
# plot_all_rewards(participant_data, domain='domain_1',iteration=1, save=False, show=True)
# plot_all_rewards(participant_data, domain='domain_1',iteration=2, save=False, show=True)
# plot_all_rewards(participant_data, domain='domain_1',iteration=3, save=False, show=True)
# plot_all_rewards(participant_data, domain='domain_1',iteration=4, save=False, show=True)
# plot_all_rewards(participant_data, domain='domain_1',iteration='m', save=False, show=True)

plot_all_rewards(participant_data, domain='both', iteration=1, save=False, show=True)
plot_all_rewards(participant_data, domain='both', iteration=2, save=False, show=True)
plot_all_rewards(participant_data, domain='both', iteration=3, save=False, show=True)
plot_all_rewards(participant_data, domain='both', iteration=4, save=False, show=True)
# plot_all_rewards(participant_data, domain='both', iteration='m', save=False, show=True)

# plot_change_in_rewards(participant_data, domain='both', iteration=1, save=False, show=True)
# plot_change_in_rewards(participant_data, domain='both', iteration=2, save=False, show=True)
# plot_change_in_rewards(participant_data, domain='both', iteration=3, save=False, show=True)
#

# generate_word_cloud("Human Modifies Tree", 'domain_1')
# generate_word_cloud("Optimization", 'domain_1')
# generate_word_cloud("No modification (Black-Box)", 'domain_1')
# generate_word_cloud("No modification (Interpretable)", 'domain_1')
# generate_word_cloud("FCP", 'domain_1')
#
# generate_word_cloud("Human Modifies Tree", 'domain_2')
# generate_word_cloud("Optimization", 'domain_2')
# generate_word_cloud("No modification (Black-Box)", 'domain_2')
# generate_word_cloud("No modification (Interpretable)", 'domain_2')
# generate_word_cloud("FCP", 'domain_2')

# plot_all_workloads(participant_data, domain='both', iteration=1, save=False, show=True)
# plot_all_workloads(participant_data, domain='both', iteration=2, save=False, show=True)
# plot_all_workloads(participant_data, domain='both', iteration=3, save=False, show=True)
# plot_all_workloads(participant_data, domain='both', iteration=4, save=False, show=True)
# plot_all_workloads(participant_data, domain='both', iteration='m', save=False, show=True)
