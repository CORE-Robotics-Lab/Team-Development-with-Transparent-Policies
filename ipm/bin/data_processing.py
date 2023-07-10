import pickle
import os
import csv
import numpy as np
import re
import matplotlib.pyplot as plt


# TODOS
# TODO: add in arthur's i
# TODO: everytime you reload, change Kin Man data to 6 in post
# TODO: add not included stuff


def find_male_female_split(participant_data):
    male = 0
    female = 0
    for i in participant_data:
        if participant_data[i]['gender'] == 'Male':
            male += 1
        elif participant_data[i]['gender'] == 'Female':
            female += 1
        else:
            # print(participant_data[k]['gender'])
            exit(0)
            print('you missed something')
    return male, female


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


def count_each_condition(participant_data):
    conditions = {'Human Modifies Tree': 0,
                  'Optimization': 0,
                  'No modification (Black-Box)': 0,
                  'No modification (Interpretable)': 0,
                  'FCP': 0}

    for i in participant_data:
        if 'Human Modifies Tree' in participant_data[i].keys():
            conditions['Human Modifies Tree'] += 1
        elif 'Optimization' in participant_data[i].keys():
            conditions['Optimization'] += 1
        elif 'No modification (Black-Box)' in participant_data[i].keys():
            conditions['No modification (Black-Box)'] += 1
        elif 'No modification (Interpretable)' in participant_data[i].keys():
            conditions['No modification (Interpretable)'] += 1
        elif 'FCP' in participant_data[i].keys():
            conditions['FCP'] += 1
        else:
            print('error')

    return conditions


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
                print('error')
                return None
        else:
            continue


# participants that must be ignored
not_included = [0]
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
        if int(i['Q1']) in not_included:
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
male, female = find_male_female_split(participant_data)
min_age, max_age, age_mean, age_std = find_min_max_age(participant_data)
print('Male is ', male, '/', total_participants)
print('Female is ', female, '/', total_participants)
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
        '/home/rohanpaleja/Downloads/Post-Experiment+COMBINE+Survey+IntPolMod_July+2,+2023_12.45/Post-Experiment COMBINE Survey IntPolMod_July 2, 2023_12.45.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for e, i in enumerate(csv_reader):
        print(i)
        if i['DistributionChannel'] == 'preview':
            continue
        if e <= 1:
            continue
        else:
            mapping = {'Strongly disagree': 1, 'Disagree': 2, 'Somewhat disagree': 3, 'Neither agree nor disagree': 4,
                       'Somewhat agree': 5, 'Agree': 6, 'Strongly agree': 7}
            participant_id = i['Q16']

            # condition

            condition_mapping = {'A': 'Human Modifies Tree',
                                 'B': 'Optimization',
                                 'C': 'No modification (Black-Box)',
                                 'D': 'No modification (Interpretable)',
                                 'E': 'FCP'}

            condition = condition_mapping[i['Q4']]

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

print(count_each_condition(participant_data))

# objective results

# go through each participant and add their rewards, times, and nasatlx
for participant in participant_data.keys():
    # later people had .pt data

    # locate some files
    import os

    # get participant condition
    condition = get_participant_condition(participant_data, participant)

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
    import os
    from glob import glob

    result = [y for x in os.walk(rootdir) for y in glob(os.path.join(x[0], '*.txt'))]
    # print('result is', result)
    # do fc first
    for j in result:
        if participant == '12':
            continue
        if participant == '17':
            import torch
            reward_data = torch.load(rootdir + 'user_18/rewards.pt')

            participant_data[participant][condition]['domain_1']['rewards'] = [391, 405, 465, 468]
            participant_data[participant][condition]['domain_2']['rewards'] = reward_data['two_rooms_narrow']
            continue

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

        elif 'two_rooms_narrow' in j and 'rewards' not in j and condition != 'FCP':
            domain_workloads = []
            with open(j) as csv_file:
                csv_reader = csv.reader(csv_file)
                for line in csv_reader:
                    domain_workloads.append(int(line[3]) + int(line[4]) + int(line[5]) + int(line[7]) + int(line[8]))

            participant_data[participant][condition]['domain_2']['workload'] = domain_workloads

        elif 'rewards' not in j and condition == 'FCP':
            domain_workloads = []
            with open(j) as csv_file:
                csv_reader = csv.reader(csv_file)
                for line in csv_reader:
                    domain_workloads.append(int(line[3]) + int(line[4]) + int(line[5]) + int(line[7]) + int(line[8]))

            if participant_data[participant]['domain_ordering'] == 'normal':
                participant_data[participant][condition]['domain_1']['workload'] = domain_workloads[0:4]
                participant_data[participant][condition]['domain_2']['workload'] = domain_workloads[4:8]
            else:
                participant_data[participant][condition]['domain_1']['workload'] = domain_workloads[4:8]
                participant_data[participant][condition]['domain_2']['workload'] = domain_workloads[0:4]


print('hi')

# start coding plotters
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
    plt.rcParams['axes.labelsize'] = 34
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 2
    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(15, 10))
    index = np.arange(4)
    bar_width = 0.17
    opacity = 0.8
    error_config = {'ecolor': '0.3'}


    if iteration is None:
        rects1 = ax.bar(index, [np.mean(i) for i in data[0]], bar_width,
                        alpha=opacity, color='#FF6475',
                        yerr=[np.std(i) / np.sqrt(len(i)) for i in data[0]], error_kw=error_config, edgecolor='black',
                        label='Iteration 1')

        rects2 = ax.bar(index + 1.1 * bar_width, [np.mean(i) for i in data[1]], bar_width,
                        alpha=opacity, color='#2FF77F',
                        yerr=[np.std(i) / np.sqrt(len(i)) for i in data[1]], error_kw=error_config, edgecolor='black',
                        label='Iteration 2')

        rects3 = ax.bar(index + 2.2 * bar_width, [np.mean(i) for i in data[2]], bar_width,
                        alpha=opacity, color='#2AEDF2',
                        yerr=[np.std(i) / np.sqrt(len(i)) for i in data[2]], error_kw=error_config, edgecolor='black',
                        label='Iteration 3')

        rects4 = ax.bar(index + 3.3 * bar_width, [np.mean(i) for i in data[2]], bar_width,
                        alpha=opacity, color='#2AEDF2',
                        yerr=[np.std(i) / np.sqrt(len(i)) for i in data[2]], error_kw=error_config, edgecolor='black',
                        label='Iteration 4')
    elif iteration == 4:
        pass
    else:
        raise NotImplementedError




plot_all_rewards(participant_data, domain='domain_1',iteration=4, save=False, show=True)
