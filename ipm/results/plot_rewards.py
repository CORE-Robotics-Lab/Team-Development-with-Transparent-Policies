import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2


# # load in data
best_rewards = [854.9090909090909 ,880.010101010101, 862.5555555555555, 953.5252525252525, 955.6969696969697
]


labels = ['8 leaf IDCT', '16 leaf IDCT', '32 leaf IDCT', '64 leaf IDCT', "NN"]

plt.bar(labels,best_rewards)
plt.xlabel('AI Model Representation')
plt.ylabel('Best reward')
plt.show()


# plotting pruning results
# load in data
# pre_pruning = [64 ,32, 16, 8]
# post_pruning = [3, 2, 3, 2]
# labels = ['64 leaf', '32 leaf', '16 leaf', '8 leaf']
# x = np.arange(len(labels))
# width = 0.35 # width of bars
#
# # Creating the bar plot
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, pre_pruning, width, label='Trained Model')
# rects2 = ax.bar(x + width/2, post_pruning, width, label='After Pruning')
#
#
# # Add the values to the bars
# for rect in rects1:
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#             '%d' % int(height),
#             ha='center', va='bottom')
# for rect in rects2:
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#             '%d' % int(height),
#             ha='center', va='bottom')
#
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Tree Size')
# ax.set_xlabel('Original Tree Size')
# # ax.set_title('Two Bars per Label')
# # ax.grid()
# ax.set_ylim(0, 80)
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
#
# # Display the plot
# plt.show()


