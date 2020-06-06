import matplotlib.pyplot as plt
import pickle
import matplotlib
import numpy as np

labels,average_cluster_distances,precision,recall,transition_precision,base_model,score,perfect_score,original_score,mode_score,background_score = pickle.load(open('model_performance.pkl','rb'))


x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
# rects1 = ax.bar(x - 2*width, np.array(average_cluster_distances)/10, width, label='average cluster distance/10')
# rects2 = ax.bar(x-width, precision, width, label='precision')
# rects3 = ax.bar(x , recall, width, label='recall')
# rects4 = ax.bar(x + width, transition_precision, width, label='transition precision')
# rects5 = ax.bar(x + width*2, base_model, width, label='top5')
rects1 = ax.bar(x - 2*width, score, width, label='score from perfect cluster')
rects3 = ax.bar(x-width ,  original_score,width, label='score from original')
rects2 = ax.bar(x, perfect_score, width, label='theoretical_best(perfect cluster)')
rects4 = ax.bar(x + width, mode_score, width, label='mode_score')
rects5 = ax.bar(x + width*2, background_score, width, label='white_background')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance')
ax.set_title('Performance of Model on Program')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()



plt.show()









