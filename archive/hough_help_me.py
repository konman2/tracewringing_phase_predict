# exec(open('hough_help_me.py').read())

# # dependencies:
from hough import HoughLine
import pickle 
import numpy as np 
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt
from matplotlib import cm

# # helper funcs
def plotHoughLines(houghlines, image, save=False, fname=None):
    fig, axes = plt.subplots(1,2, figsize=(7,7), sharex=True, sharey=True)
    fig.patch.set_visible(True)
    # fig.suptitle('gcc')
    ax = axes.ravel()
    ax[0].imshow(image, cmap=cm.gray_r)
    ax[0].set_title('Input Address Projection Space',fontsize=10)
    ax[0].set_ylabel('Projected Address',fontsize=10)
    ax[0].set_xlabel('1000 Instructions')
    ax[1].imshow(image * 0, cmap=cm.gray_r)
    for line in houghlines:
        (x1, y1), (x2, y2) = line
        p0, p1 = (x1, y1), (x2, y2)
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[1].set_xlim((0, image.shape[1]))
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_title('Probabilistic Hough Approximation',fontsize=10)
    ax[1].set_xlabel('1000 Instructions')
    fig.text(0.54, 0.02, 'gcc', ha='center', va='center',fontsize=12)
    for a in ax:
        a.set_adjustable('box-forced')
    plt.tight_layout()
    # plt.subplots_adjust(top=0.12,bottom=0.1)
    if save == True:
        plt.savefig('figs/final/'+fname+'.png')
    else:
        plt.show()

def invert(heatmap_matrix):
    sub = np.ones(heatmap_matrix.shape) * heatmap_matrix.max()
    return np.subtract(sub,heatmap_matrix)

# # get + prep rep phases
rep_phases = pickle.load(open('milc_rep_phases.pkl', 'rb'))
heatmap_matrix = rep_phases[0]

from skimage.feature import canny

orig = heatmap_matrix
alt = np.sqrt(np.sqrt(heatmap_matrix))
# alt = canny(heatmap_matrix, 1, 1, 25)

# fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
# ax[0].imshow(orig, cmap=cm.gray_r)
# ax[0].set_title('original heatmap')
# ax[1].imshow(alt, cmap=cm.gray_r)
# ax[1].set_title('altered for hough')
# plt.show()

# raise('')

lines = probabilistic_hough_line(alt,threshold=80,line_length=100,line_gap=5)
# plotHoughLines(lines,alt)

# raise('')
# measuring information:
# for every line, find min and max of x1, y1, x2, y2, w
# x1, y1, x2, y2 = [], [], [], []
# for i in lines:
#     x1.append(i[0][0])
#     y1.append(i[0][1])
#     x2.append(i[1][0])
#     y2.append(i[1][1])

# import math
# def next_power_of_2(x):
#     return 1 if x == 0 else 2**math.ceil(math.log2(x))

# bits_x1 = math.log2(next_power_of_2(np.max(x1)-np.min(x1)))
# bits_y1 = math.log2(next_power_of_2(np.max(y1)-np.min(y1)))
# bits_x2 = math.log2(next_power_of_2(np.max(x2)-np.min(x2)))
# bits_y2 = math.log2(next_power_of_2(np.max(y2)-np.min(y2)))

# for every line, find number of unique symbols for each of the above
# log2((max-min)/unique) = number of bits to represent 
# raise('')
from skimage.draw import line 
houghlines, hough_weights, hough_full_weight = [], [], []
filter_pc = 90
pc = filter_pc*len(lines)//100
print('num of lines: {}, num of lines to be removed: {}'.format(len(lines),pc))
for j in lines:
    if j[0][0] == j[1][0]:
        l = sorted(j, key=lambda x: x[1])
    else:
        l = sorted(j, key=lambda x: x[0])
    (x1,y1),(x2,y2) = l
    xx, yy = line(x1,y1,x2,y2)
    weight = np.mean(heatmap_matrix[yy,xx])
    houghlines.append((x1,y1,x2,y2,weight))
    hough_weights.append(weight)
    hough_full_weight.append( len(xx)*weight )

print('number of hough lines found: {}'.format(len(houghlines)))
ind = np.argpartition(hough_full_weight, pc)[:pc]

for i in sorted(ind, reverse=True):
    del houghlines[i]
    del hough_weights[i]
    del hough_full_weight[i]

sum_weight = sum(hough_weights)
all_hough = [HoughLine(x1,y1,x2,y2,weight/sum_weight) for (x1,y1,x2,y2,weight) in houghlines if weight > 0]

print('number of lines remaining: {}'.format(len(all_hough)))