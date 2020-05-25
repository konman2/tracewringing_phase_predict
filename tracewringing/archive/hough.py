#from __future__ import print
import math
import numpy as np
from skimage.transform import probabilistic_hough_line
from trace_generator import kBlkBits

class HoughLineRN(object):
    def __init__(self, low, high, weight):
        assert high >= low
        assert weight >= 0, weight
        self.low = low
        self.high = high
        self.weight =weight

    def inRange(self, x):
        return True

    def getAddress(self, x, window_size):
        bitBlockOffset = kBlkBits
        assert self.inRange(x)
        addrs = np.random.randint(self.low, high = self.high, size = window_size)
        return [hex(int(addr) << bitBlockOffset) for addr in addrs]

class HoughLine(object):

    def __init__(self, x1, y1, x2, y2, weight):
        assert x2 >= x1, '({}, {}), ({}, {})'.format(x1, y1, x2, y2)
        assert weight > 0, weight
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.weight = weight

    def inRange(self, x):
        return x >= min(self.x1, self.x2) and x <= max(self.x1, self.x2)

    def getAddress(self, x):
        bitBlockOffset = kBlkBits
        if not self.inRange(x):
            return None
        if self.x2 > self.x1:
            slp = (1. * self.y2 - self.y1) / (1. * self.x2 - self.x1)
            addr = int(np.rint(self.y1 + (x - self.x1) * slp))
        else:
            # Then this is a vertical line, i.e., inf slp.
            assert self.x2 == self.x1
            low = min(self.y1, self.y2)
            high = max(self.y1, self.y2)
            addr = int(np.random.randint(low, high=high+1))
        return hex(addr << bitBlockOffset)

    def dump(self):
        print('Hough line: ({}, {}) -- ({}, {}) ({})'.format(
            self.x1, self.y1, self.x2, self.y2, self.weight))

class HoughComputation(object):

    def getHoughLines(self,heatmap,threshold,line_length,line_gap,theta_factor_list,filter_percent):
        """ Calls out to hough transform lib to detect all hough lines in the heatmap.

            Args:
                heatmap:        from HeatmapGenerator.getHeatmapMatrix
                threshold:      int, optional
                line_length:    int, optional: 
                                Minimum accepted length of detected lines. 
                                *Increase* the parameter to extract longer lines.
                line_gap:       int, optional 
                                Maximum gap between pixels to still form a line. 
                                *Increase* the parameter to merge broken lines more 
                                aggresively.

            Returns:
                hough_lines: a list of HoughLine objects
            
            Notes:
                -- heatmap_matrix is white on black --> use for finding hough lines
                -- image is black on white --> use for get_weights, figures
                -- use prep4hough to find lines, BUT use original heatmap to find weights
        """
        from skimage.draw import line, line_aa
        import math

        # helper funcs:
        def invert(heatmap_matrix):
            ''' Inverts image. '''
            sub = np.ones(heatmap_matrix.shape) * heatmap_matrix.max()
            return np.subtract(sub,heatmap_matrix)

        assert heatmap.shape[0] > 0, 'heatmap has no y dimension (rows)'
        assert heatmap.shape[1] > 0, 'heatmap has no x dimension (columns)'
        heatmap_matrix = heatmap
        image = invert(heatmap_matrix)

        # these asserts might be useless if they are not nested. #TODO:
        for i,j in zip(range(heatmap_matrix.shape[0]), range(heatmap_matrix.shape[1])):
            assert heatmap_matrix[i, j] >= 0, '({}, {}): {} {}'.format(i, j, heatmap_matrix[i, j], heatmap_matrix.shape)
        for i,j in zip(range(image.shape[0]), range(image.shape[1])):
            assert image[i, j] >= 0, '({}, {}): {} {}'.format(i, j, image[i, j], image.shape)
        
        if len(theta_factor_list) == 2:
            theta = np.array([theta_factor_list[0]*math.pi/2,theta_factor_list[1]*math.pi])
        elif len(theta_factor_list) == 1:
            theta = np.array([theta_factor_list[0]*math.pi/2])
        else:
            print('wrong theta')
        lines = probabilistic_hough_line(heatmap_matrix,threshold=threshold,
                    line_length=line_length,theta=theta)
        houghlines, hough_weights, hough_full_weight = [], [], []
        pc = filter_percent * len(lines) // 100
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
            hough_full_weight.append( len(xx)* weight)
        ind = np.argpartition(hough_full_weight, pc)[:pc]
        for i in sorted(ind, reverse=True):
            del hough_full_weight[i]
            del houghlines[i]
            del hough_weights[i]
        
        sum_weight = sum(hough_weights)
        all_hough = [HoughLine(x1,y1,x2,y2,weight/sum_weight) for (x1,y1,x2,y2,weight) in houghlines if weight > 0]
        return all_hough, image

    def plotHoughLines(self, houghlines, image, image_alt, save=False, fname=None):
        """ Plots the Hough Lines after they are detected, alongside the heatmap.

            Args:
                houghlines: a list of HoughLine objects with the ((x1,y1),(x2,y2),
                            weight) information.
                image: an image of the heatmap_matrix (4th root for now)

            Returns:
                a matplotlib plot object.

        """
        import matplotlib.pyplot as plt
        from matplotlib import cm

        def invert(heatmap_matrix):
            ''' Inverts image. '''
            sub = np.ones(heatmap_matrix.shape) * heatmap_matrix.max()
            return np.subtract(sub,heatmap_matrix)

        if image.shape[1] > image.shape[0]:
            fig, axes = plt.subplots(3,1,figsize=(7,7), sharex=True, sharey=True)
        elif image.shape[0] > image.shape[1]:
            fig, axes = plt.subplots(1,3,figsize=(7,7), sharex=True, sharey=True)
        
        fig.patch.set_visible(True)
        ax = axes.ravel()
        ax[0].imshow(invert(np.sqrt(np.sqrt(image))), cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_ylabel('Cacheset')
        ax[1].imshow(invert(image_alt), cmap=cm.gray)
        ax[1].set_title('Manipulated image')
        ax[1].set_ylabel('Cacheset')
        ax[2].imshow(image_alt * 0, cmap=cm.gray_r)
        for line in houghlines:
            p0, p1 = (line.x1, line.y1), (line.x2, line.y2)
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[2].set_xlim((0, image.shape[1]))
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_title('Probabilistic Hough')
        fig.text(0.55, 0.04, '1K Instructions', ha='center', va='center')
        for a in ax:
            a.set_adjustable('box-forced')
        plt.tight_layout()
        if save == True:
            plt.savefig('figs/hough/'+fname+'.png')
        else:
            plt.show()
