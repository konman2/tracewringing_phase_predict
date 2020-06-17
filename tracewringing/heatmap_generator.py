import math
import numpy as np
import os

class Heatmap(object):
    def __init__(self, matrix):
        # Each column must have a non-zero element.
        # Sums over rows (collaps rows) after testing for 0.
        assert not 0 in (matrix != 0).sum(0)
        self.matrix = matrix

    def __sub__(self, other):
        assert self.matrix.shape == other.matrix.shape
        raise NotImplementedError

    def dump(self):
        print('Heatmap of shape {}\n'.format(self.matrix.shape))
        print(self.matrix)

class HeatmapGenerator(object):

    def getHeatmapMatrix(self, trace, hm_height, window_size):
        """ Takes an address trace,
            maps each address to a "virtual" cache set,
            the total # of virtual cache set = n_cacheset * multiplier,
            accumulates all accesses per cache set for each window (# of memory accesses)

            Args:
                trace: input address trace, each line is a hex string
                n_cacheset: # of total cache sets
                associativity: n_cacheset = associativity * # cache lines
                window_size: # of memory accesses (measure of time interval)

            Returns:
                heatmap_matrix: a Heatmap object of N * M,
                    where N = len(trace) / window_size, M = n_cacheset
        """
        
        rows = hm_height
        # Keep the residual addresses in an additional window.
        cols = (np.ceil(1. * len(trace) / window_size)).astype(int)
        matrix = np.zeros((rows, cols), dtype=int)
        max_addr = 0
        for idx in range(len(trace)):
            if trace[idx] > max_addr:
                max_addr = trace[idx]
            matrix[trace[idx] % rows][idx // window_size] += 1
        #print('Heatmap generated w/ dimensions: {}'.format(matrix.shape))
        heatmap = Heatmap(matrix)
        return heatmap

    def getHeatmapFigure(self,heatmap_matrix,xaxis,save=False,name=None):
        """ Generates a figure from heatmap_matrix (from getHeatmapMatrix).

        Args:
            heatmap_matrix: from getHeatmapMatrix

        Returns:
            heatmap_fig: a heatmap figure
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        plt.figure(dpi=120)
        plt.ylabel('Projected Address')
        plt.xlabel(xaxis+' Instructions')
        plt.imshow(heatmap_matrix,cmap=cm.gray_r) #plt.imshow(k,cmap=cm.Greys, interpolation='nearest')
        if save == True:
            if not os.path.exists('./figs/heatmaps'):
                os.makedirs('./figs/heatmaps')
            plt.savefig('./figs/heatmaps/'+name+'.png') 
            print('Heatmap figure saved..')
        else:
            plt.tight_layout()
            plt.show()
        plt.close()

    
    def compareHeatmaps(self, heatmaps, name,save=False,titles=("source","proxy"),path=None,four=False):
        """ Figure of two heatmaps, for side by side comparison. 
        
        Arguments:
            heatmap_orig {[np.ndarray]} -- original trace heatmap 
            heatmap_new {[np.ndarray]} -- proxy trace heatmap 
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        #(15,18)
        if four:
            f,ax = plt.subplots(2,2,figsize=(15,9.9),sharex=True,frameon=True)
            ax = ax.flatten()
        elif heatmaps[0].shape[0] <= heatmaps[0].shape[1]:
            f, ax = plt.subplots(len(heatmaps),1, figsize=(15,9.9), sharex=True,frameon=True)
        else:
            f, ax = plt.subplots(1,len(heatmaps), figsize=(15,9.9), sharex=True,frameon=True)
        for c,i in enumerate(heatmaps):
            # ax[c].imshow(np.sqrt(np.sqrt(i)),cmap=cm.gray_r)
            ax[c].imshow(i,cmap=cm.gray_r)
            ax[c].set_title('Heatmap of {} trace: {}'.format(titles[c],name))
            #ax[c].axis('off')
        plt.tight_layout()
        if save == True and path != None:
            if not os.path.exists(path):
                os.makedirs(path)
            print(path)
            plt.savefig(path+'/'+name+'_comparison.png') 
            print('Comparison of heatmaps: Figure saved at ' + path)
        else:
            print("No Path")
            plt.show()
        plt.close()