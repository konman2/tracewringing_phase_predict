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
        plt.figure(figsize=(10,10),dpi=120)
        plt.ylabel('Projected Address')
        plt.xlabel(xaxis+' Instructions')
        plt.imshow(heatmap_matrix,cmap=cm.gray_r) #plt.imshow(k,cmap=cm.Greys, interpolation='nearest')
        if save == True:
            if not os.path.exists('./figs/heatmaps'):
                os.makedirs('./figs/heatmaps')
            plt.savefig('./figs/heatmaps/'+name+'.png') 
            print('Heatmap figure saved..')
        else:
            plt.show()
        plt.close()

    
    def compareHeatmaps(self, heatmap_orig, heatmap_new, name,save=False,name2=None,titles=("source","proxy"),metric=""):
        """ Figure of two heatmaps, for side by side comparison. 
        
        Arguments:
            heatmap_orig {[np.ndarray]} -- original trace heatmap 
            heatmap_new {[np.ndarray]} -- proxy trace heatmap 
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        if name2 == None:
            name2 = name
        #(15,18)
        f, ax = plt.subplots(1,2, figsize=(15,18), sharex=True,frameon=True)
        ax[0].imshow(np.sqrt(np.sqrt(heatmap_orig)),cmap=cm.gray_r)
        ax[0].set_title('Heatmap of {} trace: {}'.format(titles[0],name))
        ax[0].axis('off')
        ax[1].imshow(np.sqrt(np.sqrt(heatmap_new)),cmap=cm.gray_r)
        ax[1].set_title('Heatmap of {} trace: {}'.format(titles[1],name2))
        ax[1].axis('off')
        plt.tight_layout()
        if save == True:
            if not os.path.exists('./figs/compare/'+metric):
                os.makedirs('./figs/compare/'+metric)
            plt.savefig('./figs/compare/'+metric+"/"+name+'.png') 
            print('Comparison of heatmaps: Figure saved..')
        else:
            plt.show()
        plt.close()