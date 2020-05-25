import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

class Clustering(object):

    def kmeans(self, k, collapse_factor, heatmap_matrix):
        """ Reduces heatmap_matrix dimensions by collapse_factor,
            calls K-Means clustering to cluster collapsed heatmap_matrix,
            returns a sequence of cluster labels.

            Args:
                k: number of clusters
                collapse_factor: number of matrix columns collapsed
                heatmap_matrix: from HeatmapGenerator.getHeatmapMatrix

            Returns:
                labels: a string, each char is a cluster label.
                        Each char represents collapse_factor number of columns in heatmap_matrix.
                centers: k centroids.
        """
        assert collapse_factor is not None, 'Collapse_factor' 
        # def sum_chunk(x, collapse_factor, axis=-1):
        #     extra = x.shape[-1] % collapse_factor
        #     x = x[:,:-extra]
        #     shape = x.shape
        #     if axis < 0:
        #         axis += x.ndim
        #     shape = shape[:axis] + (0, collapse_factor) + shape[axis+1:]
        #     x = x.reshape(shape)
        #     return x.sum(axis=axis+1)
        # reduced_heatmap_matrix = sum_chunk(heatmap_matrix,collapse_factor)
        # print(reduced_heatmap_matrix.shape)
        # cluster = KMeans(n_clusters=k).fit(reduced_heatmap_matrix.T)
        cluster = KMeans(n_clusters=k).fit(heatmap_matrix.T)
        return cluster.labels_, cluster.cluster_centers_

    def getClusterFigure(self, labels, heatmap_matrix, collapse_factor, trace_height, name, save=False, fname=None):
        """ Plots Figure 4(a) (clusters) in paper.

            Args:
                labels: from kmeans

            Returns:
                cluster_fig: Figure 4(a)
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        pre = np.repeat(labels,1)
        f, ax = plt.subplots(2, figsize=(10,2), sharex=True,frameon=True)
        ax[0].imshow(heatmap_matrix,cmap=cm.gray_r)
        ax[0].axis('off')
        ax[1].imshow(np.tile(pre,(int(trace_height),1)),cmap=cm.gist_rainbow)
        ax[0].set_title(name)
        ax[1].axis('off')
        plt.tight_layout()
        if save == True:
            if not os.path.exists('figs/clustering'):
                os.mkdir('figs/clustering')
            plt.savefig('figs/clustering/'+fname+'.png')
        else:
            plt.show()

    def getRepresentativePhases(self, labels, collapse_factor, heatmap_matrix):
        """ Computes distance between each pair of collapsed column and cluster center,
            selectes a representative phase (closest to center) for each cluster.

            Args:
                labels: from kmeans
                centers: from kmeans
                heatmap_matrix: from HeatmapGenerator.getHeatmapMatrix

            Returns:
                phases: a list, each element is the most representative phase for each cluster.
        """
        all_clusters = np.array([np.where(labels == i)[0] for i in np.unique(labels)])
        step = 1
        rep_phases = [] # the longest phase for now
        for i in all_clusters:
            cons = np.split(i, np.where(np.diff(i) != step)[0]+1) # consecutive labels
            rep_phases.append(cons[np.argmax([len(j) for j in cons])])
            # rep_phases.append(cons [np.argsort(j)[len(j)//2] for j in cons] )
        # these rep_phases give us the ranges for where the phases exist in the clustered image
        ranges = [(min(k)*collapse_factor,(max(k)+1)*collapse_factor) for k in rep_phases]
        phases = [heatmap_matrix[:,l:m] for l,m in ranges]
        return phases
