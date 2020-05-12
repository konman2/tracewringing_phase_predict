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
        
        def collapseHeatmap(heatmap_matrix, collapse_factor):
            M = heatmap_matrix.shape[0]
            N = heatmap_matrix.shape[1]
            R = N%collapse_factor
            if R == 0:
                reduced_heatmap = np.zeros((M, int(N/collapse_factor)))
            else:
                reduced_heatmap = np.zeros((M, N//collapse_factor + 1))

            for i in range(N//collapse_factor):
                cols = heatmap_matrix[:,i*collapse_factor:i*collapse_factor+collapse_factor]
                col_sum = np.sum(cols,axis=1)
                reduced_heatmap[:,i] = col_sum

            if R != 0:
                cols = heatmap_matrix[:,N-R:N]
                col_sum = np.sum(cols,axis=1)
                reduced_heatmap[:,-1] = col_sum

            return reduced_heatmap

        reduced_heatmap = collapseHeatmap(heatmap_matrix,collapse_factor)
        cluster = KMeans(n_clusters=k,random_state=0).fit(reduced_heatmap.T)
        #print(cluster.labels_)
        return cluster.labels_, cluster.cluster_centers_,cluster

    def getClusterFigure(self, labels, heatmap_matrix, collapse_factor, trace_height, name, save=False, fname=None):
        """ Plots Figure 4(a) (clusters) in paper.

            Args:
                labels: from kmeans

            Returns:
                cluster_fig: Figure 4(a)
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        f, ax = plt.subplots(2, figsize=(10,20), sharex=True,frameon=True)
        ax[0].imshow(heatmap_matrix,cmap=cm.gray_r)
        ax[0].axis('off')
        ax[1].imshow(np.tile(np.repeat(labels,collapse_factor),(int(trace_height/10),1)),cmap=cm.gist_rainbow)
        ax[0].set_title(name)
        ax[1].axis('off')
        plt.tight_layout()
        if save == True:
            if not os.path.exists('figs/clustering'):
                os.makedirs('figs/clustering')
            plt.savefig('figs/clustering/'+fname+'.png')
        else:
            plt.show()
        plt.close()

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