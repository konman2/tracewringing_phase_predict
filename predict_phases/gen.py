import sys
sys.path.append("..")
from clustering import *
from heatmap_generator import *
from sklearn.manifold import TSNE
import numpy as np
from wring_opt import *
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
# NAME = sys.argv[1]
NAME = 'gcc-1B'
# NAME = 'gzip'
TYPE = 'mem'
ID = '1'

# WINDOW_SIZE = 10000
WINDOW_SIZE = 100
HEIGHT = 2048
COLLAPSE_FACTOR = 1
clusters = 5
def gen_cluster(name,clusters):
    wl_path =  '/home/mkondapaneni/Research/tracewringing/{}.trace'.format(name)
    hm_path = '/home/mkondapaneni/Research/tracewringing/predict_phases/heatmaps/{}'.format(name)
    if os.path.exists(hm_path):
        with open(hm_path,'rb') as f:
            hm_matrix = pickle.load(f)
        print(hm_path)

    else:
        hm_matrix = generate_heatmap(wl_path, HEIGHT, WINDOW_SIZE, hm_path)

    heatmap_fig = np.sqrt( np.sqrt( hm_matrix ))

    cl = Clustering()
    labels, centroids,cluster = cl.kmeans(clusters,COLLAPSE_FACTOR,heatmap_fig)


    return labels,centroids,cluster


label1,centroid1,cluster1, = gen_cluster("gcc-1B",5)
label2,centroid2,cluster2 = gen_cluster("gzip",5)
print(label1)
print(label2)
l = np.zeros(5)
for i in range(len(centroid2)):
    m = -1
    m_i = 0
    for i in range(len(centroid1)):
        print(distance.euclidean(centroid1[i],centroid2[i]), euclidean_distances(centroid1[i].reshape(1,-1),centroid2[i].reshape(1,-1)))
        if m == -1 or distance.euclidean(centroid1[i],centroid2[i]) < m:
            m_i = i
            m = distance.euclidean(centroid1[i],centroid2[i])
    l[i] = m_i
print(l)

print(cluster1.predict(centroid2))
print(centroid2.shape,centroid1.shape)

dist = [0 for i in range(5)]
for i in label2:
   dist[i]+=1

print(dist)

    


fig,ax= plt.subplots()
overall_centroids = np.append(centroid1,centroid2,axis=0)
print(overall_centroids.shape)

X_tsne = TSNE(n_components=2,n_iter=1000,random_state=2).fit_transform(overall_centroids)
# X_tsne = tsne.fit_transform(centroid1)
# X_tsne2 = tsne.tra
#graph(X_tsne,("t1","t2"),fig,ax)
# print("max " +param+ "=",np.max(y))
print(X_tsne.shape)
colors = ["blue","red","green","magenta","orange","brown","brown","brown","brown","brown"]
for i,X in enumerate(X_tsne):
    print(i)
    ax.scatter(X.T[:1].T,X.T[1:].T,c=colors[i])
# plot = ax.scatter(X_tsne[:5].T[:1].T,X_tsne[:5].T[1:].T,c="blue")
# plot = ax.scatter(X_tsne[5:].T[:1].T,X_tsne[5:].T[1:].T,c='red')
# cbar = fig.colorbar(plot)
# cbar.set_label('max_scaled '+ param)
ax.set_xlabel("t1")
ax.set_ylabel("t2")
# graph(X_tsne2,("t1","t2"),fig,ax)
plt.show()
