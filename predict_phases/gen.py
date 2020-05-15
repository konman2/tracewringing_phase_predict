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
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.metrics import silhouette_samples,silhouette_score
import matplotlib.cm as cm
from sklearn.neighbors import DistanceMetric
import pickle
#dist = DistanceMetric.get_metric('minkowski',p=2)

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
def gen_cluster(names,clusters,show=False):
    heatmap_fig = []
    sizes = []
    if type(names) is not list and type(names) is not tuple:
        names = [names]
    for name in names:
        wl_path =  '/home/mkondapaneni/Research/tracewringing/{}.trace'.format(name)
        hm_path = '/home/mkondapaneni/Research/tracewringing/predict_phases/heatmaps/{}'.format(name)
        if os.path.exists(hm_path):
            with open(hm_path,'rb') as f:
                hm_matrix = pickle.load(f)
            print(hm_path)
        else:
            hm_matrix = generate_heatmap(wl_path, HEIGHT, WINDOW_SIZE, hm_path)
        if len(heatmap_fig) == 0:
            heatmap_fig = np.sqrt( np.sqrt( hm_matrix ))
        else:
           heatmap_fig = np.append(heatmap_fig,(np.sqrt( np.sqrt( hm_matrix ))),axis=1)
        sizes.append(len(hm_matrix.T))

    cl = Clustering()
    labels, centroids,cluster,heatmap = cl.kmeans(clusters,COLLAPSE_FACTOR,heatmap_fig)

    if show:
        cl.getClusterFigure(labels,heatmap_fig,COLLAPSE_FACTOR,HEIGHT,"-".join(names))
    return labels,centroids,cluster,heatmap,sizes



name1 = "gcc-1B"
name2 = "gzip"
name3 = "gcc-734B"


def t_sne(tot,nums,colors=['b','r']):
    _=tot
    pca = PCA(n_components=50).fit_transform(_)
    X_tsne = TSNE(n_components=2,verbose=1,n_iter=1000).fit_transform(pca)
    curr_ind = 0
    for c,i in enumerate(nums):
        plt.scatter(X_tsne[curr_ind:curr_ind+i].T[:1].T,X_tsne[curr_ind:curr_ind+i].T[1:].T,color=colors[c])
        curr_ind += i
        #plt.scatter(X_tsne[-k:].T[:1].T,X_tsne[-k:].T[1:].T,color=colors[1])
def to_file(label,names,sizes=None):
    if type(names) is not list and type(names) is not tuple:
        names = [names]
        sizes = [len(label)]
    else:
        assert len(names) == len(sizes)
    curr_ind = 0
    for c,name in enumerate(names):
        f = open("phases/"+name+".phase",'w')
        for l in label[curr_ind:curr_ind+sizes[c]]:
            f.write(str(l)+"\n")
        f.close()

names = ["cat","cp","echo","findmnt","git","ls"]
name2 = "mkdir"
k=20
if __name__ == "__main__":
   
    label1,centroid1,cluster1,heatmap1,size_names = gen_cluster(names,k,False)
    print(size_names)
    print(sum(size_names),len(heatmap1))
    #exit(1)
    k2 = 6
    label2,centroid2,cluster2,heatmap2,_ = gen_cluster(name2,k2,False)
    plt.figure()
    print(heatmap1.shape,centroid1.shape,heatmap2.shape,centroid2.shape)
    tot=np.append(np.append(heatmap1,centroid1,axis=0),np.append(heatmap2,centroid2,axis=0),axis=0)
    sizes = [len(heatmap1),len(centroid1),len(heatmap2),len(centroid2)]
    colors = ['r','b','yellow','green']
    print(sizes)
    # t_sne(tot,sizes,colors)
    # plt.show()
    to_file(label1,names,size_names)
    
    map_centroids = cluster1.predict(centroid2)
    print(map_centroids)
    map_standard_label = {}
    for c,l in enumerate(map_centroids):
        if l in map_standard_label:
            map_standard_label[l].append(c)
        else:
            map_standard_label[l] = [c]
    actual_k = len(set(map_centroids))
    count = np.zeros(k2)
    for c,l in enumerate(label2):
        count[l] += 1
    
    print(map_standard_label)
    print(count)
    for item in map_standard_label.items():
        m = -1
        for ind in item[1]:
            if m == -1 or count[ind] > count[m]:
                m = ind
        map_standard_label[item[0]] = m
    print(map_standard_label)


    mapped_labels =[map_centroids[l] for l in label2]
    
    print(mapped_labels)
    to_file(mapped_labels,name2)
    print(centroid2.shape,centroid1.shape)
    f = open("standard_to_label.pkl",'wb')
    pickle.dump(map_standard_label,f)
    f.close()
    
