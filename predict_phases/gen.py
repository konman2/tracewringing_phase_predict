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
from skimage.transform import resize
from sklearn.preprocessing import Normalizer
import matplotlib.patches as mpatches
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

def gen_cluster(names,clusters,show=False,dist='euclidean',pca=False):
    heatmap_fig = []
    sizes = []
    if type(names) is not list and type(names) is not tuple:
        names = [names]
    end = ""
    if pca:
        end = "-pca"
    cl_path = '/home/mkondapaneni/Research/tracewringing/predict_phases/clusters/{}'.format("-".join(names)+"-"+str(clusters)+"-"+dist+end)
    for name in names:
        wl_path =  '/home/mkondapaneni/Research/tracewringing/{}.trace'.format(name)
        hm_path = '/home/mkondapaneni/Research/tracewringing/predict_phases/heatmaps/{}'.format(name)
        if os.path.exists(hm_path):
            with open(hm_path,'rb') as f:
                hm_matrix = pickle.load(f)
            print("WARNING using heatmap at "+hm_path)
        else:
            hm_matrix = generate_heatmap(wl_path, HEIGHT, WINDOW_SIZE, hm_path)
        if len(heatmap_fig) == 0:
            heatmap_fig = np.sqrt( np.sqrt( hm_matrix ))
      
        else:
           heatmap_fig = np.append(heatmap_fig,(np.sqrt( np.sqrt( hm_matrix ))),axis=1)
        # print(hm_matrix.shape,heatmap_fig.shape)
        sizes.append(len(hm_matrix.T))

    if os.path.exists(cl_path):
        with open(cl_path,'rb') as f:
            kmeans = pickle.load(f)
        print("WARNING using cluster at " +cl_path)
    else:
        cl = Clustering()
        heatmap = heatmap_fig.T
        if dist == "cosine":
            normalizer = Normalizer()
            heatmap = normalizer.fit_transform(heatmap)
        if pca:
            heatmap = PCA(n_components=500).fit_transform(heatmap)

        kmeans = cl.kmeans(clusters,COLLAPSE_FACTOR,heatmap.T)
        with open(cl_path, 'wb') as f:
            pickle.dump(kmeans,f)
    labels,centroids,cluster,heatmap = kmeans
    # print(np.array_equal(heatmap_fig,heatmap.T))
    # hg  = HeatmapGenerator()
    # hg.compareHeatmaps(heatmap_fig,heatmap.T,"-".join(names))
    if show:
        cl.getClusterFigure(labels,heatmap_fig,COLLAPSE_FACTOR,HEIGHT,"-".join(names))
    return labels,centroids,cluster,heatmap_fig.T,sizes

def t_sne(tot,nums,colors,names,dist='euclidean'):
    _=tot
    pca = PCA(n_components=50).fit_transform(_)
    X_tsne = TSNE(n_components=2,verbose=1,n_iter=1000,random_state=0,metric=dist).fit_transform(pca)
    curr_ind = 0
    handles = [mpatches.Circle((0.5,0.5),fc=colors[i],label=names[i]) for i in range(len(names))]
    for c,i in enumerate(nums):
        plt.scatter(X_tsne[curr_ind:curr_ind+i].T[:1].T,X_tsne[curr_ind:curr_ind+i].T[1:].T,color=colors[c])
        curr_ind += i
    plt.legend(handles=handles)
    plt.title("Tsne of standard set of programs overlayed with new program")
    plt.xlabel("t-sne1")
    plt.ylabel("t-sne2")
    #plt.tight_layout()
        #plt.scatter(X_tsne[-k:].T[:1].T,X_tsne[-k:].T[1:].T,color=colors[1])

def to_file(label,names,sizes=None):
    # print(label,names,sizes)
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

def calc_mapping(cluster1,centroid2,k2,label2,save=True):
    map_centroids = cluster1.predict(centroid2)
    print(map_centroids)
    map_standard_label = {}
    for c,l in enumerate(map_centroids):
        if l in map_standard_label:
            map_standard_label[l].append(c)
        else:
            map_standard_label[l] = [c]
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
    if save == True:
        print(mapped_labels)
        to_file(mapped_labels,name2)
        #print(centroid2.shape,centroid1.shape)
        f = open("standard_to_label.pkl",'wb')
        pickle.dump(map_standard_label,f)
        f.close()
        #print(label1)

    return map_centroids,map_standard_label,mapped_labels

def visualize(names,name2,heatmap1,centroid1,heatmap2,centroid2,label1,label2,map_centroids,save=False,metric="euclidean"):
    tot=np.append(np.append(heatmap1,centroid1,axis=0),np.append(heatmap2,centroid2,axis=0),axis=0)
    sizes = [len(heatmap1),len(centroid1),len(heatmap2),len(centroid2)]
    colors = ['red','lime','yellow','blue']
    plot_labels = ["standard heatmap","standard centroids",name2+" heatmap",name2+" centroids"]
    print(tot.shape)
    print(sizes)
    t_sne(tot,sizes,colors,names=plot_labels,dist=metric)
    figure = plt.gcf()
    figure.set_size_inches(32,18)
    if save:
        if not os.path.exists("figs/compare/"+metric+"/"+name2):
            os.makedirs("figs/compare/"+metric+"/"+name2)
        plt.savefig("figs/compare/"+metric+"/"+name2+"/clusterspace.png")
    else:
        plt.show()
    hg = HeatmapGenerator()
    cl = Clustering()
    phase1 = cl.getRepresentativePhases(label1,COLLAPSE_FACTOR,heatmap1.T)
    phase2 = cl.getRepresentativePhases(label2,COLLAPSE_FACTOR,heatmap2.T)
    print(len(phase1),len(phase2),len(set(label2)))
    for i in set(label2):
        print(phase1[map_centroids[i]].shape,phase2[i].shape)
        a = phase1[map_centroids[i]]#.reshape(-1,1)
        b = phase2[i]#.reshape(-1,1)
        reshape_size = (HEIGHT,200)
        a = resize(a,reshape_size)
        b = resize(b,reshape_size)
        hg.compareHeatmaps(a**4,b**4,"cluster-"+str(map_centroids[i]),save,"cluster-"+str(i),titles=("standard",name2),metric=metric)


def run(names,name2,k,metric='euclidean',save=False,viz=False):
    label1,centroid1,cluster1,heatmap1,size_names = gen_cluster(names,k,False,dist=metric)
    print(heatmap1.shape)
    print(size_names)
    print(sum(size_names),len(heatmap1))
    #exit(1)
    k2 = 8
    label2,centroid2,cluster2,heatmap2,_ = gen_cluster(name2,k2,False,dist=metric)
    print(heatmap1.shape,centroid1.shape,heatmap2.shape,centroid2.shape)
    #####REMOVE#####
    # heatmap1 = heatmap1.T
    # heatmap2 = heatmap2.T
    ################

    # tot=np.append(np.append(heatmap1,centroid1,axis=0),np.append(heatmap2,centroid2,axis=0),axis=0)
    # sizes = [len(heatmap1),len(centroid1),len(heatmap2),len(centroid2)]
    # colors = ['red','lime','yellow','blue']
    # plot_labels = ["standard heatmap","standard centroids",name2+" heatmap",name2+" centroids"]
    # print(tot.shape)
    # print(sizes)
    # t_sne(tot,sizes,colors,names=plot_labels,dist='euclidean')
       
    # figure = plt.gcf()
    # figure.set_size_inches(32,18)
    # if save:
    #     if not os.path.exists("figs/compare/"+metric+"/"+name2):
    #         os.makedirs("figs/compare/"+metric+"/"+name2)
    #     plt.savefig("figs/compare/"+metric+"/"+name2+"/clusterspace.png")
    # else:
    #     plt.show()
    # plt.figure()

    if save:
        to_file(label1,names,size_names)
    # hg = HeatmapGenerator()
    # cl = Clustering()
    map_centroids = calc_mapping(cluster1,centroid2,k2,label2,save=save)[0]
    if viz:
        visualize(names,name2,heatmap1,centroid1,heatmap2,centroid2,label1,label2,map_centroids,False,metric)
        plt.figure()
    # phase1 = cl.getRepresentativePhases(label1,COLLAPSE_FACTOR,heatmap1.T)
    # phase2 = cl.getRepresentativePhases(label2,COLLAPSE_FACTOR,heatmap2.T)
    # print(len(phase1),len(phase2),len(set(label2)))
    # for i in set(label2):
    #     print(phase1[map_centroids[i]].shape,phase2[i].shape)
    #     a = phase1[map_centroids[i]]#.reshape(-1,1)
    #     b = phase2[i]#.reshape(-1,1)
    #     reshape_size = (HEIGHT,200)
    #     a = resize(a,reshape_size)
    #     b = resize(b,reshape_size)
    #     hg.compareHeatmaps(a**4,b**4,"cluster-"+str(map_centroids[i]),save,"cluster-"+str(i),titles=("standard",name2),metric=metric)
    

names = ["cat","cp","echo","findmnt","git","ls",]
name2 = "mkdir"
k=120
metric = 'euclidean'

if __name__ == "__main__":
   run(names,name2,k,metric,False,True)
