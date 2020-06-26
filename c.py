import sys

from tracewringing.clustering import *
from tracewringing.heatmap_generator import *
from sklearn.manifold import TSNE
import numpy as np
from tracewringing.wring_opt import *
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
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import KNeighborsClassifier as KNN
import hdbscan

#from scipy.ndimage.interpolation import shift
from bars import make_bars
#dist = DistanceMetric.get_metric('minkowski',p=2)
os.chdir('/home/mkondapaneni/Research/tracewringing_phase_predict/')
# NAME = sys.argv[1]
NAME = 'gcc-1B'
# NAME = 'gzip'
TYPE = 'mem'
ID = '1'

# WINDOW_SIZE = 10000
WINDOW_SIZE = 100
HEIGHT = 2048
COLLAPSE_FACTOR = 1
def custom_score(heatmap1,heatmap2):
    assert heatmap1.shape == heatmap2.shape
    # heatmap1-=np.mean(heatmap1)
    # heatmap2-= np.mean(heatmap2)
    base = np.zeros(heatmap1.shape)
    total_both_zeros = np.sum(np.logical_and(heatmap1==base,heatmap2==base))
    total_points = heatmap1.shape[0]
    if(len(heatmap1.shape) == 2):
        total_points = heatmap1.shape[0]*heatmap1.shape[1]
    score = mse(heatmap1,heatmap2)*total_points
    #return np.sqrt(np.sqrt((np.linalg.norm(heatmap1.flatten()-heatmap2.flatten(),4)/total_points)))
    if total_both_zeros == total_points:
        return 0.0
    return score/(total_points-total_both_zeros)

def gen_cluster(names,clusters,show=False,dist='euclidean',pca=False,verbose=False,method='kmeans'):
    heatmap_fig = []
    sizes = []
    if type(names) is not list and type(names) is not tuple:
        names = [names]
    end = ""
    if pca:
        end = "-pca"
    #cl_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/clusters/{}/{}'.format(method,"-".join(names)+"-"+str(clusters)+"-"+dist+end)
    cl_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/clusters/{}/{}'.format(method,"polytest2-"+str(clusters))
    for name in names:
        wl_path =  '/home/mkondapaneni/Research/tracewringing_phase_predict/traces/{}.trace'.format(name)
        hm_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/heatmaps2/{}'.format(name+"-"+str(WINDOW_SIZE))
        if os.path.exists(hm_path):
            with open(hm_path,'rb') as f:
                hm_matrix = pickle.load(f)
            if verbose:
                print("WARNING using heatmap at "+hm_path)
        else:
            hm_matrix = np.sqrt(np.sqrt(generate_heatmap(wl_path, HEIGHT, WINDOW_SIZE, hm_path)))
        hm_matrix = np.sqrt(np.sqrt(hm_matrix))    
        if len(heatmap_fig) == 0:
            heatmap_fig = hm_matrix
            #print(hm_matrix.shape)
            #heatmap_fig = np.sqrt( np.sqrt( hm_matrix ))
            #heatmap_shifted =  np.array([np.roll(i,-np.argmax(i)) for i in hm_matrix.T]).T
            # print(heatmap_shifted.shape)
            # print(heatmap_shifted)
            # print(np.max(hm_matrix,axis=0))
            
            shift = np.argmax(hm_matrix.T[0])

            # new = np.array([np.roll(i,-np.argmax(i)) for i in hm.T]).T
            heatmap_shifted = np.roll(hm_matrix.T,-shift).T
            #heatmap_fig = hm_matrix
      
        else:
            heatmap_fig = np.append(heatmap_fig,hm_matrix,axis=1)
            #heatmap_fig = np.append(heatmap_fig,(np.sqrt( np.sqrt( hm_matrix ))),axis=1)
            #heatmap_shifted =  np.append(heatmap_shifted,np.array([np.roll(i,-np.argmax(i)) for i in hm_matrix.T]).T,axis=1)
            #heatmap_fig = np.append(heatmap_fig,hm_matrix,axis=1)
            shift = np.argmax(hm_matrix.T[0])

            # new = np.array([np.roll(i,-np.argmax(i)) for i in hm.T]).T
            heatmap_shifted = np.append(heatmap_shifted,np.roll(hm_matrix.T,-shift).T,axis=1)
        # print(hm_matrix.shape,heatmap_fig.shape)
        sizes.append(len(hm_matrix.T))
    # heatmap_fig-=np.mean(heatmap_fig)
    #print(cl_path)
    return heatmap_fig,heatmap_shifted,sizes

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

def make_heatmap(phases,labels):
    counts = np.zeros(120).astype('int')
    b = []
    # for c,i in enumerate(labels):
    #     if phases[i].T.shape[0]<=counts[i]:
    #         counts[i] = 0
    #     print(counts[i])
    #     b.append(phases[i].T[counts[i]])
    #     counts[i]+=1
    curr_count = 0
    prev = None
    for c,i in enumerate(labels):
        if prev == None or prev != i:
            curr_count = 0
        #print(i,phase1[i].T.shape,curr_count)
        if phases[i].T.shape[0] <= curr_count:
            #print(i,c,curr_count,phases[i].T.shape)
            curr_count=0
        # print(i)
        # print(phases[i].shape)
        b.append(phases[i].T[curr_count])
        curr_count+=1
        prev=i
    #print(counts)
    return np.array(b)

def visualize(names,name2,heatmap1,centroid1,heatmap2,centroid2,label1,label2,save=False,metric="euclidean"):
    # tot=np.append(np.append(heatmap1,centroid1,axis=0),np.append(heatmap2,centroid2,axis=0),axis=0)
    # sizes = [len(heatmap1),len(centroid1),len(heatmap2),len(centroid2)]
    # colors = ['red','lime','yellow','blue']
    # plot_labels = ["standard heatmap","standard centroids",name2+" heatmap",name2+" centroids"]
    # print(tot.shape)
    # print(sizes)
    # t_sne(tot,sizes,colors,names=plot_labels,dist=metric)
    # figure = plt.gcf()
    # figure.set_size_inches(32,18)
    # if save:
    #     if not os.path.exists("figs/compare/"+metric+"/"+name2):
    #         os.makedirs("figs/compare/"+metric+"/"+name2)
    #     plt.savefig("figs/compare/"+metric+"/"+name2+"/clusterspace.png")
    # else:
    #     plt.show()
    hg = HeatmapGenerator()
    cl = Clustering()
    print(heatmap1.shape)
    phase1 = cl.getRepresentativePhases(label1,COLLAPSE_FACTOR,heatmap1)
    print(label1)
    #print(phase1)
    #phase2 = cl.getRepresentativePhases(label2,COLLAPSE_FACTOR,heatmap2.T)
    #print(centroid1.shape,type(centroid1))
    
    #label2 = [i if i != 90 else 94 for i in label2]
    print(list(label2))
    b = make_heatmap(phase1,label2)
    # print(np.array(b).shape)
    print(b.shape)
    # print(heatmap2)
    # print(a.shape,heatmap2.shape,type(a))
    # hg.getHeatmapFigure(heatmap2.T,"echo")
    # hg.getHeatmapFigure(b.T,"echo")
    # print(heatmap2.shape)
    path='figs/compare/euclidean/'+name2
    print(np.mean(heatmap2),np.mean(b),np.std(heatmap2),np.std(b))
    print(np.max(heatmap2),np.min(heatmap2),np.max(b),np.min(b))
    # heatmap2-=np.mean(heatmap2)
    # b-=np.mean(b)
    print(np.std(b),np.std(heatmap2))
    prev= None

    #hg.getHeatmapFigure((heatmap1.T),"axi")
    # cl.getClusterFigure(label2,b.T,COLLAPSE_FACTOR,HEIGHT,"test")
    # cl.getClusterFigure(label2,heatmap2,COLLAPSE_FACTOR,HEIGHT,"test")
    hg.compareHeatmaps((np.sqrt(heatmap2),np.sqrt(b.T)),name2,False,titles=('orig','frankenstein'),path=path)
    hg.getHeatmapFigure(phase1[22],"f")
    for name in names:
        hm_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/heatmaps/{}'.format(name)
        if os.path.exists(hm_path):
            with open(hm_path,'rb') as f:
                hm_matrix = pickle.load(f)
        hm = np.sqrt(np.sqrt(hm_matrix))
        shift = np.argmax(hm.T[0])
        
        #new = np.roll(hm.T,-shift).T
        new = np.array([np.roll(i,-np.argmax(i)) for i in hm.T]).T
        arr = [np.argmax(i) for i in hm.T]
        if prev:
            print([i-prev[3][c] for c,i in enumerate(arr)])
            print(hm.shape,shift,hm.T[0].shape, "prev: ",prev[0].shape,prev[2],prev[0].T[0].shape)
            hg.compareHeatmaps((hm,prev[0],new,prev[1]),name2,False,titles=(name,name2,"o","sdf"),path=path,four=True)
            
        prev = (hm,new,shift,arr)
    # hg.getHeatmapFigure(heatmap2.T,"findmnt")
    # hg.getHeatmapFigure(b.T,"findmnt")

def shift_and_score(a,b):
    darkest_b = np.argmax(a)
    darkest_a = np.argmax(b)
    
    shifted = np.roll(a,darkest_b-darkest_a)
    # print(len(shifted))
    # print(len(b))
    return np.linalg.norm(shifted-b)

def gen(names,name2,k,metric='euclidean',save=False,viz=False,verbose=True,log_file=None):
    heatmap1,heatmap_shifted1,size_names = gen_cluster(names,k,False,dist=metric,verbose=verbose,method='kmeans')
    f = None
    if log_file != None:
        f = open("./logs/"+log_file+".log",'w')
    # print(heatmap1.shape)
    # print(size_names)
    # print(sum(size_names),len(heatmap1))
    k2 = 8
    heatmap2,heatmap_shifted2,_= gen_cluster(name2,k2,False,dist=metric,method='kmeans')
    cl = Clustering()
    # phase1 = cl.getRepresentativePhases(label1,COLLAPSE_FACTOR,heatmap1.T)
    #map_centroids = cluster1.predict(centroid2)
    print("Clustering...")
    #print(cl_path)
    print(heatmap_shifted1.shape)
    cl_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/clusters2/{}'.format("polytest2-"+name2+"-"+str(k)+"-"+str(WINDOW_SIZE))
    if os.path.exists(cl_path):
        with open(cl_path,'rb') as f:
            clusters = pickle.load(f)
        if verbose:
            print("WARNING using cluster at " +cl_path)
    else:         
        cl = Clustering()
        #print(heatmap.shape)
        #print(heatmap_fig.shape,heatmap_shifted.shape)
        # clusters = cl.kmeans(k,COLLAPSE_FACTOR,heatmap1)
        c = hdbscan.HDBSCAN(prediction_data=True)
        c.fit(heatmap1.T)
        cent = None
        clusters = (c.labels_,cent,c)

        with open(cl_path, 'wb') as f:
            pickle.dump(clusters,f)
    labels,centroids,cluster = clusters

    map_new_standard = cluster.approximate_predict(heatmap2.T)

    visualize(names,name2,heatmap1,None,heatmap2,None,labels,map_new_standard,False,metric)
    plt.figure()
    # print(np.array_equal(heatmap_fig,heatmap.T))
    # hg  = HeatmapGenerator()
    # hg.compareHeatmaps(heatmap_fig,heatmap.T,"-".join(names))
  

# names = ["cat","cp","echo","findmnt","git","ls"]
# name2 = "mkdir"
name2 = "echo"
names = ['cat','cp','findmnt','git','ls','mkdir']
# names = os.listdir('/home/mkondapaneni/Research/trace_src/bin')
# name2 = names[0]
# names = names[1]
print(name2,names)


k=60
metric = 'euclidean'

if __name__ == "__main__":
   gen(names,name2,k,metric,save=False,viz=True)
