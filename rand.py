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
from scipy.ndimage.interpolation import shift
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
    cl_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/clusters/{}/{}'.format(method,"-".join(names)+"-"+str(clusters)+"-"+dist+end)
    for name in names:
        wl_path =  '/home/mkondapaneni/Research/tracewringing_phase_predict/traces/{}.trace'.format(name)
        hm_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/heatmaps/{}'.format(name)
        if os.path.exists(hm_path):
            with open(hm_path,'rb') as f:
                hm_matrix = pickle.load(f)
            if verbose:
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
            clusters = pickle.load(f)
        if verbose:
            print("WARNING using cluster at " +cl_path)
    else:
        
        cl = Clustering()
        heatmap = heatmap_fig.T
        if dist == "cosine":
            normalizer = Normalizer()
            heatmap = normalizer.fit_transform(heatmap)
        if pca:
            heatmap = PCA(n_components=500).fit_transform(heatmap)
        if method == 'kmeans':
            print(heatmap.shape)
            clusters = cl.kmeans(clusters,COLLAPSE_FACTOR,heatmap.T)
        if method == 'aff_prop':
            print(heatmap.astype('float32').dtype)
            aff_prop = AffinityPropagation(damping=0.5).fit(heatmap.astype('float32'))
            clusters = (aff_prop.labels_,aff_prop.cluster_centers_,aff_prop)
        with open(cl_path, 'wb') as f:
            pickle.dump(clusters,f)
    labels,centroids,cluster = clusters
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
        b.append(phases[i].T[curr_count])
        curr_count+=1
        prev=i
    #print(counts)
    return np.array(b)

def mse_scaled(a,b,scale=1):
    #return scale*mse(a,b)
    return 0
    #return np.linalg.norm(a.flatten()-b.flatten(),ord=2)
def norm_ssim(a,b,scale=1):
    a1 = a-np.mean(a)
    b1 = b-np.mean(b)
    return one_minus(a1,b1)
def one_minus(a,b):
    r =max(np.max(a),np.max(b))-min(np.min(a),np.min(b))
    #print(max(np.max(a),np.max(b)),min(np.min(a),np.min(b)))
    return (1-ssim(a,b,data_range=r))


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
    phase1 = cl.getRepresentativePhases(label1,COLLAPSE_FACTOR,heatmap1.T)
    phase2 = cl.getRepresentativePhases(label2,COLLAPSE_FACTOR,heatmap2.T)
    print(centroid1.shape,type(centroid1))
    
    #label2 = [i if i != 90 else 94 for i in label2]
    
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
    # hg.compareHeatmaps((heatmap2.T**4,b.T**4),name2,False,titles=('orig','frankenstein'),path=path)
    # hg.getHeatmapFigure(heatmap2.T,"findmnt")
    # hg.getHeatmapFigure(b.T,"findmnt")
    funcs = (norm_ssim,mse,one_minus)
    # ssim(heatmap2,b)
    metric_compare = [[i(heatmap2,b) for i in funcs],[i(heatmap2,np.zeros(heatmap2.shape)) for i in funcs],[ i(heatmap2,np.random.rand(heatmap2.shape[0],heatmap2.shape[1])) for i in funcs], [ i(b,np.random.rand(heatmap2.shape[0],heatmap2.shape[1])) for i in funcs]]
    
    # metric_compare = [[i(heatmap2,b),i(heatmap2,np.zeros(heatmap2.shape))] for c in enumerate(funcs)]
    
    print(metric_compare)
    names = ['1-SSIM (Mean Centered Heatmap)','MSE','1-SSIM',]
    labels = ['Frankenstein v Original','All White v Original','Random v Original','Random v Frankenstein']
    title = 'Metric Comparison for '+name2
    make_bars(metric_compare,labels,names,title=title)
    #plt.savefig(path+'/'+name2+'_metric_comparison.png')
    plt.show()
    start = 670
    end = 695
    print(label2)
    print(label2[start:end])
    print(len(set(label2[start:end])))
    prev = None
    surroundings = set()
    for i in label2[600:700]:
        if prev != None and prev == 94:
            surroundings.add(i)
        if i == 94 and prev != None:
            surroundings.add(prev)
        prev=i
    print(surroundings)
    
    # hg.compareHeatmaps([phase1[i] for i in set(label2[start:end])],name2,False,titles=[str(i) for i in set(label2[start:end])])
    reshape_size = (HEIGHT,200)
    # for i in surroundings:
    #     hg.compareHeatmaps([resize(phase1[94],reshape_size),resize(phase1[i],reshape_size)],name2,False,titles=['94',str(i)])
    # hg.compareHeatmaps([resize(phase1[94],reshape_size),resize(phase1[90],reshape_size)],name2,False,titles=['94','90'])
    # print(len(phase1),len(phase2),len(set(label2)))
    # print(set(label2))
    # for c,i in enumerate(set(label2)):
    #     print(c,i)
    #     print(phase1[i].shape,phase2[c].shape)
    #     a = phase1[i]#.reshape(-1,1)
    #     b = phase2[c]#.reshape(-1,1)
    #     reshape_size = (HEIGHT,200)
    #     a = resize(a,reshape_size)
    #     b = resize(b,reshape_size)
    #     hg.compareHeatmaps(a**4,b**4,"cluster-"+str(i),save,"cluster-"+str(i),titles=("standard",name2),metric=metric)

def p(s="",f=None,end="\n"):
    if f == None:
        print(s)
    if f:
        f.write(s+end)

def shift_and_score(a,b):
    first_activity_b = np.nonzero(b)[0][0]
    first_activity_a = np.nonzero(a)[0][0]
    
    shifted = np.roll(a,first_activity_b-first_activity_a)
    # print(len(shifted))
    # print(len(b))
    return np.linalg.norm(shifted-b)

def gen(names,name2,k,metric='euclidean',save=False,viz=False,verbose=True,log_file=None):
    label1,centroid1,cluster1,heatmap1,size_names = gen_cluster(names,k,False,dist=metric,verbose=verbose,method='kmeans')
    f = None
    if log_file != None:
        f = open("./logs/"+log_file+".log",'w')
    # print(heatmap1.shape)
    # print(size_names)
    # print(sum(size_names),len(heatmap1))
    k2 = 8
    label2,centroid2,cluster2,heatmap2,_ = gen_cluster(name2,k2,False,dist=metric,method='kmeans')
    #print(heatmap1.shape,centroid1.shape,heatmap2.shape,centroid2.shape)
    
    #map_centroids,map_labels = calc_mapping(cluster1,centroid1,heatmap2,save=save)[0]
    #knn = KNN(n_neighbors=11,metric=custom_score).fit(heatmap1,label1)
    cl = Clustering()
    phase1 = cl.getRepresentativePhases(label1,COLLAPSE_FACTOR,heatmap1.T)
    map_centroids = cluster1.predict(centroid2)
    
    #map_new_standard = knn.predict(heatmap2)
    # map_new_standard = []
    # print(heatmap2.shape)
    # print(centroid1.shape)
    # for i in heatmap2:
    #     score = None
    #     score_ind = 0
    #     for c,j in enumerate(centroid1):
    #         # print(i.shape,j.shape)
    #         s = shift_and_score(j,i)
    #         if score == None or s<score:
    #             score = s
    #             score_ind = c
    #     map_new_standard.append(score_ind)
            
    # print(len(map_new_standard))
    map_new_standard = cluster1.predict(heatmap2)
    
    # orig = cluster1.predict(heatmap2)
    # map_new_standard = []
    # for i in heatmap2:
    #     smallest = None
    #     small_ind = 0
    #     for c,phase in enumerate(centroid1):
    #         #score = np.linalg.norm(i-phase)
    #         score = custom_score(i,phase)
    #         if smallest == None or score<smallest:
    #             smallest = score
    #             small_ind = c
    #     map_new_standard.append(small_ind)
    # map_new_standard = []
    # for count,i in enumerate(heatmap2):
    #     smallest = None
    #     small_ind = 0
    #     #print(count)
    #     for c,phase in enumerate(phase1):
    #         # phase = phase.T
    #         # if phase.shape[1]>=30:
    #         #     hg = HeatmapGenerator()
    #         #     hg.compareHeatmaps((phase,resize(i,phase.shape)),name2,False,titles=('orig','frankenstein'))
    #         #     exit()
    #         #print(phase.shape,resize(i,phase.shape).shape)
    #         s = mse(resize(i,phase.shape),phase)
    #         if smallest == None or s <= smallest:
    #             smallest = s
    #             small_ind = c
    #     map_new_standard.append(small_ind)
    #print(len(orig),len(map_new_standard))
    #distances = np.array([euclidean_distances(centroid1[i].reshape(1,-1),heatmap2[c].reshape(1,-1)).item() for c,i in enumerate(map_new_standard)  ])
    cluster_dict = {}
    for c,i in enumerate(map_new_standard):
        if i not in cluster_dict:
            cluster_dict[i] = []
        cluster_dict[i].append(euclidean_distances(centroid1[i].reshape(1,-1),heatmap2[c].reshape(1,-1)).item())
    # p("unique clusters: " + str(len(set(map_new_standard))),f)
    # p(f=f)
    # for key in cluster_dict.keys():
    #     p("Cluster: " +str(key),f)
    #     distances = np.array(cluster_dict[key])
    #     p("number of items: " +str(len(distances)),f)
    #     p("average_distance: " + str(np.mean(distances)),f)
    #     p("standard deviation: " + str(np.std(distances)),f)
    #     p(f=f)
    # # p(silhouette_score(heatmap1,label1))
    # # p(silhouette_score(heatmap2,map_new_standard))
    # p()
    # print("average_distance: " + str(np.mean(distances)))
    # print("standard deviation: " + str(np.std(distances)))
    if save:
        to_file(label1,names,size_names)
        to_file(map_new_standard,name2)
        cl = Clustering()
        rep_phases = cl.getRepresentativePhases(label1,COLLAPSE_FACTOR,heatmap1.T)
        pickle.dump(rep_phases,open("phases.pkl",'wb'))
        # pickle.dump(heatmap2,open(".pkl",'wb'))
    if viz:
        visualize(names,name2,heatmap1,centroid1,heatmap2,centroid2,label1,map_new_standard,False,metric)
        plt.figure()
    if f!=None:
        f.close()

names = ["cat","cp","echo","findmnt","git","ls"]
name2 = "mkdir"
# name2 = "echo"
# names = ['cat','cp','findmnt','git','ls','mkdir']
# name2 = "findmnt"
# names = ['cat','cp','echo','git','ls','mkdir']
# name2 = "git"
# names = ['cat','cp','echo','findmnt','ls','mkdir']
k=120
metric = 'euclidean'

if __name__ == "__main__":
   gen(names,name2,k,metric,save=True,viz=True)
