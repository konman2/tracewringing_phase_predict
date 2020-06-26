import sys
sys.path.append('..')
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


def gen_heatmap(names,verbose=True):
    heatmap_fig = []
    sizes = []
    if type(names) is not list and type(names) is not tuple:
        names = [names]
    #cl_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/clusters/{}/{}'.format(method,"-".join(names)+"-"+str(clusters)+"-"+dist+end)
    for name in names:
        wl_path =  '/home/mkondapaneni/Research/tracewringing_phase_predict/traces/{}.trace'.format(name)
        hm_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/heatmap_only/heatmaps/{}'.format(name+"-"+str(WINDOW_SIZE))
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
    return heatmap_fig,sizes



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

def visualize(names,name2,heatmap1,heatmap2,save=False,path=None):

    hg = HeatmapGenerator()
    cl = Clustering()
    # heatmap2-=np.mean(heatmap2)
    # b-=np.mean(b)
    prev= None
    print(heatmap1.shape,heatmap2.shape)
    hg.getHeatmapFigure((heatmap2),name2)
    # hg.compareHeatmaps((np.sqrt(heatmap2),np.sqrt()),name2,False,titles=('orig','frankenstein'),path=path)
    curr_ind = 0
    for c,name in enumerate(names):
        hm_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/heatmap_only/heatmaps/{}'.format(name+"-"+str(WINDOW_SIZE))
        if os.path.exists(hm_path):
            with open(hm_path,'rb') as f:
                hm_matrix = pickle.load(f)
        hm = np.sqrt(np.sqrt(hm_matrix))
        
        hg.getHeatmapFigure(hm,name)
    # hg.getHeatmapFigure(heatmap2.T,"findmnt")
    # hg.getHeatmapFigure(b.T,"findmnt")

def shift_and_score(a,b):
    darkest_b = np.argmax(a)
    darkest_a = np.argmax(b)
    
    shifted = np.roll(a,darkest_b-darkest_a)
    # print(len(shifted))
    # print(len(b))
    return np.linalg.norm(shifted-b)

seq_len = 50
def get_sequences(names):
    if type(names) is not list and type(names) is not tuple:
        names = [names]
    sequences = [] 
    length = seq_len+1
    for c,name in enumerate(names):
        hm_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/heatmap_only/heatmaps/{}'.format(name+"-"+str(WINDOW_SIZE))
        if os.path.exists(hm_path):
            with open(hm_path,'rb') as f:
                hm_matrix = pickle.load(f)
        hm = np.sqrt(np.sqrt(hm_matrix))
        print(hm.shape)
        for i in range(length, len(hm.T)):
            seq = hm.T[i-length:i]
            sequences.append([seq[:-1],seq[-1]])
    return sequences
def gen(names,name2,save=True,viz=False,verbose=True,log_file=None):
    heatmap1,size_names = gen_heatmap(names,verbose)
    f = None
    if log_file != None:
        f = open("./logs/"+log_file+".log",'w')
    heatmap2,_= gen_heatmap(name2,verbose)
    print(heatmap1.shape)
    print(heatmap2.shape)
    if viz:
        visualize(names,name2,heatmap1,heatmap2,False)
    train_data = get_sequences(names)
    val_data = get_sequences(name2)
    print("num_seq:",len(train_data))
    # for i in val_data:
    #     print()
    #     print(i[0])
    #     print()
    #     print(i[1])
    #     print()
    return train_data,val_data
    # for i in X_val:
    #     print(i.shape)

  

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
   gen(names,name2,save=False,viz=False)
