from skimage.metrics import mean_squared_error as mse
from skimage.metrics import mean_squared_error as n_
import numpy as np 
import pickle
from tracewringing.heatmap_generator import HeatmapGenerator

def metric(heatmap1,heatmap2):
    assert heatmap1.shape == heatmap2.shape
    base = np.zeros(heatmap1.shape)
    total_both_zeros = np.sum(np.logical_and(heatmap1==base,heatmap2==base))
    total_points = heatmap1.shape[0]*heatmap1.shape[1]
    score = mse(heatmap1,heatmap2)*total_points
    if total_both_zeros == total_points:
        return 0.0
    return score/(total_points-total_both_zeros)

r = np.zeros((6,7))
r1 = np.ones((6,7))*3*np.random.rand(1)
r1 = np.random.rand(6,7)
print(r1)
hg = HeatmapGenerator()
hg.getHeatmapFigure(r1,"asfhj")
hg.compareHeatmaps((r,r1),str(metric(r,r1)))
hm_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/heatmaps/{}'.format('mkdir')
heatmap_orig =np.sqrt(np.sqrt(pickle.load(open(hm_path,'rb'))))

hg.compareHeatmaps((heatmap_orig,heatmap_orig/(np.max(heatmap_orig.flatten()))),"jkh")
print(np.max(heatmap_orig.flatten()))
#hg.compareHeatmaps((heatmap_orig.T,heatmap1.T,heatmap2.T,heatmap_modes.T),val_name,titles=('original','perfect clustering','lstm','modes'))
print(metric(r1,r))