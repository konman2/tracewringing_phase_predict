"""Wring for optimization.
"""
import os
import pickle
import numpy as np 
from clustering import Clustering
from hough import HoughLine, HoughComputation
import struct
import math
import tarfile
from itertools import groupby
from trace_generator import TraceGenerator, AddressTrace
from cachesim import CacheStats, CacheSimulator, get_mr_errs

def generate_heatmap(wl_path, height, window_size, hm_path=None):
    """Generates heatmap from trace at given path and saves it in savepath
    
    Arguments:
        wl_path {str} -- path to the workload
    """
    from trace_generator import WorkloadGenerator
    from heatmap_generator import HeatmapGenerator

    workload_gen = WorkloadGenerator()
    trace = workload_gen.getTrace(wl_path)

    hmgen = HeatmapGenerator()
    heatmap = hmgen.getHeatmapMatrix(trace, height, window_size)
    hm_matrix = heatmap.matrix

    if hm_path is not None:
        with open(hm_path, 'wb') as f:
            pickle.dump(hm_matrix,f)

    return hm_matrix

def plot_heatmap(heatmap):
    import matplotlib.pyplot as plt 
    plt.imshow(heatmap, cmap='gray_r')
    plt.show()

def encodeRLE(encodeStr, collapse_factor):
    """Run-length encoding for cluster labels. """
    encoded = ""
    count = 0
    i = 0
    character = encodeStr[i]
    for i in range(0,len(encodeStr)):
        if(encodeStr[i] == character):
            count += collapse_factor
        else:
            encoded += (str(character)+""+str(count))
            character = encodeStr[i]
            count = collapse_factor
        if(i==(len(encodeStr)-1)):
            encoded += (str(character)+""+str(count))
    return encoded

# generate quantized hough packet and write it 
def generate_hough_packets(hl_quantized_path, hough_lines, formatspec='4h1B', pow2=4096):
    quantized_hl = []
    with open(hl_quantized_path, 'wb') as f:
        for rep in hough_lines:
            quantized_hl.append(len(rep))
            for h in rep:
                if h.weight > 255/4096: # i suppose there are *few* weights that are this high
                    # print('Weight getting saturated')
                    weight = 255/4096
                else:
                    weight = h.weight
                packed = struct.pack(formatspec, h.x1, h.y1, h.x2, h.y2, math.ceil((weight*pow2)))
                f.write(packed)
        f.close()
    return quantized_hl

def read_hough_packets(hl_quantized_path, quantized_hl, formatspec='4h1B', pow2=4096):
    hough_lines_quantized = []
    with open(hl_quantized_path,'rb') as f:
        for i in range(len(quantized_hl)):
            rep_quantized = []
            for j in range(quantized_hl[i]):
                line = f.read(9)
                x1, y1, x2, y2, weight = struct.unpack(formatspec,line)
                rep_quantized.append(HoughLine(x1, y1, x2, y2, float(weight/pow2)))
            hough_lines_quantized.append(rep_quantized)
    return hough_lines_quantized

def run(trace_name, trace_type, trace_id, thres, gap, length, blocksize, fp, clusters, window_size=10000, height=2048, collapse_factor=1):

    param = [thres, gap, length, blocksize, fp, clusters]
    param_name = ['Threshold', 'Line Gap', 'Line Length', 'Block Size', 'Filter Percentage (partition)', 'Clusters']

    for i in range(len(param)):
        param[i] = int(np.rint(param[i]))

        assert isinstance(param[i],int) and param[i] > 0, '{} ({}) should be a positive integer.'.format(param_name[i],param[i])

    thres, gap, length, blocksize, fp, clusters = param
    wl_path = '{}/{}/{}{}.trace'.format(trace_name, trace_type, trace_type[0], trace_id)
    #print(wl_path)
    param_str = '{}_{}_{}_{}_{}_{}'.format(str(thres), str(gap), str(length), str(blocksize), str(fp), str(clusters))
    id_string = '{}_{}{}_{}'.format(trace_name, trace_type[0], trace_id, param_str)
    hm_path = 'heatmaps/{}_{}{}'.format(trace_name, trace_type[0], trace_id)
    cluster_path = 'labels/{}.rle'.format(id_string)
    hl_quantized_path = 'hough_lines/{}.fxdpt'.format(id_string)
    packet_path = 'packets/{}.tar.bz2'.format(id_string)
    proxy_path = 'proxy/{}.trace'.format(id_string)

    if os.path.exists(hm_path):
        with open(hm_path,'rb') as f:
            print(hm_path)
            hm_matrix = pickle.load(f)

    else:
        hm_matrix = generate_heatmap(wl_path, height, window_size, hm_path)
    
    heatmap_fig = np.sqrt( np.sqrt( hm_matrix ))
    
    cl = Clustering()
    labels, _ = cl.kmeans(clusters,collapse_factor,heatmap_fig)
    print(labels[:1000])
    label_rle = encodeRLE(labels, collapse_factor)
    #print(label_rle)
    with open(cluster_path, 'wb') as f:
        f.write(label_rle.encode())

    rep_phases = cl.getRepresentativePhases(labels, collapse_factor, heatmap_fig)

    h = HoughComputation()
    hough_lines_and_images = []
    for i in range(len(rep_phases)):
        hlines_image = h.getHoughLines(rep_phases[i],thres,length,gap,fp)
        hough_lines_and_images.append(hlines_image)

    hough_lines, _ = zip(*hough_lines_and_images)
    
    quantized_hl = generate_hough_packets(hl_quantized_path, hough_lines)
    _ = [generate_hough_packets('{}/mem/{}_phase_{}'.format(trace_name, param_str, ki), [houghline]) for ki, houghline in enumerate(hough_lines)]

    tar = tarfile.open(packet_path, 'w:bz2')
    for i in [hl_quantized_path, cluster_path]:
        tar.add(i)
    tar.close()

    counts = [sum(1 for _ in group) for _, group in groupby(labels)]
    cumsum = np.cumsum(counts)
    label_called = [labels[i-1] for i in cumsum]
    uncollapsed_counts = np.multiply(counts,collapse_factor)
    packet = zip(label_called, uncollapsed_counts)

    assert( len(labels)*collapse_factor == sum(uncollapsed_counts) )

    hough_lines_quantized = read_hough_packets(hl_quantized_path, quantized_hl)

    all_traces = [TraceGenerator(hough_lines_quantized[i], rep_phases[i].shape[1],
        window_size, height, blocksize)
        for i in range(len(rep_phases))]

    proxy_trace = AddressTrace()
    for i,j in packet:
        proxy_trace.concat(all_traces[i].generateTrace(j))

    assert len(proxy_trace) == len(labels) * window_size * collapse_factor, '{} != {}'.format(len(proxy_trace), len(labels) * window_size * collapse_factor)
    
    proxy_trace.dump2file(proxy_path, trace_type[0])

    packet_size = os.path.getsize(os.getcwd() + '/' + packet_path) * 8
    mr_proxy, (abs_err, rel_err) = get_mr_errs(wl_path, proxy_path, trace_type[0])
    # import pdb; pdb.set_trace()

    # clean up: delete all generated files:
    os.remove(packet_path)
    os.remove(hl_quantized_path)
    os.remove(proxy_path)
    os.remove(cluster_path)

    print('{}_{}_{}_{}_{}_{}, {}, {}, {}, {}'.format(thres, gap, length, blocksize, fp, clusters, mr_proxy, packet_size, abs_err, rel_err))
    # return thres, gap, length, blocksize, fp, clusters, errors[0], packet_size
    return '{}_{}_{}_{}_{}_{}'.format(thres, gap, length, blocksize, fp, clusters), mr_proxy, packet_size, abs_err, rel_err

if __name__ == "__main__":
    import sys 
    assert sys.argv[1], 'Usage --> python wring.py $trace_name $trace_id $param_str'

    trace_name = sys.argv[1] 
    trace_type = 'mem'
    trace_id = sys.argv[2]
    pareto_str = sys.argv[3].split('_')

    thres = int(pareto_str[0])
    gap = int(pareto_str[1])
    length = int(pareto_str[2])
    blocksize = int(pareto_str[3])
    fp = int(pareto_str[4])
    clusters = int(pareto_str[5])
    window_size = 10000
    height = 2048
    collapse_factor = 1

    param_id, bits, mr_proxy, abs_err, rel_err = run(trace_name, trace_type, trace_id, thres, gap, length, blocksize, fp, clusters, window_size, height, collapse_factor)