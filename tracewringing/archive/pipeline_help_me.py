# exec(open('run_pipeline.py').read())

import os
import pickle
from cachesim import CacheSimulator
from options import *
from trace_generator import WorkloadGenerator, TraceGenerator, AddressTrace
from heatmap_generator import Heatmap, HeatmapGenerator
from clustering import Clustering
from hough import HoughLine, HoughComputation
from info import Information
import numpy as np
from itertools import groupby
from skimage.feature import canny
from timeit import default_timer as timer

def main():

    # Parses cmdline arguments.
    parser = get_parser()
    add_trace_options(parser)
    add_clustering_options(parser)
    add_hough_options(parser)
    args = parse_args(parser)

    pname = '{}_m{}_run{}'.format(args.name, str(args.multiplier), str(args.run))
    # Load/Generate workload trace.
    workload = WorkloadGenerator()
    trace = workload.getTrace(args.path, args.bin)
    start = timer()
    hmgen = HeatmapGenerator()
    heatmap = hmgen.getHeatmapMatrix(trace, args.cacheset, args.multiplier, args.window_size)
    
    # heatmap_fig = np.sqrt(np.sqrt(heatmap.matrix[:,:heatmap.matrix.shape[1]//5]))
    hmgen.getHeatmapFigure(np.sqrt(np.sqrt(heatmap.matrix)),save=True,name=args.name)
    
    # raise('')

    # instantiate clustering object
    cl = Clustering()
    labels, centers = cl.kmeans(args.n_clusters,args.collapse_factor,heatmap.matrix)
    cl.getClusterFigure(labels, np.sqrt(np.sqrt(heatmap.matrix)), args.collapse_factor, args.name, save=True, fname=pname)
    if not os.path.exists('labels'):
        os.mkdir('labels')
    with open('labels/' + pname + '.labels', 'wb') as f:
        pickle.dump(labels, f)
        f.close()
    rep_phases = cl.getRepresentativePhases(labels,args.collapse_factor,heatmap.matrix)
    print('clustering complete..cluster image saved!')
    
    # raise('')
   
    # for debugging hough:
    with open('cactus_rep_phases.pkl', 'wb') as f:
        pickle.dump(rep_phases, f)
        
    print('rep phases saved as pickle')
        
    raise('')
    
    # hough computations:
    h = HoughComputation()
    hough_lines_and_images = []
    for i in range(len(rep_phases)):
        if args.iptype == 'sqrt':
            image_alt = np.sqrt(np.sqrt(rep_phases[i])) 
        elif args.iptype == 'canny':
            image_alt = canny(rep_phases[i],1,1,25)
        hlines_image = h.getHoughLines(image_alt,threshold=args.threshold,line_length=args.line_length,
            line_gap=args.line_gap)
        hough_lines_and_images.append(hlines_image)
        
        # generating images:
        name = '{}_m{}_run{}_phase_{}'.format(args.name, str(args.multiplier), str(args.run), i)
        h.plotHoughLines(hlines_image[0],rep_phases[i],image_alt,save=True, fname=name)
    print('hough figures saved..')
    
    # raise('')
    
    hough_lines, images = zip(*hough_lines_and_images)
    if not os.path.exists('hough_lines'):
        os.mkdir('hough_lines')
    with open('hough_lines/' + pname + '.hls', 'wb') as f:
        pickle.dump(labels, f)
        f.close()
    phases_and_ids = dict(zip(range(len(hough_lines)), hough_lines))

    # mapping hough lines to their labels:
    counts = [sum(1 for _ in group) for _, group in groupby(labels)]
    cumsum = np.cumsum(counts)
    label_called = [labels[i-1] for i in cumsum]
    uncollapsed_counts = np.multiply(counts,args.collapse_factor)
    packet = zip(label_called, uncollapsed_counts)
    assert( len(labels)*args.collapse_factor == sum(uncollapsed_counts) )

    # measuring information using info.py:
    info = Information()
    info_content = []
    for i in hough_lines:
        info_content.append((info.linesInfo(i)))
    
    assert( len(rep_phases) == len(info_content) )

    print('Information given away: {}'.format(sum(info_content)))

    # raise('')
    # trace generation:
    trace_gen = TraceGenerator()
    proxy_trace = AddressTrace()
    print('Number of phases for address generation: {}'.format(len(uncollapsed_counts)))
    for i,j in packet:
        proxy_trace.concat(trace_gen.generateTraceFromHoughLines(
            phases_and_ids[i], j, args.window_size, args.cacheset * args.multiplier, args.blocksize))
    
    print('saving proxy trace to workload/proxy/' + pname)
    assert len(proxy_trace) == len(labels) * args.window_size * args.collapse_factor, \
            '{} != {}'.format(len(proxy_trace), len(labels) * args.window_size * args.collapse_factor)
    proxy_trace.dump2file('workload/proxy/'+pname)
    end = timer()
    print('Time taken: {}s'.format(end-start))
    
    cache_sim = CacheSimulator()
    cache_stats = cache_sim.simulate('workload/proxy/' +pname, '-l1-isize 16k -l1-dsize 16k'\
            ' -l1-ibsize 32 -l1-dbsize 64 -l1-dassoc 4 -warmupcount 100000')
    miss_rate = cache_stats.mr
    print('Workload: {}, Miss Rate: {}'.format(pname,miss_rate))
    
    # to generate heatmap of new trace
    #new_heatmap = hmgen.getHeatmapMatrix(proxy_trace, args.cacheset, args.multiplier, args.window_size)
    #hmgen.compareHeatmaps(heatmap.matrix, new_heatmap.matrix, args.name,save=True)

if __name__ == '__main__':
    main()
