# exec(open('run_pipeline.py').read())

import os
import pickle
from array import array
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
    
    start = timer()
    # Parses cmdline arguments.
    parser = get_parser()
    add_trace_options(parser)
    add_clustering_options(parser)
    add_hough_options(parser)
    args = parse_args(parser)

    pname = args.name
    print(pname, args.path, args.bin)
    # Load/Generate workload trace.
    workload = WorkloadGenerator()
    print('Loading trace...')
    trace = workload.getTrace(args.path, args.bin)
    hmgen = HeatmapGenerator()
    print('Generating heatmap...')
    heatmap = hmgen.getHeatmapMatrix(trace, args.height, args.window_size)
    
    heatmap_fig = np.sqrt(np.sqrt(heatmap.matrix))
    #hmgen.getHeatmapFigure(heatmap_fig,str(args.window_size),save=True,name=args.name)
   
    # instantiate clustering object
    cl = Clustering()
    print('Clustering...')
    labels, centers = cl.kmeans(args.n_clusters,args.collapse_factor,heatmap_fig)
    #cl.getClusterFigure(labels, np.sqrt(np.sqrt(heatmap.matrix)), args.collapse_factor, cacheset, args.name, save=True, fname=pname)

    if not args.fast:
        if not os.path.exists('labels'):
            os.mkdir('labels')
        with open('labels/' + pname + '.labels', 'wb') as f:
            label_array = array('d', labels)
            label_array.tofile(f)
            f.close()

    rep_phases = cl.getRepresentativePhases(labels,args.collapse_factor,heatmap.matrix)

    # to debug:
    # with open('milc_h2048_rep_phases.pkl', 'wb') as f:
    #     pickle.dump(rep_phases, f)

    # raise('')

    print('Computing hough lines...')
    h = HoughComputation()
    hough_lines_and_images = []
    for i in range(len(rep_phases)):
        if args.iptype == 'sqrt':
            image_alt = np.sqrt(np.sqrt(rep_phases[i]))
        hlines_image = h.getHoughLines(image_alt,threshold=args.threshold,line_length=args.line_length,line_gap=args.line_gap, theta_factor_list=args.theta_factor,filter_percent=args.filter_percent)
        hough_lines_and_images.append(hlines_image)
        # generating images:
        #name = '{}_phase_{}'.format(args.name, i)
        #h.plotHoughLines(hlines_image[0],rep_phases[i],image_alt,save=True, fname=name)

    hough_lines, images = zip(*hough_lines_and_images)

    if not args.fast:
        n_hl = 0
        if not os.path.exists('hough_lines'):
            os.mkdir('hough_lines')
        with open('hough_lines/' + pname + '.hls', 'wb') as f:
            cord_list = []
            for rep in hough_lines:
                n_hl += len(rep)
                for h in rep:
                    cord_list.extend([h.x1, h.y1, h.x2, h.y2, h.weight])
            cord_array = array('d', cord_list)
            cord_array.tofile(f)
            f.close()
        print('# of hough lines detected: {}'.format(n_hl))

    phases_and_ids = dict(zip(range(len(hough_lines)), hough_lines))

    info = Information()
    info_content = []
    for i in hough_lines:
        if len(i) > 0:
            info_content.append((info.linesInfo(i)))
        else:
            print('no hough lines: zero info given away!')
    
    # mapping representative phases to their labels:
    counts = [sum(1 for _ in group) for _, group in groupby(labels)]
    cumsum = np.cumsum(counts)
    label_called = [labels[i-1] for i in cumsum]
    uncollapsed_counts = np.multiply(counts,args.collapse_factor)
    packet = zip(label_called, uncollapsed_counts)

    assert( len(labels)*args.collapse_factor == sum(uncollapsed_counts) )

    # trace generation:
    print('Generating proxy trace...')
        
    all_traces = [TraceGenerator(hough_lines[i], rep_phases[i].shape[1],
        args.window_size, args.height, args.blocksize)
        for i in range(len(rep_phases))]

    proxy_trace = AddressTrace()
    for i,j in packet:
        proxy_trace.concat(all_traces[i].generateTrace(j))

    dump_path = args.proxypath
    assert len(proxy_trace) == len(labels) * args.window_size * args.collapse_factor, \
            '{} != {}'.format(len(proxy_trace), len(labels) * args.window_size * args.collapse_factor)
    proxy_trace.dump2file(dump_path + pname)
    end = timer()
    
    print('==========SUMMARY=========')
    print('workload: {}'.format(args.path))
    print('hough args: threshold {}, gap {}, length {}'.format(
        args.threshold, args.line_gap, args.line_length))
    print('proxy path: {}'.format(dump_path + pname))
    print('labels saved to: {}'.format('labels/'+pname+'.labels'))
    print('hough lines saved to: {}'.format('hough_lines/'+pname+'.hls'))
    print('Information: {}'.format( sum(info_content)+( len(labels)*3) ))
    print('Time taken: {}s'.format(end - start))
    print('==========================')

    # draw graph:
    #heatmap_new = hmgen.getHeatmapMatrix(proxy_trace, args.height, args.window_size)
    #hmgen.compareHeatmaps(heatmap.matrix,heatmap_new.matrix,name='comparesjeng',save=True)
if __name__ == '__main__':
    main()
