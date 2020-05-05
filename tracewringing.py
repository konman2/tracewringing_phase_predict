import os
import numpy as np
from itertools import groupby
import struct
import math
import sys
import subprocess
import random
import pickle
import csv
from cachesim import CacheStats, CacheSimulator
from clustering import Clustering
from heatmap_generator import Heatmap, HeatmapGenerator
from hough import HoughLine, HoughComputation
from trace_generator import WorkloadGenerator, TraceGenerator, AddressTrace

class Pipeline:
    
    def __init__(self, path):
        self.path = path
        self.wd = self.getWD(path)
        self.filename = self.getFilename(path)
        self.trace_size = self.getTraceSize(path)
        
    def getWD(self,path):
        """Returns the directory that contains the trace from the path. 
        For example: if path = 'home/user/workload/git/1.trace.dinero' then getWD(path) will return '/home/user/workload/git'

        :param String path: path of the trace
        """
        if path[-1] == '/':
            path = path[:-1]
        index = path.rfind('/')
        wd = path[:index]
        return wd
    
    def getFilename(self,path):
        """Returns the filename of the trace from the path.
        For example: if path = '/home/user/workload/git/1.trace.dinero' then getFilename(path) will return '1.trace.dinero'

        :param String path: path of the trace
        """
        if path[-1] == '/':
            path = path[:-1]
        start_index = path.rfind('/')
        filename = path[start_index+1:]
        return filename
    
    def getTraceSize(self,path):
        """Returns the size of the trace in bits.

        :param String path: path of the trace
        """
        ls_cmd = ["ls","-al"]
        proc = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, cwd=self.wd)
        output = proc.stdout.read()
        output_str = output.decode()
        for line in output_str.split("\n"):
            if self.filename in line:
                line = line.split()
                return int(line[4])
    
    def encodeRLE(self, encodeStr, collapse_factor):
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
        
    def run(self, save_image, trace_type, window_size, height, blocksize, clusters, thres, gap, length, fp):
        theta = 1.0
        collapse_factor = 1

        end_index = self.filename.find('.')
        savename = self.filename[:end_index]
        
        # create and save heatmap if it doesn't exist
        hmgen = HeatmapGenerator()
        heatmap_name = savename + '_' + str(window_size)
        if not os.path.exists('heatmaps'):
            os.mkdir('heatmaps')
        if not os.path.exists('heatmaps/'+heatmap_name):
            workload_gen = WorkloadGenerator()
            trace = workload_gen.getTrace(self.path)

            heatmap = hmgen.getHeatmapMatrix(trace, height, window_size)
            hm_matrix = heatmap.matrix
            
            f = open('heatmaps/'+heatmap_name, 'wb')
            pickle.dump(hm_matrix,f)
            f.close()
            
            heatmap_fig = np.sqrt(np.sqrt(hm_matrix))
            hmgen.getHeatmapFigure(heatmap_fig,str(window_size),save_image,heatmap_name)
        
        # load heatmap
        f = open('heatmaps/'+heatmap_name,'rb')
        saved_hm_matrix = pickle.load(f)
        saved_heatmap_fig = np.sqrt(np.sqrt(saved_hm_matrix))
        
        cl = Clustering()
        labels, centers = cl.kmeans(clusters,collapse_factor,saved_heatmap_fig)
        #labels = np.repeat(labels,collapse_factor)
        cl.getClusterFigure(labels,saved_heatmap_fig,collapse_factor,height,savename,save_image,savename)
        
        if not os.path.exists('labels'):
            os.mkdir('labels')
        rle_path = os.getcwd()+'/labels/' + savename + '.camera.rle.labels'
        with open(rle_path, 'wb') as f:
            label_rle = self.encodeRLE(labels,collapse_factor)
            f.write(label_rle.encode())
            f.close()

        rep_phases = cl.getRepresentativePhases(labels,collapse_factor,saved_hm_matrix)

        h = HoughComputation()
        hough_lines_and_images = []
        for i in range(len(rep_phases)):
            image_alt = np.sqrt(np.sqrt(rep_phases[i]))
            hlines_image = h.getHoughLines(image_alt,thres,length,gap,theta,fp)
            hough_lines_and_images.append(hlines_image)

        hough_lines, images = zip(*hough_lines_and_images)

        if not os.path.exists('hough_lines'):
            os.mkdir('hough_lines')
        formatspec = '4h1B'

        hl_path = os.getcwd()+'/hough_lines/' + savename + '.camera.fixedpoint.hls'
        my_list = []
        with open(hl_path, 'wb') as f:
            pow2 = 4096
            for rep in hough_lines:
                my_list.append(len(rep))
                for h in rep:
                    if h.weight > 255/4096:
                        weight = 255/4096
                    else:
                        weight = h.weight
                    packed = struct.pack(formatspec, h.x1, h.y1, h.x2, h.y2, math.ceil((weight*pow2)))
                    f.write(packed)
            f.close()

        if not os.path.exists('bz2'):
            os.mkdir('bz2')
        tar_name = savename + '.tar.bz2'
        tar_cmd = ["tar", "-cjvf", tar_name, hl_path, rle_path]
        bz2_wd = os.getcwd()+'/bz2/'
        subprocess.Popen(tar_cmd, cwd=bz2_wd)

        counts = [sum(1 for _ in group) for _, group in groupby(labels)]
        cumsum = np.cumsum(counts)
        label_called = [labels[i-1] for i in cumsum]
        uncollapsed_counts = np.multiply(counts,collapse_factor)
        packet = zip(label_called, uncollapsed_counts)

        assert( len(labels)*collapse_factor == sum(uncollapsed_counts) )
        
        hough_lines_quantized = []
        with open(hl_path,'rb') as f:
            for i in range(len(my_list)):
                rep_quantized = []
                for j in range(my_list[i]):
                    line = f.read(9)
                    x1,y1,x2,y2,weight = struct.unpack(formatspec,line)
                    rep_quantized.append(HoughLine(x1,y1,x2,y2,float(weight/pow2)))
                hough_lines_quantized.append(rep_quantized)
                    
        all_traces = [TraceGenerator(hough_lines_quantized[i], rep_phases[i].shape[1],
            window_size, height, blocksize)
            for i in range(len(rep_phases))]

        proxy_trace = AddressTrace()
        for i,j in packet:
            proxy_trace.concat(all_traces[i].generateTrace(j))

        if not os.path.exists('proxy'):
            os.mkdir('proxy')
        proxy_path = 'proxy/'+savename+'.wrung'

        assert len(proxy_trace) == len(labels) * window_size * collapse_factor, \
                '{} != {}'.format(len(proxy_trace), len(labels) * window_size * collapse_factor)
        proxy_trace.dump2file(proxy_path,trace_type)

        heatmap_new = hmgen.getHeatmapMatrix(proxy_trace, height, window_size)
        hmgen.compareHeatmaps(saved_hm_matrix,heatmap_new.matrix,savename,save_image)
    
        ls_cmd = ["ls","-al"]
        proc = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, cwd=bz2_wd)
        output = proc.stdout.read()
        output_str = output.decode()
        for line in output_str.split("\n"):
            if tar_name in line:
                line = line.split()
                packet_size = int(line[4])*8
#                 print("PACKET SIZE:",int(line[4])*8,"bits")

#         print("WINDOW SIZE:", window_size)
#         print("BLOCKSIZE:", blocksize)
#         print("CLUSTERS:", clusters)
#         print("THRESHOLD:", thres)
#         print("GAP:", gap)
#         print("LENGTH:", length)
#         print("FP:", fp)

        cache_size = ["8192", "16384", "32768", "65536"]
        cache_assoc = ['1','4']
        traces = [self.path,proxy_path]
        error_avg = 0
        cache_sim = CacheSimulator()

        for assoc in cache_assoc:
            baseline_miss_rates = []
            scrubbed_miss_rates = []
            for size in cache_size:
                if trace_type == 'm':
                    sim_cmd = '-l1-dsize ' + size + ' -l1-dbsize 64 -l1-dassoc ' + \
                    assoc + ' -warmupcount 100000'
                elif trace_type == 'i':
                    sim_cmd = '-l1-isize ' + size + ' -l1-ibsize 64 -l1-iassoc ' + \
                    assoc + ' -warmupcount 100000'
                miss_rates = []
                for trace in traces:
                    cache_stats = cache_sim.simulate(trace, sim_cmd, trace_type)
                    miss_rates.append(cache_stats.misses/cache_stats.fetches)
                
                baseline_mr = miss_rates.pop()
                baseline_miss_rates.append(baseline_mr)
                scrubbed_mr = miss_rates.pop()
                scrubbed_miss_rates.append(scrubbed_mr)
                
                #error = abs(scrubbed_mr - baseline_mr)/baseline_mr * 100
                error = abs(scrubbed_mr - baseline_mr)
                error_avg += error
#                 print("Cache Size:",size,"\tAssociativity:",assoc,"\tError:",error)
                
        error_avg /= 8
#         print("Average Error:", error_avg)
#         print("\n")
        return [error_avg, packet_size, window_size, blocksize, clusters, thres, gap, length, fp]
    
    def highLeakageParameters(self):
        trace_type = 'm'
        save_image = True
        
        MAX_WINDOW_SIZE = 10000
        HEIGHT = 2048
        MIN_BLOCKSIZE = 1
        MIN_CLUSTERS = 1
        MIN_THRES = 1
        MIN_GAP = 1
        MIN_LENGTH = 1
        MIN_FP = 0
        
        SMALL_TRACE = 200000
        MEDIUM_TRACE = 50000000
        SMALL_WINDOW = 10
        MEDIUM_WINDOW = 100
        LARGE_WINDOW = 1000
       
        blocksizes = [3,5,10,25,50,100]
        clusters = [2,3,4,5]
        
        # Set minimum window size
        if self.trace_size <= SMALL_TRACE:
            MIN_WINDOW_SIZE = SMALL_WINDOW
        elif self.trace_size <= MEDIUM_TRACE:
            MIN_WINDOW_SIZE = MEDIUM_WINDOW
        else:
            MIN_WINDOW_SIZE = LARGE_WINDOW
        
        # Create table to hold results from each run
        results_table = []
        
        # Find best window size
        lowest_error = 100000000
        window_size = MAX_WINDOW_SIZE
        while window_size >= MIN_WINDOW_SIZE:
            results = self.run(save_image,trace_type,window_size,HEIGHT,MIN_BLOCKSIZE,MIN_CLUSTERS,MIN_THRES,MIN_GAP,MIN_LENGTH,MIN_FP)
            results_table.append(results)
            error = results[0]
            if error < lowest_error:
                lowest_error = error
                bits_leaked = results[1]
                best_window_size = window_size  
                window_size = int(window_size/10)
            else:
                break

         # Find best blocksize
        best_blocksize = MIN_BLOCKSIZE
        for blocksize in blocksizes:
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,blocksize,MIN_CLUSTERS,MIN_THRES,MIN_GAP, MIN_LENGTH,MIN_FP)
            results_table.append(results)
            error = results[0]
            if error < lowest_error:
                lowest_error = error
                bits_leaked = results[1]
                best_blocksize = blocksize
            else:
                break

        # Find best number of clusters
        best_clusters = MIN_CLUSTERS
        for cluster in clusters:
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,cluster,MIN_THRES,MIN_GAP,MIN_LENGTH,MIN_FP)
            results_table.append(results)
            error = results[0]
            if error < lowest_error:
                lowest_error = error
                bits_leaked = results[1]
                best_clusters = cluster

                # Find best length
        best_length = MIN_LENGTH
        for length in range(5,51,5):
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,best_clusters,MIN_THRES,MIN_GAP,length,MIN_FP)
            results_table.append(results)
            error = results[0]
            if error < lowest_error:
                lowest_error = error
                bits_leaked = results[1]
                best_length = length
            else:
                break        
        
                # Find best gap
        best_gap = MIN_GAP
        for gap in range(5,51,5):
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,best_clusters,MIN_THRES,gap,best_length,MIN_FP)
            results_table.append(results)
            error = results[0]
            if error < lowest_error:
                lowest_error = error
                bits_leaked = results[1]
                best_gap = gap
            else:
                break
                
        # Find best threshold
        best_thres = MIN_THRES
        for thres in range(20,101,20):
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,best_clusters,thres,best_gap,best_length,MIN_FP)
            results_table.append(results)
            error = results[0]
            if error < lowest_error:
                lowest_error = error
                bits_leaked = results[1]
                best_thres = thres
            
        # Find best fp
        best_fp = MIN_FP
        fps = [25,50,75,90]
        for fp in fps:
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,best_clusters,best_thres,best_gap,best_length,fp)
            results_table.append(results)
            error = results[0]
            bits_leaked = results[1]
            if error < lowest_error:
                lowest_error = error
                bits_leaked = results[1]
                best_fp = fp
                
        return (results_table, [lowest_error, bits_leaked, best_window_size, best_blocksize, best_clusters, best_thres, best_gap, best_length, best_fp])
#         print("*********************************HIGH LEAKAGE PARAMETERS*************************************")
#         best_error, bits_leaked  = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,best_clusters,best_thres,best_gap,best_length,best_fp)
#         return best_error, bits_leaked, best_window_size, best_blocksize, best_clusters, best_thres, best_gap, best_length, best_fp
        
        
    def lowLeakageParameters(self, high_leakage_parameters):
        trace_type = 'm'
        save_image = True
        
        # extracting high leakage parameters
        best_error = high_leakage_parameters[0]
        bits_leaked = high_leakage_parameters[1]
        high_window_size = high_leakage_parameters[2]
        high_blocksize = high_leakage_parameters[3]
        high_clusters = high_leakage_parameters[4]
        high_thres = high_leakage_parameters[5]
        high_gap = high_leakage_parameters[6]
        high_length = high_leakage_parameters[7]
        high_fp = high_leakage_parameters[8]
        
        # Create table to hold results from each run
        results_table = []
        
        # max error we're willing to tolerate
        max_error = 1.25*best_error
        
        MAX_WINDOW_SIZE = 10000
        HEIGHT = 2048
        
        SMALL_TRACE = 200000
        MEDIUM_TRACE = 50000000
        SMALL_WINDOW = 10
        MEDIUM_WINDOW = 100
        LARGE_WINDOW = 1000
       
        blocksizes = [1,3,5,10,25,50,100]
        clusters = [1,2,3,4,5]
        
        # Set minimum window size
        if self.trace_size <= SMALL_TRACE:
            MIN_WINDOW_SIZE = SMALL_WINDOW
        elif self.trace_size <= MEDIUM_TRACE:
            MIN_WINDOW_SIZE = MEDIUM_WINDOW
        else:
            MIN_WINDOW_SIZE = LARGE_WINDOW

        # Find best window size
        min_bits_leaked = bits_leaked
        low_leakage_error = best_error
        
        best_window_size = high_window_size
        window_size = MAX_WINDOW_SIZE
        while window_size >= MIN_WINDOW_SIZE:
            results = self.run(save_image,trace_type,window_size,HEIGHT,high_blocksize,high_clusters,high_thres,high_gap,high_length,high_fp)
            results_table.append(results)
            error = results[0]
            bits_leaked = results[1]
            if bits_leaked < min_bits_leaked and error < max_error:
                min_bits_leaked = bits_leaked
                low_leakage_error = error
                best_window_size = window_size  
                window_size = int(window_size/10)
            elif error > max_error:
                window_size = int(window_size/10)
            else:
                break

        # Find best blocksize
        best_blocksize = high_blocksize
        for blocksize in blocksizes:
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,blocksize,high_clusters,high_thres,high_gap,high_length,high_fp)
            results_table.append(results)
            error = results[0]
            bits_leaked = results[1]
            if bits_leaked < min_bits_leaked and error < max_error:
                min_bits_leaked = bits_leaked
                low_leakage_error = error
                best_blocksize = blocksize
            else:
                break

        # Find best number of clusters
        best_clusters = high_clusters
        for cluster in clusters:
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,cluster,high_thres,high_gap,high_length,high_fp)
            results_table.append(results)
            error = results[0]
            bits_leaked = results[1]
            if bits_leaked < min_bits_leaked and error < max_error:
                min_bits_leaked = bits_leaked
                low_leakage_error = error
                best_clusters = cluster

        # Find best length
        best_length = high_length
        for length in range(5,51,5):
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,best_clusters,high_thres,high_gap,length,high_fp)
            results_table.append(results)
            error = results[0]
            bits_leaked = results[1]
            if bits_leaked < min_bits_leaked and error < max_error:
                min_bits_leaked = bits_leaked
                low_leakage_error = error
                best_length = length
            else:
                break        
        
        # Find best gap
        best_gap = high_gap
        for gap in range(5,51,5):
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,best_clusters,high_thres,gap,best_length,high_fp)
            results_table.append(results)
            error = results[0]
            bits_leaked = results[1]
            if bits_leaked < min_bits_leaked and error < max_error:
                min_bits_leaked = bits_leaked
                low_leakage_error = error
                best_gap = gap
            else:
                break
                
        # Find best threshold
        best_thres = high_thres
        for thres in range(20,101,20):
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,best_clusters,thres,best_gap,best_length,high_fp)
            results_table.append(results)
            error = results[0]
            bits_leaked = results[1]
            if bits_leaked < min_bits_leaked and error < max_error:
                min_bits_leaked = bits_leaked
                low_leakage_error = error
                best_thres = thres
            
        # Find best fp
        best_fp = high_fp
        fps = [25,50,75,90]
        for fp in fps:
            results = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,best_clusters,best_thres,best_gap,best_length,fp)
            results_table.append(results)
            error = results[0]
            bits_leaked = results[1]
            if bits_leaked < min_bits_leaked and error < max_error:
                min_bits_leaked = bits_leaked
                low_leakage_error = error
                best_fp = fp
        
        return (results_table, [low_leakage_error, min_bits_leaked, best_window_size, best_blocksize, best_clusters, best_thres, best_gap, best_length, best_fp])
#         print("*********************************LOW LEAKAGE PARAMETERS*************************************")
#         error, min_bits_leaked  = self.run(save_image,trace_type,best_window_size,HEIGHT,best_blocksize,best_clusters,best_thres,best_gap,best_length,best_fp)
            
        
if __name__ == "__main__":
    
    #p1 = Pipeline('/home/junayed/tracewringing/workload/git/mem/1.trace.dinero')
    p1 = Pipeline('/home/mkondapaneni/Research/tracewringing/gzip/mem/m1.trace')
    high_leakage_tuple = p1.highLeakageParameters()
    high_leakage_table = high_leakage_tuple[0]
    high_leakage_param = high_leakage_tuple[1]
    
    low_leakage_tuple = p1.lowLeakageParameters(high_leakage_param)
    low_leakage_table = low_leakage_tuple[0]
    low_leakage_param = low_leakage_tuple[1]
    
    print('High Leakage Results')
    for row in high_leakage_table:
        print(row)
    print('Best High Leakage Parameters')
    print(high_leakage_param)
    
    print('Low Leakage Results')
    for row in low_leakage_table:
        print(row)
    print('Best Low Leakage Parameters')
    print(low_leakage_param)