import argparse

# Usage example in run_pipeline.py

def get_parser():
    parser = argparse.ArgumentParser()
    return parser

def parse_args(parser):
    args = parser.parse_args()
    return args

def add_trace_options(parser):
    parser.add_argument('--path', action='store', dest='path',
            help='Path to trace file.')
    parser.add_argument('--proxypath', action='store', default='workload/proxy/',
            help='Path to save proxy trace.')
    parser.add_argument('--bin', action='store', nargs='+', dest='bin',
            help='cmd to call workload binary.')
    #parser.add_argument('--multiplier', action='store', type=int,
    #        help='multiplier to generate heatmap.')
    parser.add_argument('--name', action='store', type=str,
            help='name of the workload')
    parser.add_argument('--run', action='store', type=int, default=None,
            help='which run of the trace is this? (for variance)')
    parser.add_argument('--window-size', action='store', type=int, dest='window_size',
            help='window size of the heatmap.')
    parser.add_argument('--cachesize', action='store', type=int, dest='cachesize',
            help='size of the cache')
    parser.add_argument('--assoc', action='store', type=int, dest='assoc', 
            help='associativity of the cache')
    parser.add_argument('--blocksize', action='store', type=int, default=None,
            help='size for block-based address generation.')
    parser.add_argument('--height', action='store', type=int, default=2048,
            help='rows in the heatmap')
    parser.add_argument('--fast', action='store_true', default=False,
            help='fast run without saving labels and hough_lines.')

def add_clustering_options(parser):
    parser.add_argument('--n_clusters', action='store', dest='n_clusters',
            type=int, help='Number of clusters for KMeans')
    parser.add_argument('--collapse_factor', action='store', dest='collapse_factor', default=1,
            type=int, help='By how much to compress/collapse the heatmap_matrix before KMeans')

def add_hough_options(parser):
    parser.add_argument('--iptype', action='store', dest='iptype', type=str,
            help='Type of image pre-processing: sqrt/canny')
    parser.add_argument('--threshold', action='store', dest='threshold', type=int,
            help='The threshold value to detect the lines, ~150')
    parser.add_argument('--line_length', action='store', dest='line_length', type=int,
            help='The length of the line the hough computation is trying to detect, ~50')
    parser.add_argument('--line_gap', action='store', dest='line_gap', type=int,
            help='The minimum gap between the lines found, ~2')
    parser.add_argument('--filter_percent', action='store', type=int, default=0,
            help='percentage of hough lines to filter out')
    parser.add_argument('--theta_factor', action='store', type=float, default=1, nargs='+',
            help='a factor multiplied to pi/2; pi/2 -> horizontal lines and pi -> vertical lines; so --theta 1 detects horizontal lines')


