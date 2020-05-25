"""In this script, we try to sweep the parameter space for the wringing function.
The parameters we could sweep are given below, but we can also fix many of these to simplify the optimization problem.

We assume the following dir structure. $benchmarkname/mem/m1.trace (to be wrung). $benchmarkname is one of: scp, scp_keys, py, hmmer, and is set as NAME below. 

window_size = 10000
blocksize = 
clusters = 
threshold = 
line_gap = 
line_length = 
filter_percent = 
height = 2048
collapse_factor = 1
"""

import numpy as np 
import scipy.optimize as optimize
from .wring_opt import *
import sys 

# NAME = sys.argv[1]
NAME = 'gcc-1B'
TYPE = 'mem'
ID = '1'

# WINDOW_SIZE = 10000
WINDOW_SIZE = 100
HEIGHT = 2048
COLLAPSE_FACTOR = 1

sweep_path = 'sweep/{}_{}_{}_{}_{}_{}.csv'.format(NAME, TYPE, ID, WINDOW_SIZE, HEIGHT, COLLAPSE_FACTOR)

init_params = [
    5,  # thres, 
    13, # gap, 
    26, # length, 
    3,  # blocksize, 
    58, # fp (filter percent)
    3   # clusters
]   

bound = [
    (1,60),     # thres, 
    (1,60),     # gap, 
    (1,60),     # length, 
    (1,60),     # blocksize, 
    (25,99),    # fp (filter percent)
    (2,5)       # clusters
]
# bound = [
#     (1,60),     # thres, 
#     (1,60),     # gap, 
#     (1,60),     # length, 
#     (40,40),     # blocksize, 
#     (50,50),    # fp (filter percent)
#     (2,5)       # clusters
# ]

def round_up(fp, round_to=10):
    if fp > 90:
        return np.floor(fp)
    else:
        return ((fp // round_to) + 1)*round_to

import pandas as pd 
global df 

cols = ['param_id', 'bits', 'mr', 'abs_err', 'rel_err']
if os.path.exists(sweep_path):
    df = pd.read_csv(sweep_path)

else:
    df = pd.DataFrame(columns=cols)

# def bit_err_product(init_params):
#     thres, gap, length, blocksize, fp, clusters = init_params 
#     thres, gap, length, blocksize, fp, clusters, err, packet = run(NAME, TYPE, ID, thres, gap, length, blocksize, fp, clusters, WINDOW_SIZE, HEIGHT, COLLAPSE_FACTOR)
#     prod = err * packet 
#     return prod


def error(init_params):
    thres, gap, length, blocksize, fp, clusters = init_params
    fp = round_up(fp)
    posints = [int(np.rint(i)) for i in [thres, gap, length, blocksize, fp, clusters]]
    thres, gap, length, blocksize, fp, clusters = posints
    param_id = '{}_{}_{}_{}_{}_{}'.format(thres, gap, length, blocksize, fp, clusters)

    global df
    if param_id in df.param_id: # memoization
        err_row = df.loc[df['param_id'] == param_id]
        rel_err = err_row['rel_err'].values[0]
        
        return rel_err

    else:
        param_id, bits, mr, abs_err, rel_err = run(NAME, TYPE, ID, thres, gap, length, blocksize, fp, clusters, WINDOW_SIZE, HEIGHT, COLLAPSE_FACTOR)
        df_temp = pd.DataFrame([[param_id, bits, mr, abs_err, rel_err]],columns=cols)
        df = df.append(df_temp, ignore_index=True)
        df.to_csv(sweep_path)
    
    return rel_err
# params = set()


p = init_params
p[-1] = 5
error(p)
        
    
#result = optimize.differential_evolution(func=error, bounds=bound, popsize=60, mutation=1.8, updating='deferred', maxiter=6, workers=1)

# print(df)

# df.to_csv(sweep_path)