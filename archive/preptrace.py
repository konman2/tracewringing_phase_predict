# exec(open('preptrace.py').read())

""" Takes files of traces from workload/aes_traces and converts them as so:
    
    "addr" --> "M addr 4"

    Saves the new files in workload/aes
"""

import numpy as np 
import os 

if not os.path.exists('workload/aes'):
    os.mkdir('workload/aes')

for k in range(1000,10000):
    trace = np.loadtxt('aes/{}.txt'.format(str(k)),dtype=str)
    addr = []
    for i in trace:
        addr.append('M '+i+' 4') 
    addr = np.array(addr)
    np.savetxt('new_aes/aes{}'.format(str(k)), addr, fmt='%s')
