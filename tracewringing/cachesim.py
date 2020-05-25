from __future__ import print_function
import subprocess
import numpy as np

class CacheStats(object):
    def __init__(self, trace_in, sim_args, fetches, misses):
        self.trace_in = trace_in
        self.sim_args = sim_args
        self.fetches = fetches 
        self.misses = misses

    def dump(self):
        print(self.misses/self.fetches)

    def __sub__(self, other):
        raise NotImplementedError

class CacheSimulator(object):

    SIM = './d4-7/dineroIV '
    FETCHES = b'Demand Fetches\t'
    MISSES = b'Demand Misses\t'
    DCACHE = b'l1-dcache'
    ICACHE = b'l1-icache'
    UNICACHE = b'l2-ucache'

    def simulate(self, trace_in, sim_args, trace_type):
        """ Takes an address trace,
            calls out to Pin and simulates trace,
            collects stats from Pin.
            Args:
                trace_in: path to input address trace
                sim_args: arguments to $ simulator
            Returns:
                stats: a CacheStats object
        """
        cmd = CacheSimulator.SIM + sim_args
        with open(trace_in, 'r') as f:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=f)
            output = proc.stdout.read()
        f.close()
        if trace_type == 'm':
            output = output[output.find(CacheSimulator.DCACHE):]
        elif trace_type == 'i':
            output = output[output.find(CacheSimulator.ICACHE):]
        
        start_fetches = output.find(CacheSimulator.FETCHES) + len(CacheSimulator.FETCHES)
        start_misses = output.find(CacheSimulator.MISSES) + len(CacheSimulator.MISSES)
        fetches = float(output[ start_fetches: ].split()[0])
        misses = float(output[ start_misses: ].split()[0])
        stats = CacheStats(trace_in, sim_args, fetches, misses)
        return stats

def run_cachesim(path, trace_type):
    cache_assoc = ['1','4']
    cache_size = ["8192", "16384", "32768", "65536"]
    cache_sim = CacheSimulator()
    miss_rates = []
    trace_type = trace_type[0]
    for assoc in cache_assoc:
        for size in cache_size:
            if trace_type == 'm':
                sim_cmd = '-l1-dsize ' + size + ' -l1-dbsize 64 -l1-dassoc ' + assoc + ' -warmupcount 100000'
            elif trace_type == 'i':
                sim_cmd = '-l1-isize ' + size + ' -l1-ibsize 64 -l1-iassoc ' + assoc + ' -warmupcount 100000'
            cache_stats = cache_sim.simulate(path, sim_cmd, trace_type)
            miss_rates.append(cache_stats.misses/cache_stats.fetches)
    return miss_rates

def get_mr_errs(wl_path, proxy_path, trace_type):
    mr_wl = np.array(run_cachesim(wl_path, trace_type))
    mr_proxy = np.array(run_cachesim(proxy_path, trace_type))

    mr_abs_err = np.abs(mr_wl - mr_proxy)
    mr_rel_err = np.abs((mr_wl - mr_proxy)*100/mr_wl)
    
    errors = (np.mean(mr_abs_err), np.mean(mr_rel_err))
    return np.mean(mr_proxy), errors

if __name__ == "__main__":
    import sys 
    assert sys.argv[1], 'Usage --> python3 cachesim.py $wl_path'
    wl_path = sys.argv[1]
    trace_type = 'mem'
    mr = run_cachesim(wl_path, trace_type)
    print('Average miss rate: {}'.format(np.mean(mr)))