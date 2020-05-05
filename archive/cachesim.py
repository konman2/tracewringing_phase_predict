from __future__ import print_function
import subprocess

class CacheStats(object):
    def __init__(self, trace_in, sim_args, miss_rate):
        self.trace_in = trace_in
        self.sim_args = sim_args
        self.mr = miss_rate

    def dump(self):
        print('Miss Rate: {}'.format(self.mr))

    def __sub__(self, other):
        raise NotImplementedError

class CacheSimulator(object):

    SIM = './d4-7/dineroIV '
    MR_TAG = b'Demand miss rate\t'
    DCACHE = b'l1-dcache'

    def simulate(self, trace_in, sim_args):
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
        output = output[output.find(CacheSimulator.DCACHE):]
        start = output.find(CacheSimulator.MR_TAG) + len(CacheSimulator.MR_TAG)
        # This only works with the very output of DineroIV std output.
        miss_rate = float(output[start : output.find(b'\t', start)].strip())
        stats = CacheStats(trace_in, sim_args, miss_rate)
        return stats

if __name__ == '__main__':
    import sys
    cache_sim = CacheSimulator()
    sim_cmd = '-l1-dsize ' + sys.argv[2] + ' -l1-dbsize 64 -l1-dassoc ' + \
            sys.argv[3] + ' -warmupcount 100000'
    cache_stats = cache_sim.simulate(sys.argv[1], sim_cmd)

    print('==========SUMMARY=========')
    print('workload: {}'.format(sys.argv[1]))
    print('sim args: {}'.format(sim_cmd))
    print('stats: size {} assoc {} miss_rate {}'.format(sys.argv[2], sys.argv[3], cache_stats.mr))
    print('==========================')
