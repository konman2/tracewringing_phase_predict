from __future__ import print_function
from collections import deque
import numpy as np
import os
import subprocess

kBlkBits = 6

class AddressTrace(object):
    def __init__(self, addresses=None):
        if addresses is None:
            addresses = []
        if isinstance(addresses, list):
            addresses = np.asarray(addresses)
        assert isinstance(addresses, np.ndarray)
        self.addrs = addresses

    def concat(self, other):
        assert isinstance(other, AddressTrace), \
                'Cannot concat {} to AddressTrace'.format(type(other))
        self.addrs = np.concatenate((self.addrs, other.addrs))

    def __getitem__(self, key):
        print(self.addrs[key][:len(self.addrs[key])-2])
        return int(self.addrs[key], 0) >> kBlkBits

    def __len__(self):
        return len(self.addrs)

    def dump(self):
        print('Address trace of length {}'.format(len(self)))

    def dump2file(self, path):
        with open(path, 'w') as f:
            f.write('\n'.join(['M ' + addr + ' 4' for addr in self.addrs]))
            f.close()

class WorkloadGenerator(object):
    """ spec06 workload -> address trace
    """

    # TODO: set PIN_HOME to your pin dir, MEM_TRACER to compiled memtrace.so
    PIN_HOME = '/home/weilong/pin-3.2-81205-gcc-linux'
    MEM_TRACER = PIN_HOME + '/source/tools/InstLibExamples/obj-intel64/memtrace.so'
    # Pinpoints for spec06_input: start length
    # bzip2_train/input/byoudoin.jpg: 65100004529 100000005
    # mcf_train/input/inp.in: 8800000541 100000002
    # gcc_train/input/integrate.i: 3000000111 100000006
    # sjeng_train/input/train.txt: 410700017901 100000010
    # hmmer_train/input/leng100.hmm: 4300000854 100000013
    # libquantum 143 25 (train/input/control): 8600000261 100000003
    # milc < train/input/su3imp.in: 16000003368 100000006
    # cactusADM_train/input/benchADM.par: 1300000407 100000061

    MEM_TRACER_ARGS = ' -controller_skip 1300000407 -controller_length 100000061 -addr'

    def getTrace(self, path, bin_cmd=None):
        # Path should not be None.
        assert path, 'Path to trace cannot be empty.'

        # If trace exists.
        if os.path.exists(path):
            with open(path, 'r') as f:
                trace = AddressTrace([l.split(' ')[1] for l in f.readlines()])
                # trace = AddressTrace([l  for l in f.readlines()])
            return trace

        print('Trace file not found, generating trace from executable...')
        # Otherwise, calls out to Pin and generate trace.
        assert bin_cmd, 'Trying to generate trace, no executable given in --bin.'
        cmd = WorkloadGenerator.PIN_HOME + '/pin -t ' + WorkloadGenerator.MEM_TRACER + \
                WorkloadGenerator.MEM_TRACER_ARGS + ' -- ' + ' '.join(bin_cmd)

        print('\n' + cmd + '\n')

        # Run cmd to grab memory trace.
        assert subprocess.call(cmd, shell=True) == 0

        # Move trace file to destination.
        try:
            os.renames('pinatrace.out', path)
            print('Trace file stored at {}'.format(path))
        except:
            raise ValueError('Failed to move trace file to destination.')

        with open(path, 'r') as f:
            trace = AddressTrace([l.split(' ')[1] for l in f.readlines()])

        return trace

class TraceGenerator(object):

    def __init__(self, hough_lines, length, window_size, hm_height, block_size=None):
        assert window_size > 0
        assert length > 0
        assert hm_height > 0
        self.hough_lines = hough_lines
        self.window_size = window_size
        self.hm_height = hm_height
        self.block_size = block_size
        self.length = length
        self.x = 0

    def __traceGenRandomSample(self, x, hough_lines):
        tot_weight = sum([l.weight for l in hough_lines])
        return (np.random.choice(
            [l.getAddress(x) for l in hough_lines],
            p = [l.weight/tot_weight for l in hough_lines],
            size = self.window_size)).tolist()

    def __traceGenBB(self, x, hough_lines):
        assert self.block_size > 0, 'invalid block size for addr gen: {}'.format(self.block_size)
        tot_weight = sum([l.weight for l in hough_lines])
        cans = np.random.choice(range(len(hough_lines)),
                p = [l.weight/tot_weight for l in hough_lines],
                size = int(np.ceil(1. * self.window_size / self.block_size)))
        list_addrs = []
        # List concatenation is much faster than numpy concat.
        for can in cans:
            list_addrs.extend([hough_lines[can].getAddress(x)] * self.block_size)
        return list_addrs[:self.window_size]

    def __traceGenRR(self, x, hough_lines):
        # Round-robin between hough lines.
        rr = deque()
        for l in hough_lines:
            rr.append(l)
        # How many addresses to generate for each hough line based on weight.
        ration = deque()
        tot_weight = sum([l.weight for l in hough_lines])
        for l in hough_lines:
            ration.append(int(self.window_size * l.weight / tot_weight))
        count = 0
        list_of_addrs = []
        while(count < window_size):
            assert ration
            ra = ration.popleft()
            assert ra > 0 
            assert rr
            l = rr.popleft()
            list_of_addrs.extend([l.getAddress(x)]*ra)
            ration.append(ra)
            rr.append(l)
            count += ra
        return list_of_addrs[:self.window_size]

    def __background_noise(self, lower, upper):
        addr = np.random.choice(np.arange(lower, upper), size=self.window_size)
        list_addr = [hex(int(a) << kBlkBits) for a in addr]
        assert len(list_addr) == self.window_size 
        return list_addr

    def generateTrace(self, count):
        """ Takes multiple hough_lines, and generates an addressTrace.

            Args:
                hough_lines: a list of AddressTrace objects
                weights: a list of weights of len(hough_addresses)
                tot_length: total number of windows to generate addresses for
                window_size: how many addresses does each pixel in a hough line represent
                hm_height: size of the heatmap
                block_size: size for block-based address gen

            Returns:
                trace: a single AddressTrace object
        """
        counter = count
        list_addr = []
        while counter > 0:
            lines_in_range = [l for l in self.hough_lines if l.inRange(self.x)]
            if not lines_in_range:
                addr_per_window = self.__background_noise(0, self.hm_height)
            else:
                #addr_per_window = self.__traceGenRandomSample(self.x, lines_in_range)
                #addr_per_window = self.__traceGenRR(self.x, lines_in_range)
                addr_per_window = self.__traceGenBB(self.x, lines_in_range)
            list_addr.extend(addr_per_window)
            counter = counter - 1
            self.x = (self.x + 1) % self.length
        assert len(list_addr) == self.window_size * count
        return AddressTrace(list_addr)

    def minimize(self, ground_truth_heatmap, approximate_heatmap):
        """ Computes the difference between heatmaps and minimize/optimize the difference.

            Args:
                ground_truth_heatmap: a Heatmap object.
                approximate_heatmap: a Heatmap object.

            Returns:
                optimal_heatmap: a Heatmap object.
        """
        diff = ground_truth_heatmap - approximate_heatmap
        #TODO: optimize diff, by adjusting weights
        raise NotImplementedError
