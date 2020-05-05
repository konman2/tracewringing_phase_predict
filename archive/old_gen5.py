# exec(open('gen5.py').read())

'''
steps:

'''

# dependencies
from itertools import islice
import numpy as np
import random
import pickle

# params
trace = 'gzipm16'
hl_loc = 'hough_lines/'+trace+'.npy'
dataloc = 'spec_trace/gzip_m2b.1.D.txt'
hmloc = 'cache_1_' + trace + '_heatmap.dict'
test_hmloc = 'cache_1_test.dict'
N = 100000

# global params
icount=0

# helper funcs
def make_hough(hl_loc, image):
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1,1)
	lines = np.load(hl_loc) 
	ax.imshow(image * 0)
	for line in lines:
		p0, p1 = line
		ax.plot((p0[0], p1[0]), (p0[1], p1[1]))
	ax.set_xlim((0, image.shape[1]))
	ax.set_ylim((image.shape[0], 0))
	ax.set_title('Probabilistic Hough')
	plt.tight_layout()
	plt.show()

def get_info(pt, m=16):
	y0 = 256 * m - pt[1][1]
	y1 = 256 * m - pt[0][1]
	x0 = pt[1][0]
	x1 = pt[0][0]
	#rise = y1 - y0
	#run = x1 - x0
	rise = -pt[1][1] + pt[0][1]
	run = -pt[1][0] + pt[0][0]
	slope = (rise/run)
	if rise == 0:
		run = abs(run)
		slope = abs(rise/run)
	#intercept = - slope * run / y1
	intercept = y0 - slope * x0 
	xstart = pt[0][0]
	return slope, intercept, xstart, run, rise

def heatmap(hmloc,tag='a',title='a'):
	''' k = np array
		tag = string to savefig '''
	# hmloc = 'cache_1_test.dict'
	import pickle
	hm = pickle.load(open(hmloc, 'rb'))
	cl = np.asarray([np.array(i) for i in hm.values()])
	import matplotlib.pyplot as plt
	from matplotlib import cm
	plt.figure(figsize=(20,15))
	plt.ylabel('Cachelines')
	plt.xlabel('1k Instructions')
	plt.title(title)
	plt.imshow(np.sqrt(np.sqrt(np.sqrt(cl))),cmap=cm.Greys,snap=True)
	#plt.savefig('figs/'+tag)#
    plt.show()	

class random_interleaved_agen():
    'address generator using fully random interleaving'
    def __init__(self, generator_list):
        self.generator_list = generator_list
    def __next__(self):
        gen = random.choice(self.generator_list)
        return next(gen)

class weighted_interleaved_gen():
    def __init__(self, generator_list, weight_list):
        self.generator_list = generator_list
        self.weight_list = weight_list
    def __next__(self):
        from numpy.random import choice
        gen = choice(self.generator_list, 1, self.weight_list)
        return next(gen[0])

def first_n(iterable, n):
    return islice(iterable, 0, n)

def gen_stride(slope, intercept, xstart, run):
		''' example: slope = 0.5, intercept = 200, xstart = 200, run = 300
		'''   
		cline = intercept*64 # address
		last = 0 # for tracking icount
		xstart = xstart * 1000 # icount
		run = run * 1000 # icount
		islope = slope * 64/1000.0 # address/icount
		while True:
			instr_since_last_call = icount - last
			last = icount
			cline += instr_since_last_call * islope
			if icount < xstart:
				yield None
			elif icount > (xstart + run):
				yield None
			else:
				print(slope, islope)
				yield int(cline)

def uniform_gen(max_addr):
    k = 0
    while True:
        print(k)
        k += 1
        yield random.randint(0,max_addr)

# workflow
hlines = np.load(hl_loc) # collect hough_lines info
info = [get_info(i) for i in hlines]
hm = pickle.load(open(hmloc, 'rb'))
cl = np.asarray([np.array(i) for i in hm.values()])
raise('')
#---
gen_list = [gen_stride(info[i][0], info[i][1], info[i][2], info[i][3]) for i in range(len(info))]
test_gen_list = gen_list #[gen_stride(slope=0.5,  intercept=0,  xstart=0,   run=500)]
#                 gen_stride(slope=0.25, intercept=45, xstart=180, run=180)]
#agen = weighted_interleaved_gen(test_gen_list, [0.45,0.45,0.1])
#agen = random_interleaved_agen(gen_list)

#gen_list = [gen_stride(slope=0.5,]
#gen_list = [gen_stride(info[i][0], info[i][1], info[i][2], info[i][3]) for i in range(len(info))]

icount = 1
i = 1
N = 100000
addr = []
uniform = uniform_gen(max_addr=16*256)
#test_gen_list = gen_list
for i in range(N):
    shuffled_gen_list = random.sample(test_gen_list, len(test_gen_list))
    for gen in shuffled_gen_list:
        k = next(gen)
        if k is not None:
            break
    if k is None:
        k = next(uniform)
    addr.append(hex(int(k)))
    icount += 1

# ----

#gen_list = [gen_stride(info[i][0], info[i][1], info[i][2], info[i][3]) for i in range(len(info))]
#gen_list.append(uniform_gen(16*256*N))
#test_gen_list = [gen_stride(0.5,0,0,500), gen_stride(0.25, 45, 180, 1000-180), uniform_gen(16*256)] 
#agen = weighted_interleaved_gen(test_gen_list, [0.45,0.45,0.1])
#agen = random_interleaved_agen(gen_list)

#addr = []
#k_arr = []
#while len(addr) < N:
	#icount = 0
#for i in range(N):
#icount = 1
#i = 1
#while True:
#	k = next(agen)
#	if k is not None:
#		addr.append(hex(int(k)))
#		icount += 1
#		i += 1
#	if i>N:
#		break

raise('')
np.savetxt('test.txt', addr, fmt='%s')

#for addr in first_n(g1,1000000):
#    print(addr)
#    icount += 1

#### DEBUGGING:
#make_hough(hl_loc, np.sqrt(cl[:,:2000]))
#test_gen_list = [gen_stride(slope=0.5,  intercept=0,  xstart=0,   run=500), 
#                 gen_stride(slope=0.25, intercept=45, xstart=180, run=180)]
agen = weighted_interleaved_gen(test_gen_list, [0.45,0.45,0.1])
#agen = random_interleaved_agen(gen_list)

icount = 1
i = 1
N = 100000
addr = []
uniform = uniform_gen(max_addr=16*256)

for i in range(N):
    shuffled_gen_list = random.sample(gen_list, len(gen_list))
    for gen in shuffled_gen_list:
        k = next(gen)
        if k is not None:
            break
    if k is None:
        k = next(uniform)
    addr.append(hex(int(k)))
    icount += 1
