import numpy as np
import time
import pickle

def compute():
    lib = pickle.load(open('/home/mkondapaneni/Research/tracewringing_phase_predict/lib_markov2.pkl','rb'))
    lib_ind = {}
    l_count = 0
    start = time.time()
    print(lib)
    # global l_count
    # global start
    s = start
    for w,d in lib.items():
        if (l_count)%100 == 0:
            print("word: ",l_count)
            end = time.time()
            print("time:", (end-s)/60)
            print("time from beg:", (end-start)/60) 
            s = end 
            print()
        prob = []
        words = []
        # print(d)
        for key,val in d.items():
            words.append(key)
            prob.append(val)
        p = np.array(prob)
        if np.sum(p) != 0:
            p = p/np.sum(p)
            lib_ind[w] = (words,p)
        l_count+=1
        dict_file = open('/home/mkondapaneni/Research/tracewringing_phase_predict/lib2.pkl',"wb")
        pickle.dump(lib_ind,dict_file)
        dict_file.close()
        print(lib_ind)



                    