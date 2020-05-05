import numpy as np
import time
import pickle
lib = pickle.load(open('lib_markov.pkl','rb'))
lib_ind = {}
l_count = 0
start = time.time()
print(lib)
def compute():
    global l_count
    global start
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

compute()
dict_file = open('lib2.pkl',"wb")
pickle.dump(lib_ind,dict_file)
dict_file.close()
print(lib_ind)
                    