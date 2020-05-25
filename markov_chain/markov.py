import numpy as np
import pickle
import time
import string

#f=open('/home/mkondapaneni/Research/LSTM_prac/training-monolingual/news.2011.en.shuffled')


lib = {}
print("running...")

def process(prev,word):
    global lib
    global uniq_count
    # print(lib)
    if word not in lib[prev]:
        lib[prev][word] = 1  
    else:
        lib[prev][word]+=1
    if word not in lib:
        lib[word] = {}
    return lib[word]

names = ["cat","cp","echo","findmnt","git","ls"]

m = pickle.load(open('../map_new_standard.pkl','rb'))
for name in names:
    start = time.time()
    count = 0
    epsilon = 0.000
    w_count = 0
    uniq_count = 0
    name = "../phases/"+name+".phase"
    times = np.array([])        
    l_count = 0
    f = open(name,'r')
    prev = ""
    first = True
    for line in f.readlines():
        if line == "\n":
            continue
        if (l_count)%10000 == 0 and l_count != 0:
            dict_file = open('lib_markov.pkl',"wb")
            pickle.dump(lib,dict_file)
            dict_file.close()
            print("line: ",l_count)
            print("unique words: ",uniq_count)
            print("tot words: ",w_count )
            print("ratio: ", uniq_count/w_count)
            print("time:", (time.time()-start)/60)
            times = np.append(times,time.time()-start)
            np.save('times.npy',times)
            print()
        #print(line,end=" ")
        word = int(line.strip())
        #print(word)
        # word = word.lower()
        # word = word.translate(str.maketrans('', '', string.punctuation))
        # print(word,first)
        if first:
            prev = word
            if prev not in lib:
                lib[prev] = {}
                uniq_count+=1
            first = False
        else:
            process(prev, word)
        w_count+=1
        prev = word
    l_count+=1

print(len(lib))
print(lib)
dict_file = open('lib_markov2.pkl',"wb")
pickle.dump(lib,dict_file)
dict_file.close()
end = time.time()
print(end - start)