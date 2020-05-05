import torch
import numpy as np
import pickle
import sys
import time
from random import randint
from net import TraceGen
name = "gcc-1B"
dev = ""
if torch.cuda.is_available():
    dev="cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
def pad(encoded,maxlen=50):
    if len(encoded) > maxlen:
        return encoded[len(encoded)-maxlen:]
    elif len(encoded):
        padded = [0 for i in range(maxlen-len(encoded))]
        return padded+encoded
    return encoded
# generate a sequence from a language model
def guess(model, lib, seq_length, seed_text):
    out_word = ""
    in_text = seed_text
    # generate a fixed number of words
    with torch.no_grad():
        # encode the text as integer
        encoded = [lib[w] for w in in_text]
        #print(encoded)
        # truncate sequences to a fixed length
        
        encoded = [pad(encoded,maxlen=seq_length)]
        # predict probabilities for each word
        yhat = model(torch.tensor(encoded,dtype=torch.long).to(device))
        #print(yhat)
        yhat = torch.softmax(yhat,dim=1)
        yhat = torch.max(yhat,dim=1)[1]

        for word, index in lib.items():
            if index == yhat:
                out_word = word
                break
    return out_word

in_filename = name+'_test.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = 50
lib = pickle.load(open('lib.pkl','rb'))
words = pickle.load(open('words.pkl','rb'))
PATH = './torch_model.pth'
# words = pickle.load(open('words.pkl'.'rb'))
# exit(1)
# load the model
model = TraceGen(len(lib),50,100)
model.load_state_dict(torch.load(PATH))
model = model.cuda()

seed_text = lines[:51]
correct = 0
total = 0
s = time.time()
guess(model, lib, seq_length, seed_text)
print(time.time()-s)
for line in lines[51:]:
    g = guess(model, lib, seq_length, seed_text)
    if line == g:
        correct+=1
    total+=1
    seed_text = seed_text[1:]
    seed_text.append(line)
    if total%500 == 0:
        print(correct/total)
        # print(line)
        # print(g)

print(correct/total)
