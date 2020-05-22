import torch
import numpy as np
import pickle
import sys
import time
from random import randint
from net import TraceGen
from torch.utils.data import DataLoader
from torch import nn
from gen import k
from clean import seq_len
from train import params
name = sys.argv[1]
dev = "cpu"
if torch.cuda.is_available():
    dev="cuda:0"
device = torch.device(dev)
map_new_standard = np.array(pickle.load(open("map_new_standard.pkl",'rb')))
map_standard_new = np.array(pickle.load(open("map_standard_new.pkl",'rb')))
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
def pad(encoded,maxlen=30):
    if len(encoded) > maxlen:
        return encoded[len(encoded)-maxlen:]
    elif len(encoded):
        padded = [0 for i in range(maxlen-len(encoded))]
        return padded+encoded
    return encoded

# generate a sequence from a language model
def generate_seq(model, seq_length, first_seq, n_words):
    result = first_seq
    x = first_seq
    # print(device)
    # print(x,len(x))
    for _ in range(n_words):
        #print(type(x))
        #print(x)
        x_in = torch.tensor(x,dtype=torch.long).reshape(1,seq_length).to(device)
        outputs = model(x_in)
        yhat = torch.softmax(outputs,dim=1)
        yhat = torch.max(yhat,dim=1)[1]
        yhat = yhat.item()
        #print(yhat)
        #print(x[1:])
        x =x[1:]
        x.append(yhat)
        #print(x)
        result.append(yhat)
    return result

lib = pickle.load(open('lib.pkl','rb'))
words = pickle.load(open('words.pkl','rb'))
PATH = './models/epoch_8.pth'

in_filename = "phases/"+name+'.phase'
doc = load_doc(in_filename)
lines = doc.split('\n')
print(lines)

first_seq = [map_new_standard[int(i)] for i in lines[:seq_len]]
model = TraceGen(params[0],params[1],params[2])
model.load_state_dict(torch.load(PATH))
model.to(device)
model.eval()

print(first_seq)
result = generate_seq(model,seq_len,first_seq,len(lines)-seq_len)
print()
print(result)
print(len(result))