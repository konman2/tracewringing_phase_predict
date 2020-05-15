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
name = sys.argv[1]
dev = "cpu"
if torch.cuda.is_available():
    dev="cuda:0"
device = torch.device(dev)
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
    result = first_thirty
    x = first_thirty
    # print(device)
    # print(x,len(x))
    for _ in range(n_words):
        #print(type(x))
        #print(x)
        x_in = torch.tensor(x).reshape(1,seq_length).to(device)
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
PATH = './models/epoch_2.pth'

in_filename = "phases/"+name+'.phase'
doc = load_doc(in_filename)
lines = doc.split('\n')
print(lines)
first_seq = [int(i) for i in lines[:seq_len]]
model = TraceGen(k,k,50)
model.load_state_dict(torch.load(PATH))
model.to(device)

print(first_seq)
result = generate_seq(model,seq_len,first_seq,len(lines)-seq_len)
print(result)
print(len(result))