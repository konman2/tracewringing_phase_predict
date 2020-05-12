import torch
import numpy as np
import pickle
import sys
import time
from random import randint
from net import TraceGen
from torch.utils.data import DataLoader
from torch import nn
name = "gzip"
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
lib = pickle.load(open('lib.pkl','rb'))
words = pickle.load(open('words.pkl','rb'))
PATH = './models/epoch_.pth'
def split_X_y(lines):
    sequences = []
    for line in lines:
        l = []
        for word in line.split(' '):
            l.append(lib[word])
        sequences.append(l)
    sequences = np.array(sequences)
    return sequences[:,:-1],sequences[:,-1]
in_filename = name+'_seq.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
X_test,y_test = split_X_y(lines)
seq_length = 50

# words = pickle.load(open('words.pkl'.'rb'))
# exit(1)
# load the model



correct = 0
total = 0
guess_one = 0
guess_zero=0
running_loss = 0
y_test = torch.tensor(y_test,dtype=torch.float).reshape(y_test.shape[0],1)
#print(y)
test_data = []
batch_size = 64
for i in range(X_test.shape[0]):
    test_data.append([X_test[i],y_test[i]])
print(X_test)
for i in range(30):
    PATH = "./models/epoch_"+str(i)+".pth"
    model = TraceGen(2,1,100)
    model.load_state_dict(torch.load(PATH))
    model = model.to(device)
    testloader = DataLoader(test_data,batch_size=batch_size)
    loss_function = nn.BCEWithLogitsLoss().to(device)
    with torch.no_grad():
        for x,y in testloader:
            x = x.to(device)
            y = y.to(device)
            x= x.float()
            x=x.reshape(x.shape[0],x.shape[1],1)
            outputs = model(x)
            yhat = torch.sigmoid(outputs)
            yhat = yhat.round()
            #print(yhat)
            correct += (y==yhat).float().sum().item()
            guess_one += (y.new_ones(y.shape)==y).float().sum().item()
            guess_zero += (y.new_zeros(y.shape)==y).float().sum().item()
            total+=batch_size
            loss = loss_function(outputs,y)
            running_loss+=loss.item()
    print('model: %d,loss: %.3f accuracy: %.5f,guess Ones acc: %.3f,guess Zeros acc: %.3f' % (i,running_loss,correct/total,guess_one/total,guess_zero/total))
    