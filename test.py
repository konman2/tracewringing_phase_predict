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

lib = pickle.load(open('lib.pkl','rb'))
words = pickle.load(open('words.pkl','rb'))
PATH = './models/epoch_6.pth'
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
seq_length = 30

model = TraceGen(k,k,100)
model.load_state_dict(torch.load(PATH))
model.to(device)
test_data = []
batch_size = 64
for i in range(X_test.shape[0]):
    test_data.append([X_test[i],y_test[i]])
print(X_test)

testloader = DataLoader(test_data,batch_size=batch_size)

loss_function = nn.CrossEntropyLoss().to(device)
val_loss = 0.0
correct = 0.0
total = 0.0
guess_one = 0.0
guess_zero=0.0
with torch.no_grad():
    start = time.time()
    for x,y in testloader:
        x = x.to(device)
        y = y.to(device)
        # x= x.float()
        # x=x.reshape(x.shape[0],x.shape[1],1)
        outputs = model(x)
        yhat = torch.softmax(outputs,dim=1)
        yhat = torch.max(yhat,dim=1)[1]
        #print(yhat)
        #print(yhat)
        correct += (y==yhat).float().sum().item()
        
        guess_one += (y.new_ones(y.shape)*3==y).float().sum().item()
        guess_zero += (y.new_zeros(y.shape)==y).float().sum().item()
        total+=batch_size
        loss = loss_function(outputs,y)
        val_loss+=loss.item()

    end =  time.time()-start
    print( 'loss: %.3f accuracy: %.5f,guess Ones acc: %.3f,guess Zeros acc: %.3f' % (val_loss,correct/total,guess_one/total,guess_zero/total),"time=",end)
    #print('model: %d,loss: %.3f accuracy: %.5f,guess Ones acc: %.3f,guess Zeros acc: %.3f' % (i,running_loss,correct/total,guess_one/total,guess_zero/total))
    