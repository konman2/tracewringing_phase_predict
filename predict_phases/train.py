import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import string
import pickle
import time
from net import TraceGen
from gen import k
import sys
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

# load
name = sys.argv[1]
in_filename = name+'_seq.txt'
#val_filename = name+"_val.txt"
val_filename = "mkdir_seq.txt"
doc = load_doc(in_filename)
lines = doc.split('\n')
val_doc = load_doc(val_filename)
val_lines = val_doc.split("\n")
lib = {}
unique_word_count = 0
words = []

def add_to_lib(lines):
    global lib
    global words
    global unique_word_count
    for line in lines:    
        for word in line.split(' '):
            if word not in lib and word != "0" and word!= "1":
                lib[word] = unique_word_count
                words.append(word)
                unique_word_count+=1
           

# add_to_lib(lines)
# l = unique_word_count
# add_to_lib(val_lines)
# l2 = unique_word_count
# add_to_lib(test_lines)
# print(l2-l,unique_word_count-l2)
# print(unique_word_count)

for i in range(k):
    lib[str(i)] = i



def split_X_y(lines):
    sequences = []
    for line in lines:
        l = []
        for word in line.split(' '):
            l.append(lib[word])
        sequences.append(l)
    sequences = np.array(sequences)
    return sequences[:,:-1],sequences[:,-1]

X,y = split_X_y(lines)
X_val,y_val = split_X_y(val_lines)
unique_word_count = 4
model = TraceGen(k,5,50)
print("sequence length:",len(X[0]))
print(model)
model.to(device)
dict_file = open('lib.pkl',"wb")
pickle.dump(lib,dict_file)
dict_file.close()
word_file = open("words.pkl","wb")
pickle.dump(words,word_file)
word_file.close()
# X = torch.from_numpy(X)
y = torch.tensor(y,dtype=torch.long)
y_val = torch.tensor(y_val,dtype=torch.long)
print(y)
#print(y)
train_data = []
batch_size = 32
val_data = []
for i in range(X.shape[0]):
    train_data.append([X[i],y[i]])
for i in range(X_val.shape[0]):
    val_data.append([X_val[i],y_val[i]])
print(X_val)
trainloader = DataLoader(train_data,batch_size=batch_size)
val_loader = DataLoader(val_data,batch_size=batch_size)

#print(unique_word_count)
#print(X.shape)
val_curve = []
PATH = './torch_model.pth'
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters())

val_loss = 0.0
correct = 0.0
total = 0.0
guess_one = 0.0
guess_zero=0.0

for epoch in range(10):
    start = time.time()
    running_loss = 0.0
    for x_batch,y_batch in trainloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        #print(y_batch)
        #x_batch= x_batch.float()
        #x_batch=x_batch.reshape(x_batch.shape[0],x_batch.shape[1],1)
        #print(x_batch.shape)
        #print(x_batch.shape)
        optimizer.zero_grad()
        outputs = model(x_batch)
        #print(outputs.shape)
        loss = loss_function(outputs,y_batch)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    
    val_loss = 0.0
    correct = 0.0
    total = 0.0
    guess_one = 0.0
    guess_zero=0.0
    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device)
            y = y.to(device)
            # x= x.float()
            # x=x.reshape(x.shape[0],x.shape[1],1)
            outputs = model(x)
            yhat = torch.softmax(outputs,dim=1)
            yhat = torch.max(yhat,dim=1)[1]
            #print(yhat)
            correct += (y==yhat).float().sum().item()
            
            guess_one += (y.new_ones(y.shape)*4==y).float().sum().item()
            guess_zero += (y.new_zeros(y.shape)==y).float().sum().item()
            total+=batch_size
            loss = loss_function(outputs,y)
            val_loss+=loss.item()

    end =  time.time()-start
    val_curve.append(val_loss)
    val_file = open("val.pkl","wb")
    pickle.dump(val_curve,val_file)
    word_file.close()
    print('[%d,%5d] loss: %.3f validation_loss: %.3f validation accuracy: %.5f,guess Ones acc: %.3f,guess Zeros acc: %.3f' % (epoch+1,100,running_loss,val_loss,correct/total,guess_one/total,guess_zero/total),"time=",end)
    
    torch.save(model.state_dict(),"./models/epoch_%d.pth" %(epoch))    
    
print("done training")


torch.save(model.state_dict(),"./models/epoch_%d.pth" %(epoch))