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
name = "gcc-1B"
in_filename = name+'_seq.txt'
test_filename = name+"_test.txt"
val_filename = name+"_val.txt"
doc = load_doc(in_filename)
lines = doc.split('\n')
val_doc = load_doc(val_filename)
val_lines = val_doc.split("\n")
test_doc = load_doc(test_filename)
test_lines = test_doc.split("\n")
lib = {}
unique_word_count = 1
words = [""]
lib[""] = 0
m = 0
def one_hot(a):
    y = []
    global lib
    for w in a:
        arr = np.zeros(unique_word_count)
        arr[w]=1
        y.append(arr)
    y = np.array(y)
    return y
def add_to_lib(lines):
    global lib
    global words
    global unique_word_count
    global m
    for line in lines:
        for word in line.split(' '):
            #word = int(word,0)
            if word not in lib:
                lib[word] = unique_word_count
                words.append(word)
                unique_word_count+=1
add_to_lib(lines)
l = unique_word_count
add_to_lib(val_lines)
l2 = unique_word_count
add_to_lib(test_lines)
print(l2-l,unique_word_count-l2)
print(unique_word_count)

def split_X_y(lines):
    sequences = []
    for line in lines:
        l = []
        for word in line.split(' '):
            #word = int(word,0)
            l.append(lib[word])
        sequences.append(l)
    sequences = np.array(sequences)
    return sequences[:,:-1],sequences[:,-1]

X,y = split_X_y(lines)
#y = np.array([lib[i] for i in y])
# X=X/np.max(X)
X_val,y_val = split_X_y(val_lines)
#y_val = np.array([lib[i] for i in y_val])
# X=X_val/np.max(X_val)
print(X)
model = TraceGen(unique_word_count,50,100)
print(model)
model.cuda()
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
train_data = []
batch_size = 64
val_data = []
for i in range(X.shape[0]):
    train_data.append([X[i],y[i]])
for i in range(X_val.shape[0]):
    val_data.append([X_val[i],y_val[i]])
trainloader = DataLoader(train_data,batch_size=batch_size)
val_loader = DataLoader(val_data,batch_size=batch_size)
print(unique_word_count)
#print(X.shape)
val_curve = []
PATH = './torch_model.pth'
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters())
val_loss = 0.0
correct = 0.0
total = 0.0

for epoch in range(100):
    start = time.time()
    running_loss = 0.0
    for x_batch,y_batch in trainloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        #x_batch= x_batch.float()
        #print(x_batch)
        #print(x_batch.dtype)
        #x_batch=x_batch.reshape(x_batch.shape[0],x_batch.shape[1],1)
        optimizer.zero_grad()
        outputs = model(x_batch)
        yhat = torch.softmax(outputs,dim=1)
        yhat = torch.max(yhat,dim=1)[1]
        
        loss = loss_function(outputs,y_batch)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    
    val_loss = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device)
            y = y.to(device)
            #x = x.float()
            #x=x.reshape(x.shape[0],x.shape[1],1)
            outputs = model(x)
            yhat = torch.softmax(outputs,dim=1)
            yhat = torch.max(yhat,dim=1)[1]
            # print(yhat.shape)
            #print(y.shape)
            correct += (y==yhat).float().sum()
            words = np.array(words)
            print(words[y]-words[yhat])
            total+=batch_size
            loss = loss_function(outputs,y)
            val_loss+=loss.item()

    end =  time.time()-start
    val_curve.append(val_loss)
    val_file = open("val.pkl","wb")
    pickle.dump(val_curve,val_file)
    word_file.close()
    print('[%d,%5d] loss: %.3f validation_loss:%d validation accuracy: %.5f' % (epoch+1,100,running_loss,val_loss,correct/total),"time=",end)
    if epoch%10 == 0 and epoch>0:
        torch.save(model.state_dict(),"./models/epoch_%d.pth" %(epoch+1))
    
print("done training")


torch.save(model.state_dict(), PATH)

