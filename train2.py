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
import os
dev = ""
if torch.cuda.is_available():
    dev="cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
#print(dev)
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


def split_X_y(lines,m=1):
    sequences = []
    for line in lines:
        l = []
        for word in line.split(' '):
            l.append(int(word))
        sequences.append(l)
    sequences = np.array(sequences)
    return sequences[:,:-1],sequences[:,-1]
    

def train(val_name,params,log_file=None):
    in_filename = 'sequences/group_seq.txt'
    #val_filename = name+"_val.txt"
    val_filename = "sequences/"+val_name+"_seq.txt"
    doc = load_doc(in_filename)
    lines = doc.split('\n')
    lines2 = load_doc('sequences/grouplens_seq.txt').split('\n')
    val_doc = load_doc(val_filename)
    val_lines = val_doc.split("\n")
    val_lines2 = load_doc('sequences/'+val_name+'lens_seq.txt').split('\n')

    f = None
    path ="./models/"+val_name+"/"
    if not os.path.exists(path):
                os.makedirs(path)
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if log_file != None:
        f = open("./logs/"+log_file+".log",'w')
    X,y = split_X_y(lines)
    X2,y2 = split_X_y(lines2)
    X_val,y_val = split_X_y(val_lines)
    X_val2,y_val2 = split_X_y(val_lines2)
    #X_val = np.array([map_new_standard[a] for a in X_val],dtype='int64')
    # X2 = X2/np.amax(X2)
    # y2 = y2/np.amax(X2)
    # X_val2 = X_val2/np.amax(X2)
    # y_val2=y_val2/np.amax(X2)
    # print(X2/np.amax(X2))
    # exit()
    model = TraceGen(params[0],params[1],params[2],embed=params[3])
    print("sequence length:",len(X[0]))
    print(model)
    model.to(device)
    # dict_file = open('lib.pkl',"wb")
    # pickle.dump(lib,dict_file)
    # dict_file.close()
    # word_file = open("words.pkl","wb")
    # pickle.dump(words,word_file)
    # word_file.close()
    # X = torch.from_numpy(X)
    y = torch.tensor(y,dtype=torch.long)
    y_val = torch.tensor(y_val,dtype=torch.long)
    print(y)
    #print(y)
    train_data = []
    train_data2 = []
    batch_size = 32
    val_data = []
    val_data2 = []
    for i in range(X.shape[0]):
        train_data.append([X[i],y[i]])
        train_data2.append([X2[i],y2[i]])
    for i in range(X_val.shape[0]):
        val_data.append([X_val[i],y_val[i]])
        val_data2.append([X_val2[i],y_val2[i]])

    # train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data),torch.tensor(train_data2))
    # val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_data),torch.tensor(val_data2))

    trainloader = DataLoader(train_data,batch_size=batch_size)
    trainloader2 = DataLoader(train_data2,batch_size=batch_size)
    val_loader = DataLoader(val_data,batch_size=batch_size)
    val_loader2 = DataLoader(val_data2,batch_size=batch_size)

    #print(unique_word_count)
    #print(X.shape)
    val_curve = []
    PATH = './torch_model.pth'
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    val_loss = 0.0
    correct = 0.0
    total = 0.0
    print("number of samples:",len(X))
    for epoch in range(20):
        model.train()
        start = time.time()
        running_loss = 0.0
        train_correct = 0.0
        train_total= 0.0
        for a,b in zip(trainloader,trainloader2):
            x_batch = a[0]
            y_batch = a[1]
            x2_batch = b[0]
            y2_batch = b[1]
            x_batch = x_batch.to(device).long()
            y_batch = y_batch.to(device).long()
            x2_batch = x2_batch.to(device)
            y2_batch = y2_batch.to(device)
            #print(y_batch)
            #x_batch= x_batch.float()
            #x_batch=x_batch.reshape(x_batch.shape[0],x_batch.shape[1],1)
            #print(x_batch.shape)
            #print(x_batch.shape)
            optimizer.zero_grad()
            outputs = model(x_batch,x2_batch)
            yhat = torch.softmax(outputs,dim=1)
            yhat = torch.max(yhat,dim=1)[1]
            train_correct += (y_batch==yhat).float().sum().item()
            train_total += len(y_batch)
            #print(outputs.shape)
            loss = loss_function(outputs,y_batch)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0.0
        total = 0.0
        guess_mode = 0.0 
        with torch.no_grad():
            for a,b in zip(val_loader,val_loader2):
                x = a[0].to(device).long()
                y = a[1].to(device).long()
                x2 = b[0].to(device)
                y2 = b[1].to(device)
                # x= x.float()
                # x=x.reshape(x.shape[0],x.shape[1],1)
                outputs = model(x,x2)
                yhat = torch.softmax(outputs,dim=1)
                yhat = torch.max(yhat,dim=1)[1]
                #rint(yhat)
                yhat = yhat
                #print(y)
                # print(torch.mode(x)[0])
                # print(yhat)
                #print(yhat)
                #print(y.shape,yhat.shape,map_standard_new[torch.mode(x)[0].reshape(y.shape)].shape)
                guess_mode += (torch.mode(x)[0] == y).sum().item()
                correct += (y==yhat).float().sum().item()
                total+=len(y)
                loss = loss_function(outputs,y)
                val_loss+=loss.item()

        end =  time.time()-start
        val_curve.append(val_loss)
        val_file = open("val.pkl","wb")
        pickle.dump(val_curve,val_file)
        
        
        results ='[%d,%5d] loss: %.3f accuracy:%.3f validation_loss: %.3f validation accuracy: %.5f,guess mode acc: %.3f' % (epoch+1,100,running_loss,train_correct/train_total,val_loss,correct/total,guess_mode/total)+" time= "+str(end)
        
        print(results)
        if log_file != None:
            f.write(results+"\n")
            
        #torch.save(model.state_dict(),path+"/epoch_%d_%.3f.pth" %(epoch,correct/total))    
        
    print("done training")


    #torch.save(model.state_dict(),path+"/epoch_%d_%.3f.pth" %(epoch,correct/total))  
    if f != None:
        f.close()
params = [k,k,200,True]
if __name__ == "__main__":
    train("mkdir",params)