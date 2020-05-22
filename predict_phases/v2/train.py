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
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


def split_X_y(lines):
    sequences = []
    for line in lines:
        l = []
        for word in line.split(' '):
            l.append(int(word))
        sequences.append(l)
    sequences = np.array(sequences)
    return sequences[:,:-1],sequences[:,-1]
    
params = [k,k,200]
def train(val_name,params,log_file=None):
    in_filename = 'group_seq.txt'
    #val_filename = name+"_val.txt"
    val_filename = val_name+"_seq.txt"
    doc = load_doc(in_filename)
    lines = doc.split('\n')
    val_doc = load_doc(val_filename)
    val_lines = val_doc.split("\n")
    f = None
    path ="./models/"+val_name+"/"
    if not os.path.exists(path):
                os.makedirs(path)
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if log_file != None:
        f = open("./logs/"+log_file+".log",'w')
    map_new_standard = np.array(pickle.load(open("map_new_standard.pkl",'rb')))
    map_standard_new = np.array(pickle.load(open("map_standard_new.pkl",'rb')))
    X,y = split_X_y(lines)
    X_val,y_val = split_X_y(val_lines)
    X_val = np.array([map_new_standard[a] for a in X_val],dtype='int64')
    model = TraceGen(params[0],params[1],params[2])
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
    batch_size = 32
    val_data = []
    for i in range(X.shape[0]):
        train_data.append([X[i],y[i]])
    for i in range(X_val.shape[0]):
        val_data.append([X_val[i],y_val[i]])
    
    trainloader = DataLoader(train_data,batch_size=batch_size)
    val_loader = DataLoader(val_data,batch_size=batch_size)

    #print(unique_word_count)
    #print(X.shape)
    val_curve = []
    PATH = './torch_model.pth'
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    map_standard_new = torch.from_numpy(map_standard_new).to(device)
    map_new_standard = torch.from_numpy(map_new_standard).to(device)
    val_loss = 0.0
    correct = 0.0
    total = 0.0
    print("number of samples:",len(X))
    for epoch in range(10):
        model.train()
        start = time.time()
        running_loss = 0.0
        train_correct = 0.0
        train_total= 0.0
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
            for x,y in val_loader:
                x = x.to(device)
                y = y.to(device)
                # x= x.float()
                # x=x.reshape(x.shape[0],x.shape[1],1)
                outputs = model(x)
                yhat = torch.softmax(outputs,dim=1)
                yhat = torch.max(yhat,dim=1)[1]
                #rint(yhat)
                yhat = map_standard_new[yhat]
                #print(y)
                # print(torch.mode(x)[0])
                # print(yhat)
                #print(yhat)
                #print(y.shape,yhat.shape,map_standard_new[torch.mode(x)[0].reshape(y.shape)].shape)
                guess_mode += (map_standard_new[torch.mode(x)[0]] == y).sum().item()
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
            
        torch.save(model.state_dict(),path+"/epoch_%d_%.3f.pth" %(epoch,correct/total))    
        
    print("done training")


    torch.save(model.state_dict(),path+"/epoch_%d_%.3f.pth" %(epoch,correct/total))  
    f.close()
if __name__ == "__main__":
    train("mkdir",params)
    # name = "group"
    # in_filename = name+'_seq.txt'
    # #val_filename = name+"_val.txt"
    # val_filename = "mkdir_seq.txt"
    # doc = load_doc(in_filename)
    # lines = doc.split('\n')
    # val_doc = load_doc(val_filename)
    # val_lines = val_doc.split("\n")


    # map_new_standard = np.array(pickle.load(open("map_new_standard.pkl",'rb')))
    # map_standard_new = np.array(pickle.load(open("map_standard_new.pkl",'rb')))
    # X,y = split_X_y(lines)
    # X_val,y_val = split_X_y(val_lines)
    # X_val = np.array([map_new_standard[a] for a in X_val],dtype='int64')
    # model = TraceGen(params[0],params[1],params[2])
    # print("sequence length:",len(X[0]))
    # print(model)
    # model.to(device)
    # # dict_file = open('lib.pkl',"wb")
    # # pickle.dump(lib,dict_file)
    # # dict_file.close()
    # # word_file = open("words.pkl","wb")
    # # pickle.dump(words,word_file)
    # # word_file.close()
    # # X = torch.from_numpy(X)
    # y = torch.tensor(y,dtype=torch.long)
    # y_val = torch.tensor(y_val,dtype=torch.long)
    # print(y)
    # #print(y)
    # train_data = []
    # batch_size = 32
    # val_data = []
    # for i in range(X.shape[0]):
    #     train_data.append([X[i],y[i]])
    # for i in range(X_val.shape[0]):
    #     val_data.append([X_val[i],y_val[i]])
    
    # trainloader = DataLoader(train_data,batch_size=batch_size)
    # val_loader = DataLoader(val_data,batch_size=batch_size)

    # #print(unique_word_count)
    # #print(X.shape)
    # val_curve = []
    # PATH = './torch_model.pth'
    # loss_function = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.Adam(model.parameters())
    # map_standard_new = torch.from_numpy(map_standard_new).to(device)
    # map_new_standard = torch.from_numpy(map_new_standard).to(device)
    # val_loss = 0.0
    # correct = 0.0
    # total = 0.0
    # guess_one = 0.0
    # guess_zero=0.0
    # print("number of samples:",len(X))
    # for epoch in range(10):
    #     model.train()
    #     start = time.time()
    #     running_loss = 0.0
    #     train_correct = 0.0
    #     train_total= 0.0
    #     for x_batch,y_batch in trainloader:
    #         x_batch = x_batch.to(device)
    #         y_batch = y_batch.to(device)
    #         #print(y_batch)
    #         #x_batch= x_batch.float()
    #         #x_batch=x_batch.reshape(x_batch.shape[0],x_batch.shape[1],1)
    #         #print(x_batch.shape)
    #         #print(x_batch.shape)
    #         optimizer.zero_grad()
    #         outputs = model(x_batch)
    #         yhat = torch.softmax(outputs,dim=1)
    #         yhat = torch.max(yhat,dim=1)[1]
    #         train_correct += (y_batch==yhat).float().sum().item()
    #         train_total += len(y_batch)
    #         #print(outputs.shape)
    #         loss = loss_function(outputs,y_batch)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss+=loss.item()

    #     model.eval()
    #     val_loss = 0.0
    #     correct = 0.0
    #     total = 0.0
    #     guess_mode = 0.0 
    #     with torch.no_grad():
    #         for x,y in val_loader:
    #             x = x.to(device)
    #             y = y.to(device)
    #             # x= x.float()
    #             # x=x.reshape(x.shape[0],x.shape[1],1)
    #             outputs = model(x)
    #             yhat = torch.softmax(outputs,dim=1)
    #             yhat = torch.max(yhat,dim=1)[1]
    #             #rint(yhat)
    #             yhat = map_standard_new[yhat]
    #             #print(y)
    #             # print(torch.mode(x)[0])
    #             # print(yhat)
    #             #print(yhat)
    #             #print(y.shape,yhat.shape,map_standard_new[torch.mode(x)[0].reshape(y.shape)].shape)
    #             guess_mode += (map_standard_new[torch.mode(x)[0]] == y).sum().item()
    #             correct += (y==yhat).float().sum().item()
    #             total+=len(y)
    #             loss = loss_function(outputs,y)
    #             val_loss+=loss.item()

    #     end =  time.time()-start
    #     val_curve.append(val_loss)
    #     val_file = open("val.pkl","wb")
    #     pickle.dump(val_curve,val_file)
    #     print('[%d,%5d] loss: %.3f accuracy:%.3f validation_loss: %.3f validation accuracy: %.5f,guess mode acc: %.3f' % (epoch+1,100,running_loss,train_correct/train_total,val_loss,correct/total,guess_mode/total),"time=",end)
        
    #     #torch.save(model.state_dict(),"./models/epoch_%d.pth" %(epoch))    
        
    # print("done training")


    # #torch.save(model.state_dict(),"./models/epoch_%d.pth" %(epoch))