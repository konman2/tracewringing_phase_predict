import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import string
import pickle
import time
from model import TraceGen
from heatmap_gen import make_heatmap,names,name2,gen,HEIGHT,WINDOW_SIZE
import sys
import os
from sklearn.metrics import average_precision_score
from tracewringing.heatmap_generator import HeatmapGenerator
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean

dev = ""
if torch.cuda.is_available():
    dev="cuda:0"
else:
    dev = "cpu"

print(dev)
device = torch.device(dev)
#print(dev)

def train(val_name,params,log_file=None):
    train_data,val_data = gen(names,name2,save=False,viz=False)
    #val_filename = name+"_val.txt"

    f = None
    path ="./models/"+val_name+"/"
    if not os.path.exists(path):
                os.makedirs(path)
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if log_file != None:
        f = open("./logs/"+log_file+".log",'a+')
        f.write("TRAINING\n")
        
    # X,y = split_X_y(lines)
    # X_val,y_val = split_X_y(val_lines)
    # print(X2/np.amax(X2))
    # exit()
    model = TraceGen(params[0],params[1],params[2],embed=params[3])
    #model = model.float()
    print("sequence length:",len(train_data[0][0]))
    print(model)
    model.to(device)
    # y = torch.tensor(y,dtype=torch.long).to(device)
    # y_val = torch.tensor(y_val,dtype=torch.long)
    #print(y)
    #print(y)
    batch_size = 32
   

    trainloader = DataLoader(train_data,batch_size=batch_size)
    val_loader = DataLoader(val_data,batch_size=batch_size)
    
    #print(unique_word_count)
    #print(X.shape)
    val_curve = []
    PATH = './torch_model.pth'
    loss_function = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters())
    
    epochs = 10
    # X_val = torch.from_numpy(X_val).to(device)
    # y_val = y_val.to(device)
    running_loss = 0.0
    total_count = 0.0 
    #random_noise_score = 0.0
    # print(val_data[0][0])
    initial_data = [ i for i in val_data[0][0]]
    # print(type(initial_data[0]))
    # print(len(initial_data))
    best_map =None
    ans = None
    for epoch in range(epochs):
        model.train()
        start = time.time()
        running_loss = 0.0
        train_correct = 0.0
        train_total= 0.0
        for x_batch,y_batch in trainloader:
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            #print(x_batch.shape,x_batch.dtype)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_function(outputs,y_batch)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            total_count+=len(y_batch)

        model.eval()
        predictions = [i for i in initial_data]
        answers = [i for i in initial_data]
        total = 0.0
        val_loss = 0.0
        best = None
        # top5 = torch.tensor([29,97,118,7,99]).to(device)
        with torch.no_grad():
            for x,y in val_loader:
                x = x.to(device).float()
                y = y.to(device).float()
                # x= x.float()
                # x=x.reshape(x.shape[0],x.shape[1],1)
                # print(x.shape)
                outputs = model(x)
                # print(outputs.shape)
                # print(y.shape)
                loss = loss_function(outputs,y)
                val_loss += loss.item()
                a = [] 
                for c,i in enumerate(y):
                    answers.append(i.cpu().numpy())
                    #a.append(i.cpu().numpy())
                    predictions.append(outputs[c].cpu().numpy())
                # for i in answers[:82]:
                #     print(i)
                #print(answers)
                total+=len(y)
                #loss = loss_function(outputs,y)
                val_loss+=loss.item()
        end =  time.time()-start
        # running_loss
        # val_loss
        #heatmap_rand = np.random.rand(heatmap1.shape[0],heatmap1.shape[1])
        # mdist = np.mean(np.linalg.norm(heatmap1-heatmap2,axis=1))
        hm_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/heatmap_only/heatmaps/{}'.format(val_name+"-"+str(WINDOW_SIZE))
        results ='[%d,%5d] loss: %.5f , val_loss: %0.5f'  % (epoch+1,epochs,running_loss,val_loss)+" time= "+str(end)
        print(results)
        if best == None or val_loss<best:
            best_map = [i for i in predictions]
            ans = [i for i in answers]
            best = val_loss
        if log_file != None:
            f.write(results+"\n")
        heatmap1 = np.array(answers)
        heatmap2 = np.array(predictions)
        # print(heatmap1.T.shape,heatmap2.T.shape)
        # print(len(answers))
        # print(len(predictions))

        # print(pred_one)
        # print(len(pred_one))
        hg = HeatmapGenerator()
        #hg.getHeatmapFigure(heatmap.T,val_name+'_epoch_'+str(epoch),save=False,path='./train_figs/'+val_name)
        torch.save(model.state_dict(),path+"/epoch_%d_%.3f.pth" %(epoch,val_loss))    
        hg.compareHeatmaps((heatmap1.T,heatmap2.T),val_name,titles=('original','generated'),save=log_file!=None,path='figs/train/'+val_name+'/norm',four=False)
    print("done training")
    torch.save(model.state_dict(),path+"/epoch_%d_%.3f.pth" %(epoch,val_loss))  
    if f != None:
        f.close()
    hg = HeatmapGenerator()
    heatmap1 = np.array(ans)
    heatmap2 = np.array(best_map)
    hg.compareHeatmaps((heatmap1.T,heatmap2.T),val_name,titles=('original','generated'),save=log_file!=None,path='figs/train/'+val_name+'/norm',four=False)
    #hg.compareHeatmaps((orig_heatmap.T,cluster_map.T,best_map.T,mode_map.T),val_name,titles=('original','perfect lstm','lstm','modes'),save=log_file!=None,path='figs/train/'+val_name+'/norm',four=True)
    # hg.compareHeatmaps((np.sqrt(np.sqrt(orig_heatmap.T)),np.sqrt(np.sqrt(cluster_map.T)),np.sqrt(np.sqrt(best_map.T)),np.sqrt(np.sqrt(mode_map.T))),val_name,titles=('original','perfect lstm','lstm','modes'),save=log_file!=None,path='figs/train/'+val_name)
params = [HEIGHT,HEIGHT,300,False]
if __name__ == "__main__":
    train(name2,params)