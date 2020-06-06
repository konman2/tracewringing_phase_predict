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
from gen import k,make_heatmap,name2
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
    
def val(model,X_val,y_val,top5,a):
    outputs = model(X_val)
    yhat = torch.softmax(outputs,dim=1)
    yhat = torch.topk(yhat,5,dim=1)[1]
    correct = 0.0
    guess_mode= 0.0
    # print(yhat.shape)
    # print(y_val.shape)
    a.extend(torch.flatten(yhat).tolist())
    # precision = average_precision_score()
    for c,i in enumerate(y_val):
        if i in yhat[c]:
            correct+=1
        if i in top5:
            guess_mode+=1
    total=len(y_val)
    return correct,guess_mode,total
def metric(heatmap1,heatmap2):
    assert heatmap1.shape == heatmap2.shape
    base = np.zeros(heatmap1.shape)
    total_both_zeros = np.sum(np.logical_and(heatmap1==base,heatmap2==base))
    total_points = heatmap1.shape[0]*heatmap1.shape[1]
    score = mse(heatmap1,heatmap2)*total_points
    #return np.sqrt(np.sqrt((np.linalg.norm(heatmap1.flatten()-heatmap2.flatten(),4)/total_points)))
    if total_both_zeros == total_points:
        return 0.0
    return score/(total_points-total_both_zeros)

def train(val_name,params,log_file=None):
    in_filename = 'sequences/group_seq.txt'
    #val_filename = name+"_val.txt"
    val_filename = "sequences/"+val_name+"_seq.txt"
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
        f = open("./logs/"+log_file+".log",'a+')
        f.write("TRAINING\n")
    X,y = split_X_y(lines)
    X_val,y_val = split_X_y(val_lines)
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
    y = torch.tensor(y,dtype=torch.long).to(device)
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
    values,counts = np.unique(y_val.flatten(),return_counts=True)
    top5 = torch.topk(torch.from_numpy(counts),5)[1]
    print(values[top5])
    top5 = torch.from_numpy(values[top5]).to(device)
    # counts = torch.zeros(k)[X_val.flatten()]+=1
    # top5 = torch.topk(counts,5)
    # print(top5)
    # train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data),torch.tensor(train_data2))
    # val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_data),torch.tensor(val_data2))

    trainloader = DataLoader(train_data,batch_size=batch_size)

    val_loader = DataLoader(val_data,batch_size=batch_size)
    phases = pickle.load(open('phases.pkl','rb'))
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
    epochs = 10
    X_val = torch.from_numpy(X_val).to(device)
    y_val = y_val.to(device)
    precision = 0.0
    recall = 0.0
    base_model = 0.0
    transition_p = 0.0
    best_score = None
    best_orig_score = 0.0
    perfect_score = 0.0
    mode_score = 0.0 
    all_background_score = 0.0
    cluster_map = None
    orig_heatmap = None
    best_map = None
    mode_map =  None
    #random_noise_score = 0.0

    for epoch in range(epochs):
        model.train()
        start = time.time()
        running_loss = 0.0
        train_correct = 0.0
        train_total= 0.0
        for x_batch,y_batch in trainloader:
            x_batch = x_batch.to(device).long()
            y_batch = y_batch.to(device).long()
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
        predictions = []
        answers = []
        transition_correct = 0.0
        transition_total = 0.0
        pred_one = X_val[0].tolist()
        pred_mode = X_val[0].tolist()
        # top5 = torch.tensor([29,97,118,7,99]).to(device)
        with torch.no_grad():
            for x,y in val_loader:
                x = x.to(device).long()
                y = y.to(device).long()
                # x= x.float()
                # x=x.reshape(x.shape[0],x.shape[1],1)
                outputs = model(x)
                yhat = torch.softmax(outputs,dim=1)
                #yhat = torch.max(yhat,dim=1)[1]
                # print(torch.topk(yhat,5,dim=1))
                # exit()
                yhat = torch.topk(yhat,5,dim=1)[1]
                #print(yhat)
                predictions.extend(torch.flatten(yhat).tolist())
                answers.extend(torch.flatten(y).tolist())
                for c,i in enumerate(y):
                    mode = torch.mode(x[c])[0]
                    values,counts = np.unique(x[c].cpu().numpy().flatten(),return_counts=True)
                    # print(values,counts)
                    #print(x[c])
                    if x[c][-1].item() != i:
                        transition_total+=1
                        if i in yhat[c]:
                            transition_correct+=1
                    if i in yhat[c]:
                        correct+=1
                    if i in top5:
                        guess_mode+=1
                    pred_one.append(yhat[c][0].item())
                    pred_mode.append(mode.item())
                    
                total+=len(y)
                #loss = loss_function(outputs,y)
                val_loss+=loss.item()

        end =  time.time()-start
        val_curve.append(val_loss)
        val_file = open("val.pkl","wb")
        pickle.dump(val_curve,val_file)
        predictions = [i for i in predictions if i in answers ]
        answers = pred_one[:50] + (answers)
        heatmap1 = make_heatmap(phases,answers)
        heatmap2 = make_heatmap(phases,pred_one)
        #heatmap_rand = np.random.rand(heatmap1.shape[0],heatmap1.shape[1])
        # mdist = np.mean(np.linalg.norm(heatmap1-heatmap2,axis=1))
        mdist = metric(heatmap1,heatmap2)
        hm_path = '/home/mkondapaneni/Research/tracewringing_phase_predict/heatmaps/{}'.format(val_name)
        heatmap_orig = np.sqrt(np.sqrt(pickle.load(open(hm_path,'rb')).T[:heatmap1.shape[0]]))
        mdist_orig_cluster = metric(heatmap_orig,heatmap1)
        mdist_orig_lstm = metric(heatmap_orig,heatmap2)
        results ='[%d,%5d] loss: %.3f accuracy:%.3f transition accuracy: %.3f precision: %.5f,guess mode acc: %.3f,recall: %0.3f, score: %0.5f, perfect score: %0.5f orig compare: %0.5f'  % (epoch+1,epochs,running_loss,train_correct/train_total,transition_correct/transition_total,correct/total,guess_mode/total,len(set(predictions))/len(set(answers)),mdist,mdist_orig_cluster,mdist_orig_lstm)+" time= "+str(end)
        print(results)
        heatmap_modes = make_heatmap(phases,pred_mode)
        print(metric(heatmap1,np.zeros(heatmap1.shape)),metric(heatmap_orig,heatmap_modes),metric(heatmap_orig,np.zeros(heatmap1.shape)))
        if best_score == None or mdist<best_score:
            precision = correct/total
            base_model = guess_mode/total
            recall = len(set(predictions))/len(set(answers))
            transition_p = transition_correct/transition_total
            best_score  = mdist
            best_orig_score = mdist_orig_lstm
            perfect_score = mdist_orig_cluster
            mode_score = metric(heatmap_orig,heatmap_modes)
            all_background_score = metric(heatmap_orig,np.zeros(heatmap1.shape))
            cluster_map = heatmap1
            orig_heatmap = heatmap_orig
            mode_map= heatmap_modes
            best_map = heatmap2
        if log_file != None:
            f.write(results+"\n")
        # print(pred_one)
        # print(len(pred_one))
        heatmap = make_heatmap(phases,pred_one)
        hg = HeatmapGenerator()
        #hg.getHeatmapFigure(heatmap.T,val_name+'_epoch_'+str(epoch),save=False,path='./train_figs/'+val_name)
        corr = 0
        #print(answers)
        diff = [] 
        for c,i in enumerate(pred_one):
            if i == answers[c]:
                corr+=1
            else:
                diff.append((answers[c],i))
        # print(answers[23:43])
        # print(pred_one[23:43])
        # print(corr/len(answers))
        # print(set(diff))
        # print(len(set(diff)))
        # heatmap1 = make_heatmap(phases,answers)
        # heatmap2 = make_heatmap(phases,pred_one)
        # hg.compareHeatmaps(heatmap1.T,heatmap_rand.T,val_name)
        # hg.compareHeatmaps(heatmap1.T,np.zeros(heatmap1.T.shape),val_name)
        #hg.compareHeatmaps((heatmap1.T,heatmap_modes.T),val_name+"mode")
        # for a,b in set(diff):
        #     print((a,b))
        #     print(phases[a].shape,phases[b].shape)
        #     h1 = phases[a][:,:phases[b].shape[1]]#.reshape(-1,1)
        #     h2 = phases[b][:,:phases[a].shape[1]]#.reshape(-1,1)
        #     print(h1.shape,h2.shape)
        #     reshape_size = (2048,200)
        #     h1 = resize(h1,reshape_size)
        #     h2 = resize(h2,reshape_size)
        #     hg.compareHeatmaps(h1,h2,"cluster-"+str(a)+' '+str(b),False)
        

        torch.save(model.state_dict(),path+"/epoch_%d_%.3f.pth" %(epoch,mdist))    
    
    print("done training")
    torch.save(model.state_dict(),path+"/epoch_%d_%.3f.pth" %(epoch,mdist))  
    if f != None:
        f.close()
    hg.compareHeatmaps((orig_heatmap.T,cluster_map.T,best_map.T,mode_map.T),val_name,titles=('original','perfect clustering','lstm','modes'),save=log_file!=None,path='figs/train/'+val_name)
    return precision,recall,transition_p,base_model,best_score,perfect_score,best_orig_score,mode_score,all_background_score
    
params = [k,k,200,True]
if __name__ == "__main__":
    train(name2,params)