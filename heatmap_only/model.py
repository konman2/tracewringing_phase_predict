import torch
import torch.nn as nn
import torch.nn.functional as F
dev = ""
if torch.cuda.is_available():
    dev="cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
class TraceGen(nn.Module):
    def __init__(self,height,embedding_dim=250,hidden_dim=300,embed=True):
        super(TraceGen,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.height = height
        self.embed = embed
        self.embeddings = nn.Embedding(height,embedding_dim)
        #self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=2,batch_first=True,dropout=0.5)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=2,batch_first=True,dropout=0.5)
        #self.dense1 = nn.Linear(hidden_dim,hidden_dim)
        self.dense2 = nn.Linear(hidden_dim,height)

    def forward(self,inputs,input2=None):
        #print(inputs.dtype)
        if self.embed:
            embeds = self.embeddings(inputs)
            print(embeds)
            print(embeds.shape)
        else:
            embeds = inputs
        #print(embeds.dtype)
        lstm_out,_ = self.lstm(embeds)
        #print(lstm_out.shape)
        #print(lstm_out.shape)
        last_out = lstm_out[:,-1]
        #d1 = F.relu(self.dense1(last_out))
        d1 = last_out
        d2 = self.dense2(d1)
        return F.relu(d2)
        