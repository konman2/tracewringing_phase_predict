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
    def __init__(self,vocab_size,embedding_dim=50,hidden_dim=100,embed=True):
        super(TraceGen,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed = embed
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        #self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=2,batch_first=True,dropout=0.5)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=2,batch_first=True,dropout=0.5)
        #self.dense1 = nn.Linear(hidden_dim,hidden_dim)
        self.dense2 = nn.Linear(hidden_dim,vocab_size)

    def forward(self,inputs,input2=None):
        #print(inputs.shape)
        # # print(inputs.shape)
        if self.embed:
            embeds = self.embeddings(inputs)
        else:
            inputs = inputs.reshape(inputs.shape[0],inputs.shape[1],1)
            embeds = torch.FloatTensor(inputs.shape[0],inputs.shape[1],self.vocab_size).zero_().to(device)
            embeds.scatter_(2,inputs,1)
        #embeds = torch.cat((embeds,input2.float().reshape(input2.shape[0],input2.shape[1],1)),2)
        lstm_out,_ = self.lstm(embeds)
        #print(lstm_out.shape)
        last_out = lstm_out[:,-1]
        #d1 = F.relu(self.dense1(last_out))
        d1 = last_out
        d2 = self.dense2(d1)
        return d2
        