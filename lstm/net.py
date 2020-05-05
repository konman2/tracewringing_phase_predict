import torch
import torch.nn as nn
import torch.nn.functional as F
class TraceGen(nn.Module):
    def __init__(self,vocab_size,embedding_dim=50,hidden_dim=100):
        super(TraceGen,self).__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=1,batch_first=True)
        #self.lstm = nn.LSTM(vocab_size,hidden_dim,num_layers=1,batch_first=True)
        #self.dense1 = nn.Linear(hidden_dim,hidden_dim)
        self.dense2 = nn.Linear(hidden_dim,vocab_size)

    def forward(self,inputs):
        #print(inputs.shape)
        embeds = self.embeddings(inputs)

        lstm_out,_ = self.lstm(embeds)
        #print(lstm_out.shape)
        last_out = lstm_out[:,-1]
        # d1 = F.relu(self.dense1(last_out))
        #d1 = last_out
        d2 = self.dense2(last_out)
        return d2
        