
import torch
from torch import nn
import numpy as np
import configuration


def init_weights(m):
    for name in m.named_parameters():
        if 'weight' in name[0]:
            torch.nn.init.uniform_(name[1], a=-0.1, b=0.1)
        elif 'bias' in name[0]:
            torch.nn.init.zeros_(name[1])

class lstmModel(nn.Module):
    def __init__(self,drop = 0.6,in_size = configuration.IMG_X,out_size = configuration.out_size, n_layers = 2,seq_length = 50):
        super(lstmModel,self).__init__()
        self.in_size = seq_length
        self.out_size = out_size
        self.n_layers = n_layers
        self.hidden_size = 2 * out_size
        self.seq_length = seq_length
        self.norm = nn.BatchNorm1d(in_size)
        self.rnn = nn.LSTM(self.in_size, self.hidden_size, n_layers,
                           dropout = drop, batch_first=True, bidirectional = True)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Sequential(nn.Linear(self.hidden_size,round(1.5 * out_size)),
                                nn.Dropout(drop),
                                nn.LeakyReLU(),
                                nn.Linear(round(1.5 * out_size),round(1.25 * out_size)),
                                nn.Dropout(drop),
                                nn.LeakyReLU(),
                                nn.Linear(round(1.25 * out_size),out_size))
        self.fc2 = nn.Linear(self.hidden_size,out_size)
        self.apply(init_weights)
    def forward(self,x,hidden):
        batch_size = x.size(0)
        seq_num = configuration.SEQ_LENGTH
        x = x.reshape(batch_size,-1)
        out = self.norm(x).reshape(batch_size,seq_num,-1)
        out = self.dropout(out)
        out,hidden = self.rnn(out,hidden)
        out = out.contiguous().view(-1, 2 * self.out_size)
        out = self.dropout(out)
        out = self.fc(out)
        out = out.view(batch_size,-1)
        out = out[:,-2 * self.out_size:]
        out = self.fc2(out)
        #out = out * -1e-5
        return out, hidden
    
    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * 2, batch_size,2 * self.out_size).zero_().to(configuration.device),
                      weight.new(self.n_layers * 2, batch_size,2 * self.out_size).zero_().to(configuration.device))
        return hidden
 #   def init_weights(m):
 #       for name in m.named_parameters():
 #           if 'weight' in name[0]:
 #               torch.nn.init.uniform_(name[1], a=-0.1, b=0.1)
 #           elif 'bias' in name[0]:
 #               torch.nn.init.zeros_(name[1])
        

class basic_model(nn.Module):

    def __init__(self,in_size = configuration.IMG_X,out_size = configuration.out_size,drop = 0.2):
        super(basic_model, self).__init__()
        self.drop = drop
        self.fc1 = nn.Sequential(nn.BatchNorm1d(in_size),
                                 nn.Linear(in_size,in_size),
                                #nn.BatchNorm1d(180),
                                nn.Dropout(p=drop),
                                #nn.ReLU(),
                                #nn.Linear(180,150),
                                #nn.BatchNorm1d(150),
                                #nn.Dropout(p=0.2),
                                nn.ReLU())
        self.fc2 = nn.Linear(in_size,out_size)
        self.apply(init_weights)
    def forward(self,x):
        batch_size = x.size(0)
        x = x.reshape(batch_size,-1)
        out = self.fc1(x)
        out = self.fc2(out)

        #out = out *-1e-5
        return out

class cnn_model(nn.Module):
    def __init__(self,in_size = configuration.IMG_X,out_size = configuration.out_size,drop = 0.2):
        super(cnn_model, self).__init__()
        self.drop = drop
        self.norm = nn.BatchNorm1d(1)
        self.block1 = nn.Sequential(nn.Conv1d(1,3,3),
                                    nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv1d(3,6,3),
                                    nn.ReLU())
        #self.block3 = nn.Sequential(nn.Conv1d(6,16,3),
        #                            nn.ReLU())
        # self.fc1 = nn.Sequential(nn.Linear((in_size - 4) * 16, in_size * 8),
        #                             nn.ReLU(),
        #                             nn.Dropout(drop),
        #                             nn.Linear(in_size * 8, in_size * 4),
        #                             nn.ReLU(),
        #                             nn.Dropout(drop))
        self.fc1 = nn.Sequential(nn.Linear((in_size - 4) * 6, in_size * 4),
                                    nn.ReLU(),
                                    nn.Dropout(drop))
        self.fc_delays = nn.Sequential(nn.Linear(in_size * 4,in_size * 2),
                                    nn.ReLU(),
                                    nn.Dropout(drop),
                                    nn.Linear(in_size * 2,round(out_size/2)))
        self.fc_amp = nn.Sequential(nn.Linear(in_size * 4,in_size * 2),
                                    nn.ReLU(),
                                    nn.Dropout(drop),
                                    nn.Linear(in_size * 2,round(out_size/2)))

        self.apply(init_weights)
    def forward(self,x):
        batch_size = x.size(0)
        x = x.reshape(batch_size,1,-1)
        x = self.norm(x)
        out = self.block1(x)
        #print(out.shape)
        out = self.block2(out)
        #print(out.shape)
        #out = self.block3(out)
        out = out.reshape(batch_size,-1)
        out = self.fc1(out)
        out1 = self.fc_delays(out)
        out2 = self.fc_amp(out)
        #out = out *-1e-5
        return torch.cat((out1,out2), dim = 1)
