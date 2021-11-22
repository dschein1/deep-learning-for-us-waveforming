
from re import M
import torch
from torch import nn
import numpy as np
from torch.nn.modules import padding
from torch.nn.modules.container import Sequential
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


class Dblock(nn.Module):
    def __init__(self,in_size,out_size, drop = 0):
        super(Dblock,self).__init__()
        if drop == 0:
            self.seq1 = nn.Sequential(nn.BatchNorm1d(in_size),
                                    nn.ReLU(),
                                    nn.Conv1d(in_size,out_size,3,stride=2, padding = 1))
            self.seq2 = nn.Sequential(nn.BatchNorm1d(out_size),
                                    nn.ReLU(),
                                    nn.Conv1d(out_size,out_size,3,padding=1),
                                    nn.BatchNorm1d(out_size),
                                    nn.ReLU(),
                                    nn.Conv1d(out_size,out_size,1))
        else:
            self.seq1 = nn.Sequential(nn.Dropout(drop),
                                    nn.ReLU(),
                                    nn.Conv1d(in_size,out_size,3,stride=2, padding = 1))
            self.seq2 = nn.Sequential(nn.Dropout(drop),
                                    nn.ReLU(),
                                    nn.Conv1d(out_size,out_size,3, padding=1),
                                    nn.Dropout(drop),
                                    nn.ReLU(),
                                    nn.Conv1d(out_size,out_size,1))

        self.short = nn.Conv1d(in_size,out_size,3,stride=2, padding=1)

    def forward(self,x):
        identity = x
        out = self.seq1(x)
        out = self.seq2(out)
        identity = self.short(identity)
        return out + identity

class Ublock(nn.Module):
    def __init__(self,in_size,out_size, drop = 0):
        super(Ublock,self).__init__()
        if drop == 0:
            self.seq = nn.Sequential(nn.BatchNorm1d(in_size),
                                    nn.ReLU(),
                                    nn.ConvTranspose1d(in_size,out_size,2,stride=2),
                                    nn.BatchNorm1d(out_size),
                                    nn.ReLU(),
                                    nn.Conv1d(out_size,out_size,3, padding= 1))
        else:
            self.seq = nn.Sequential(nn.Dropout(drop),
                                    nn.ReLU(),
                                    nn.ConvTranspose1d(in_size,out_size,2,stride=2),
                                    nn.Dropout(drop),
                                    nn.ReLU(),
                                    nn.Conv1d(out_size,out_size,3, padding= 1))
        self.short = nn.ConvTranspose1d(in_size,out_size,2,stride=2)
    def forward(self,x):
        identity = x
        out = self.seq(x)
        identity = self.short(identity)
        return out + identity
class Sblock(nn.Module):
    def __init__(self,in_size, drop= 0):
        super(Sblock,self).__init__()
        
        if drop == 0:
            self.seq = nn.Sequential(nn.Conv1d(in_size,in_size,3, padding = 1),
                                    nn.BatchNorm1d(in_size),
                                    nn.ReLU(),
                                    nn.Conv1d(in_size,in_size,3, padding = 1),
                                    nn.ReLU())
        else:
            self.seq = nn.Sequential(nn.Conv1d(in_size,in_size,3, padding = 1),
                                    nn.Dropout(drop),
                                    nn.ReLU(),
                                    nn.Conv1d(in_size,in_size,3, padding = 1),
                                    nn.ReLU())


    def forward(self,x):
        out = self.seq(x)
        return out


class multiResNet(nn.Module):
    def __init__(self,in_size = configuration.IMG_X,out_size = configuration.out_size,drop = 0, k = 6,fcn = True):
        super(multiResNet, self).__init__()
        
        self.k = k
        self.k2 = int(k/2)
        self.num_blocks = 6
        self.is_fcn = fcn
        #total of ten layers, 
        self.dlayers = nn.ModuleList()
        for i in range(self.num_blocks):
            self.dlayers.append(Dblock(k,k, drop))
        #N,k,1
        self.ulayers = nn.ModuleList()
        for i in range(self.num_blocks):
            self.ulayers.append(Ublock(k,k, drop))
        self.slayers = nn.ModuleList()
        for i in range(self.num_blocks):
            self.slayers.append(Sblock(k, drop))
        self.sfinal = nn.Sequential(Ublock(k,k, drop),
                                        Ublock(k,k, drop))
        if fcn == True:
            if drop == 0:
            #expanding channels to k
                self.conv1 = nn.Sequential(nn.Conv1d(1,self.k2,1),
                                        nn.BatchNorm1d(self.k2),
                                        nn.ReLU(),
                                        nn.Conv1d(self.k2,k,1))
            #reduce channels to 1
                self.conv2 = nn.Sequential(nn.BatchNorm1d(k),
                                        nn.ReLU(),
                                        nn.Conv1d(k,self.k2,1),
                                        nn.BatchNorm1d(self.k2),
                                        nn.ReLU(),
                                        nn.Conv1d(self.k2,1,1))
            #reduce spatial to output size
                self.conv3 = nn.Sequential(nn.BatchNorm1d(1),
                                        nn.ReLU(),
                                        nn.Conv1d(1,1,1,stride=2),
                                        nn.BatchNorm1d(1),
                                        nn.ReLU(),
                                        nn.Conv1d(1,1,1,stride=2))
            else:
                        #expanding channels to k
                self.conv1 = nn.Sequential(nn.Conv1d(1,self.k2,1),
                                        nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Conv1d(self.k2,k,1))
            #reduce channels to 1
                self.conv2 = nn.Sequential(nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Conv1d(k,self.k2,1),
                                        nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Conv1d(self.k2,1,1))
            #reduce spatial to output size
                self.conv3 = nn.Sequential(nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Conv1d(1,1,1,stride=2))
                                        # nn.Dropout(drop),
                                        # nn.ReLU(),
                                        # nn.Conv1d(1,1,1,stride=2))
        else:
            if drop == 0:
            #expanding channels to k
                self.conv1 = nn.Sequential(nn.Conv1d(1,self.k2,1),
                                        nn.BatchNorm1d(self.k2),
                                        nn.ReLU(),
                                        nn.Conv1d(self.k2,k,1))
                self.fc1 = nn.Sequential(nn.Linear(k*256,self.k2*256),
                                        nn.ReLU(),
                                        nn.Linear(self.k2 * 256,out_size))
            
            else:
                        #expanding channels to k
                self.conv1 = nn.Sequential(nn.Conv1d(1,self.k2,1),
                                        nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Conv1d(self.k2,k,1))
                self.fc1 = nn.Sequential(nn.Linear(k*256,self.k2*256),
                                        nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Linear(self.k2 * 256,out_size))
            
            
        self.apply(init_weights)


    def forward(self,x):
        x = x.reshape(x.size(0),1,-1)
        x = self.conv1(x)
        ide = []
        for i in range(self.num_blocks):
            ide.append(self.slayers[i](x))
            x = self.dlayers[i](x)
        for i in range(1,self.num_blocks):
            x = self.ulayers[i](x)
            x = x + ide[-i]
            
        x = self.sfinal(x)
        if self.is_fcn == True:
            x = self.conv2(x)
            x = x.reshape(x.size(0), -1)
        else:
            x = x.reshape(x.size(0), -1)
            x = self.fc1(x)
        #x = self.conv3(x)
        return x





        
class multiResNetTest(nn.Module):
    class Dblock(nn.Module):
        def __init__(self,in_size,out_size, drop = 0):
            super(Dblock,self).__init__()
            if drop == 0:
                self.seq1 = nn.Sequential(nn.BatchNorm1d(in_size),
                                        nn.ReLU(),
                                        nn.Conv1d(in_size,in_size,3,stride=2, padding = 1))
                self.seq2 = nn.Sequential(nn.BatchNorm1d(in_size),
                                        nn.ReLU(),
                                        nn.Conv1d(in_size,out_size,3,padding=1),
                                        nn.BatchNorm1d(out_size),
                                        nn.ReLU(),
                                        nn.Conv1d(out_size,out_size,1))
            else:
                self.seq1 = nn.Sequential(nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Conv1d(in_size,in_size,3,stride=2, padding = 1))
                self.seq2 = nn.Sequential(nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Conv1d(in_size,out_size,3, padding=1),
                                        nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Conv1d(out_size,out_size,1))

            self.short = nn.Conv1d(in_size,out_size,3,stride=2, padding=1)

        def forward(self,x):
            identity = x
            out = self.seq1(x)
            out = self.seq2(out)
            identity = self.short(identity)
            return out + identity

    class Ublock(nn.Module):
        def __init__(self,in_size,out_size, drop = 0):
            super(Ublock,self).__init__()
            if drop == 0:
                self.seq = nn.Sequential(nn.BatchNorm1d(in_size),
                                        nn.ReLU(),
                                        nn.ConvTranspose1d(in_size,in_size,2,stride=2),
                                        nn.BatchNorm1d(in_size),
                                        nn.ReLU(),
                                        nn.Conv1d(in_size,out_size,3, padding= 1),
                                        nn.BatchNorm1d(out_size),
                                        nn.ReLU(),
                                        nn.Conv1d(out_size,out_size,1))
            else:
                self.seq = nn.Sequential(nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.ConvTranspose1d(in_size,in_size,2,stride=2),
                                        nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Conv1d(in_size,out_size,3, padding= 1),
                                        nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Conv1d(out_size,out_size,1))
            self.short = nn.ConvTranspose1d(in_size,out_size,2,stride=2)
        def forward(self,x):
            identity = x
            out = self.seq(x)
            identity = self.short(identity)
            return out + identity
    class Sblock(nn.Module):
        def __init__(self,in_size, drop= 0):
            super(Sblock,self).__init__()
            
            if drop == 0:
                self.seq = nn.Sequential(nn.Conv1d(in_size,in_size,3, padding = 1),
                                        nn.BatchNorm1d(in_size),
                                        nn.ReLU(),
                                        nn.Conv1d(in_size,in_size,3, padding = 1),
                                        nn.ReLU())
            else:
                self.seq = nn.Sequential(nn.Conv1d(in_size,in_size,3, padding = 1),
                                        nn.Dropout(drop),
                                        nn.ReLU(),
                                        nn.Conv1d(in_size,in_size,3, padding = 1),
                                        nn.ReLU())


        def forward(self,x):
            out = self.seq(x)
            return out


    
    def __init__(self,in_size = configuration.IMG_X,out_size = configuration.out_size,drop = 0, k = 6):
        super(multiResNetTest, self).__init__()
        self.k = k
        self.k2 = int(k/2)
        self.num_blocks = 6
        #total of ten layers, 
        self.dlayers = nn.ModuleList()
        for i in range(1,self.num_blocks + 1):
            self.dlayers.append(Dblock(k* i ,k * (i + 1), drop))
        #N,k,1
        self.ulayers = nn.ModuleList()
        for i in range(self.num_blocks + 1,0,-1):
            self.ulayers.append(Ublock(k * (i + 1),k*i, drop))
        self.slayers = nn.ModuleList()
        for i in range(1,self.num_blocks + 1):
            self.slayers.append(Sblock(k * i, drop))
        self.sfinal = nn.Sequential(Ublock(k,k, drop))
                                       # Ublock(k,k, drop))
        if drop == 0:
        #expanding channels to k
            self.conv1 = nn.Sequential(nn.Conv1d(1,self.k2,1),
                                    nn.BatchNorm1d(self.k2),
                                    nn.ReLU(),
                                    nn.Conv1d(self.k2,k,1))
        #reduce channels to 1
            self.conv2 = nn.Sequential(nn.BatchNorm1d(k),
                                    nn.ReLU(),
                                    nn.Conv1d(k,self.k2,1),
                                    nn.BatchNorm1d(self.k2),
                                    nn.ReLU(),
                                    nn.Conv1d(self.k2,1,1))
        #reduce spatial to output size
            self.conv3 = nn.Sequential(nn.BatchNorm1d(1),
                                    nn.ReLU(),
                                    nn.Conv1d(1,1,1,stride=2),
                                    nn.BatchNorm1d(1),
                                    nn.ReLU(),
                                    nn.Conv1d(1,1,1,stride=2))
        else:
                    #expanding channels to k
            self.conv1 = nn.Sequential(nn.Conv1d(1,self.k2,1),
                                    nn.Dropout(drop),
                                    nn.ReLU(),
                                    nn.Conv1d(self.k2,k,1))
        #reduce channels to 1
            self.conv2 = nn.Sequential(nn.Dropout(drop),
                                    nn.ReLU(),
                                    nn.Conv1d(k,self.k2,1),
                                    nn.Dropout(drop),
                                    nn.ReLU(),
                                    nn.Conv1d(self.k2,1,1))
        #reduce spatial to output size
            self.conv3 = nn.Sequential(nn.Dropout(drop),
                                    nn.ReLU(),
                                    nn.Conv1d(1,1,1,stride=2))
                                    # nn.Dropout(drop),
                                    # nn.ReLU(),
                                    # nn.Conv1d(1,1,1,stride=2))
        self.apply(init_weights)


    def forward(self,x):
        x = x.reshape(x.size(0),1,-1)
        x = self.conv1(x)
        ide = []
        for i in range(self.num_blocks):
            ide.append(self.slayers[i](x))
            x = self.dlayers[i](x)
        for i in range(1,self.num_blocks + 1):
            x = self.ulayers[i](x)
            x = x + ide[-i]
            
        x = self.sfinal(x)
        x = self.conv2(x)
        #x = self.conv3(x)
        x = x.reshape(x.size(0), -1)
        return x