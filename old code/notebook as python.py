# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matlab.engine
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import pandas as pd
import os.path
from shutil import copyfile
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance_matrix


# %%
#run time configurations
num_workers = 4
create_amount = 200
batch_size = 20
IMG_X = 200
IMG_Y = 300
depth = 40e-3
dz = (80e-3 - 10e-3)/300

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# %%
eng = []
for i in range(num_workers):
    eng.append(matlab.engine.start_matlab())
    eng[i].init_field(nargout=0)


# %%



# %%
def extract_line(depth,dz,im):
    return im[:,(depth-10e-3)/dz,:]


# %%
def create_lines(eng,delays):
    delays = torch.reshape(delays,(-1,128))
    delays = matlab.double(delays.tolist())
    batch  = eng.create_new_line(delays)
    return batch


# %%
def create_images(eng,delays):
    delays = torch.reshape(delays,(-1,128))
    delays = matlab.double(delays.tolist())
    batch  = eng.create_new_image(delays)
    return batch


# %%
def check_minimal_distance(a,b,minimal):
    if isinstance(a,list):
        for loc in a:
            if abs(loc - b) < minimal:
                return False
        return True
    else:
        if a == 0:
            return True
        return abs(a - b) > minimal


# %%
dx = (15e-3 + 15e-3)/IMG_X
c = 1490
f0 = 4.464e6
D = 0.218e-3 * 128
minimal_distance = (1.206 * (c/f0) * 40e-3) / D
minimum = minimal_distance / dx
f = np.vectorize(check_minimal_distance)
x = np.arange(-10*minimal_distance,10*minimal_distance,dx)
sinc = np.sinc(1045.61 * x) #number computed numerically for the width of the sinc
plt.plot(x,sinc)
print(sinc.size)


# %%
def create_patterns_1d(amount,N,seq_length = 50):
    patterns = np.zeros((amount,N))
    number_in_each = np.random.randint(1,2,amount)
    dx = (15e-3 + 15e-3)/N
    c = 1490
    f0 = 4.464e6
    D = 0.218e-3 * 128
    minimal_distance = (1.206 * (c/f0) * 40e-3) / D
    minimum = minimal_distance / dx
    f = np.vectorize(check_minimal_distance)
    #x = np.arange(-10*minimal_distance,10*minimal_distance,dx)
    x = np.linspace(-30e-3,30e-3,N * 2)
    sinc = np.sinc(1045.61 * x) #number computed numerically for the width of the sinc
    #sinc = np.sinc()
    sinc_len = len(sinc)
    for i in range(amount):
        j = 0
        actual = []
        while j < number_in_each[i]:
            point = np.random.randint(0,N,1)
            val = point[0]
            if check_minimal_distance(actual,point,minimum):
                actual.append(val)
                j += 1
        #patterns[i,positions] = 1
        for position in actual:
            shifted_sinc = np.roll(sinc,position - 100)
            patterns[i,:] += shifted_sinc[round(N/2):round(3*N/2)]
            #position = positions[k,0]
            #lower = max(0,position - round(sinc_len/2))
            #upper = min(IMG_X,position+round(sinc_len/2) - 1)
            #actual_len = upper - lower + 1
            #print(lower,upper,actual_len)
            #if lower == 0:
            #    to_apply = sinc[-actual_len:]
            #elif upper == IMG_X:
            #      to_apply = sinc[:actual_len - 1]
            #else:
            #      to_apply = sinc
            #patterns[i,lower:upper] += to_apply
    return torch.from_numpy(patterns)


# %%
x = np.linspace(-30e-3,30e-3,IMG_X * 2)
sinc = np.sinc(1045.61 * x) #number computed numerically for the width of the sinc            shifted_sinc = np.roll(sinc,round(N/2 + position))
shifted_sinc = np.roll(sinc,150)
plt.plot(x,shifted_sinc)    


# %%
patterns = create_patterns_1d(16,IMG_X)
x = np.linspace(-15e-3,15e-3,IMG_X)
fig, ax = plt.subplots(4,4)
for i,pattern in enumerate(patterns):
    ax[i // 4][i % 4].plot(x,pattern)


# %%
class baseDataSet(Dataset):
    def __init__(self,indexes,source_data,seq_length = 50 ,csv_file = "C:/Users/drors/Desktop/code for thesis/data.csv",transforms = None):
        #self.data = pd.read_csv(csv_file,header = None, index_col = 0,skiprows = )
        self.data = source_data.iloc[indexes,:].copy()
        self.csv = csv_file
        self.measure = nn.CosineSimilarity()
        self.seq_length = seq_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        x = self.data.iloc[idx,:IMG_X]
        y = self.data.iloc[idx,IMG_X:] * 1e4
        return torch.tensor(x.values).reshape(-1,self.seq_length),torch.tensor(y.values)
    def get_data(self):
        return self.data
    def add_batch(self,new_data):
        #similarity = cosine_similarity(new_data,self.data)
        distance = distance_matrix(new_data,self.data)
        mask = np.all(distance > 5e-5, axis = 1)

        #print(new_data.shape,self.data.shape,similarity.shape,mask.shape)
        new_data = new_data.loc[mask,:]
        #new_df = pd.DataFrame(new_data,header = None)
        self.data = self.data.append(new_data,ignore_index = True)
        if len(self.data) > 10000:
            self.data = self.data.iloc[(len(self.data) - 10000):,:] 
        #new_df.to_csv(self.csv,mode = 'a', header = None)


# %%
class datasetManager():
    def __init__(self,eng,seq_length = 50,csv_file = "C:/Users/drors/Desktop/code for thesis/data.csv",orig = "C:/Users/drors/Desktop/code for thesis/data original.csv"):
        self.eng = eng
        self.csv_file = csv_file
        self.orig = orig
        self.train,self.val,self.test = self.create_datasets()
        self.seq_length = seq_length
        
    def create_datasets(self):
        data = pd.read_csv(self.orig,header = None,index_col = None, engine = 'python')
        n = len(data)
        perm = np.random.permutation(n)
        train_idx = perm[:round(0.8*n)]
        val_idx = perm[round(0.8*n):round(0.9*n)]
        test_idx = perm[round(0.9*n):]
        train_dataset = baseDataSet(train_idx,data)
        val_dataset = baseDataSet(val_idx,data)
        test_dataset = baseDataSet(test_idx,data)
        return train_dataset,val_dataset,test_dataset
    def get_datasets(self):
        return self.train,self.val,self.test
    def reset(self):
        data = pd.read_csv(self.orig,header = None,index_col = None, engine = 'python')
        n = len(data)
        perm = np.random.permutation(n)
        train_idx = perm[:round(0.8*n)]
        val_idx = perm[round(0.8*n):round(0.9*n)]
        test_idx = perm[round(0.9*n):]
        train_dataset = baseDataSet(train_idx,data)
        val_dataset = baseDataSet(val_idx,data)
        test_dataset = baseDataSet(test_idx,data)
        self.train,self.val,self.test =  train_dataset,val_dataset,test_dataset
        copyfile(self.orig,self.csv_file)
        
    def add_batch_to_data(self,batch, mode = 'create'):
        new_df = pd.DataFrame(batch).astype('float')
        new_df.to_csv(self.csv_file,mode = 'a', header = None,index = False)
        n = len(new_df)
        perm = np.random.permutation(n)
        train_idx = perm[:round(0.8*n)]
        val_idx = perm[round(0.8*n):round(0.9*n)]
        test_idx = perm[round(0.9*n):]
        if mode == 'create':
            self.train = baseDataSet(train_idx,new_df) # .add_batch(new_df.iloc[train_idx,:])
            self.val = baseDataSet(val_idx,new_df)# .add_batch(new_df.iloc[val_idx,:])
            self.test = baseDataSet(test_idx,new_df)# .add_batch(new_df.iloc[test_idx,:])
        else:
            self.train.add_batch(new_df.iloc[train_idx,:])
            self.val.add_batch(new_df.iloc[val_idx,:])
            self.test.add_batch(new_df.iloc[test_idx,:])
    def generate_base_dataset(self,amount = create_amount, mode = 'create'):
        amount *= 5
        delays = torch.rand(amount,128)
        with ThreadPoolExecutor() as executor:
            new_delay = torch.reshape(delays,(len(self.eng),-1))
            new_delay = new_delay * 1e-4
            results = executor.map(create_lines,self.eng,new_delay)
        res = np.zeros((amount,IMG_X))
        size_eng = amount / len(self.eng)
        for i,line in enumerate(results):
            res[round(size_eng * i):round(size_eng *(i+1)),:] = line
#        print(np.fromiter(results,matlab.double))
        self.add_batch_to_data(torch.cat((torch.tensor(res),delays),1),mode)
        torch.cat((torch.tensor(res),delays),1)
    def create_batch(self,net,amount = create_amount):
        patterns = create_patterns_1d(amount,IMG_X)
        net.eval()
        if isinstance(net,lstmModel):
            h = net.init_hidden(amount)
            patterns_to_send = patterns.reshape(amount,-1,self.seq_length)
            delays,_ = net(patterns_to_send.float(),h)
        else:
            #patterns_to_send = patterns.reshape(amount,-1,self.seq_length)
            delays = net(patterns.float())
        
        delays *= 1e-4
        #while np.any(delays.detach().numpy() > 1e-4):
        #    delays *= 1e-1
        if torch.any(torch.isnan(delays)):
            print(delays)
        with ThreadPoolExecutor() as executor:
            new_delay = torch.reshape(delays,(len(self.eng),-1))
            results = executor.map(create_lines,self.eng,new_delay)
        res = np.zeros((amount,IMG_X))
        size_eng = amount / len(self.eng)
        for i,line in enumerate(results):
            res[round(size_eng * i):round(size_eng *(i+1)),:] = line
#        print(np.fromiter(results,matlab.double))
        return torch.from_numpy(res),delays,patterns

    def create_pressure_batch(self,delays,amount = 8):
        with ThreadPoolExecutor() as executor:
            delays = delays * 1e-4
            new_delay = torch.reshape(delays,(len(self.eng),-1))
            results = executor.map(create_images,self.eng,new_delay)
        res = np.zeros((amount,IMG_Y,IMG_X))
        size_eng = amount / len(self.eng)
        for i,line in enumerate(results):
            res[round(size_eng * i):round(size_eng *(i+1)),:,:] = line
        return res


# %%
def test_batch(net,data_manager,amount = 8):
    results,delays,orig_patterns = data_manager.create_batch(net,amount)
    images = data_manager.create_pressure_batch(delays)
    x = np.linspace(-15e-3,15e-3,IMG_X)
    z = np.linspace(10e-3,80e-3,300)
    if amount == 4:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10,5))
        ax2.plot(x *1e3,results[0,:])
        ax2.set_title('result from net')
        ax1.plot(x *1e3,orig_patterns[0,:])
        ax1.set_title('expected')
        ax3.imshow(np.rot90(images[0,:,:],4),cmap = 'hot',extent = [-15,15,80,10])
    else:
        for i in range(amount):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10,5))
            ax2.plot(x *1e3,results[i,:])
            ax2.set_title('result from net')
            ax1.plot(x *1e3,orig_patterns[i,:])
            ax1.set_title('expected')
            ax3.imshow(np.rot90(images[i,:,:],4),cmap = 'hot',extent = [-15,15,80,10])


# %%
dataManager = datasetManager(eng)


# %%
class lstmModel(nn.Module):
    def __init__(self,drop = 0.6,in_size = IMG_X,out_size = 128, n_layers = 2,seq_length = 50):
        super(lstmModel,self).__init__()
        self.in_size = seq_length
        self.out_size = out_size
        self.n_layers = n_layers
        self.seq_length = seq_length
        self.rnn = nn.LSTM(self.in_size, out_size, n_layers,
                           dropout = drop, batch_first=True, bidirectional = True,
                           bias = False)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(out_size,out_size, bias = False)
        self.fc2 = nn.Linear(2* out_size,out_size, bias = False)
        
    def forward(self,x,hidden):
        batch_size = x.size(0)
        out = self.dropout(x)
        out,hidden = self.rnn(out,hidden)
        out = out.contiguous().view(-1, self.out_size)
        out = self.dropout(out)
        out = self.fc(out)
        out = out.view(batch_size,-1)
        out = out[:,-2 * self.out_size:]
        out = self.fc2(out)
        #out = out * -1e-5
        return out, hidden
    
    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * 2, batch_size, self.out_size).zero_().to(device),
                      weight.new(self.n_layers * 2, batch_size, self.out_size).zero_().to(device))
        return hidden


# %%
def trainLstm(net,opt,dataManager,schedular,n_epochs = 30,batch_size = 20):
    datasets = dataManager.get_datasets()
    train = DataLoader(datasets[0],batch_size = batch_size, shuffle = True, drop_last = True)
    val = DataLoader(datasets[1],batch_size = batch_size, shuffle = True, drop_last = True)
    test = DataLoader(datasets[2],batch_size = batch_size, shuffle = True, drop_last = True)
    net.to(device)
    train_loss_total = []
    val_loss_total = []
    test_loss_total = []
    criterion = nn.L1Loss() #will probably have outliers, L1 is more robust
    #criterion = nn.MSELoss()
    #criterion = nn.CosineSimilarity()
    for i in range(n_epochs):
        training_loss = 0
        val_loss = 0
        net.train()
        h = net.init_hidden(batch_size)
        for x,y in train:
            h = tuple([e.data for e in h])
            x.to(device),y.to(device)
            opt.zero_grad()
            output, h  = net(x.float(),h)
            loss = criterion(output,y.float())
            loss.mean().backward()
            training_loss += loss.mean().item()
            #nn.utils.clip_grad_norm_(net.parameters(), 1) 
            opt.step()
        with torch.no_grad():
            net.eval()
            h = net.init_hidden(batch_size)
            for x,y in val:
                h = tuple([e.data for e in h])
                x.to(device),y.to(device)
                output, h  = net(x.float(),h)
                loss = criterion(output,y.float())
                val_loss += loss.mean().item()
        training_loss = training_loss / len(train)
        if len(val) == 0:
            val_loss = 'no values'
        else:
            val_loss = val_loss / len(val)
        print(f'epoch num: {i}, train loss: {training_loss}, validation loss:{val_loss}, length of train {len(train) * batch_size}')
        schedular.step()
        if i % 3 == 0:
            for name, param in net.named_parameters():
                if param.requires_grad:
                    print (name, param.data)
        results,delays,orig_patterns = dataManager.create_batch(net,amount = 500)
        dataManager.add_batch_to_data(torch.cat((results,delays),1), mode = 'add')


# %%
class basic_model(nn.Module):
    
    def __init__(self,in_size = IMG_X,out_size = 128,drop = 0.2):
        super(basic_model, self).__init__()
        self.drop = drop
        self.fc1 = nn.Sequential(nn.BatchNorm1d(in_size),
                                 nn.Linear(in_size,150),
                                #nn.BatchNorm1d(180),
                                nn.Dropout(p=drop),
                                #nn.ReLU(),
                                #nn.Linear(180,150),
                                #nn.BatchNorm1d(150),
                                #nn.Dropout(p=0.2),
                                nn.ReLU())
        self.fc2 = nn.Linear(150,out_size)
    
    def forward(self,x):
        batch_size = x.size(0)
        x = x.reshape(batch_size,-1)
        out = self.fc1(x)
        out = self.fc2(out)

        #out = out *-1e-5
        return out


# %%
def trainModel(net,opt,dataManager,schedular,n_epochs = 30,batch_size = 20):
    datasets = dataManager.get_datasets()
    train = DataLoader(datasets[0],batch_size = batch_size, shuffle = True, drop_last = True)
    val = DataLoader(datasets[1],batch_size = batch_size, shuffle = True, drop_last = True)
    test = DataLoader(datasets[2],batch_size = batch_size, shuffle = True, drop_last = True)
    net.to(device)
    train_loss_total = []
    val_loss_total = []
    test_loss_total = []
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss() #will probably have outliers, L1 is more robust
    #criterion = nn.CosineSimilarity()
    for i in range(n_epochs):
        training_loss = 0
        val_loss = 0
        net.train()
        for x,y in train:
            x.to(device),y.to(device)
            opt.zero_grad()
            output = net(x.float())
            loss = criterion(output,y.float())
            loss.mean().backward()
            training_loss += loss.mean().item()
            opt.step()
        with torch.no_grad():
            net.eval()
            for x,y in val:
                x.to(device),y.to(device)
                output = net(x.float())
                loss = criterion(output,y.float())
                val_loss += loss.mean().item()
        training_loss = training_loss / len(train)
        if len(val) == 0:
            val_loss = 'no values'
        else:
            val_loss = val_loss / len(val)
        print(f'epoch num: {i}, train loss: {training_loss}, validation loss:{val_loss}, length of train {len(train)}')
        schedular.step()
        results,delays,orig_patterns = dataManager.create_batch(net,amount = 500)
        dataManager.add_batch_to_data(torch.cat((results,delays),1), mode = 'add')
#        if i%3 == 0:
#            print(f'testing iteration number: {i}')
#            test_batch(net,dataManager,amount = 4)


# %%
dataManager.reset()
dataManager.generate_base_dataset()
#dataManager.generate_base_dataset(mode = 'add')
#dataManager.generate_base_dataset(mode = 'add')
#dataManager.generate_base_dataset(mode = 'add')
#dataManager.generate_base_dataset(mode = 'add')


# %%
net = lstmModel(n_layers = 2)
optimizer = optim.SGD(net.parameters(),lr = 0.01, momentum = 0.9, weight_decay = 5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
trainLstm(net,optimizer,dataManager,scheduler,n_epochs = 10)
test_batch(net,dataManager)


# %%
net = basic_model(drop = 0.6)
print(net.fc1[0].weight,net.fc1[0].bias)
optimizer = optim.SGD(net.parameters(),lr = 1, momentum = 0.9, weight_decay = 5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
trainModel(net,optimizer,dataManager,scheduler,n_epochs = 20)
test_batch(net,dataManager)


# %%
print(net.fc1[0].weight,net.fc1[0].bias)


# %%
net = basic_model(drop = 0.6)
print('trying to create new batch')
results,delays,orig_patterns = dataManager.create_batch(net)
print('created new batch, trying to add to data')
dataManager.add_batch_to_data(torch.cat((results,delays),1))


# %%
test_batch(net,dataManager)




# %%
print(len(dataManager.get_datasets()[0]))


# %%
datasets = dataManager.get_datasets()
train = datasets[0]
x_1,y_1 = train[20]
x_2,y_2 = train[5]
x_1 = x_1.reshape(1,-1,50).float()
x_2 = x_2.reshape(1,-1,50).float()
criterion = nn.MSELoss()
h_1 = net.init_hidden(1)
h_2 = net.init_hidden(1)
print(x_1.size())
print(criterion(x_1,x_2))
print(criterion(net(x_1,h_1)[0],net(x_2,h_2)[0]))
print(criterion(y_1,y_2))


# %%
test_batch(net,dataManager)


# %%



# %%
'''
copied from matlab, if want to use need to adjust

def calc_delay(focus,N_elements = 128, c = 1490, pitch = 0.218e-3):
    first = np.norm(focus)
    centers = np.arange(-num_elements/2+1,num_elements/2)
    centers = centers * pitch
    centers = [centers zeros(length(centers),2)];
    second = vecnorm(centers - focus,2,2)
    Delay = (first - second)/c
'''


# %%



