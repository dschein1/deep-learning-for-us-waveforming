
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt
import matlab.engine
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import distance_matrix
import numpy as np
import configuration
import nets
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_lines(eng,delays):
    delays = torch.reshape(delays,(-1,2 * 128))
    (delays, amps) = torch.split(delays,128,1)
    #print(amps.shape,delays.shape)
    batch_size = delays.shape[0]
    batch = np.zeros((batch_size,configuration.IMG_X))
    for i in range(batch_size):
        delay = matlab.double(delays[i].tolist())
        amp = matlab.double(amps[i].tolist())
        batch[i,:] = np.asarray(eng.create_new_line(delay,amp))
    return batch

def create_images(eng,delays):
    delays = torch.reshape(delays,(-1,2 * 128))
    (delays, amps) = torch.split(delays,128,1)
    batch_size = delays.shape[0]
    batch = np.zeros((batch_size,configuration.IMG_Y, configuration.IMG_X))
    for i in range(batch_size):
        delay = matlab.double(delays[i].tolist())
        amp = matlab.double(amps[i].tolist())
        batch[i,:,:] = np.asarray(eng.create_new_image(delay,amp))
    # delays = matlab.double(delays.tolist())
    # amps = matlab.double(amps.tolist())
    # batch  = eng.create_new_image(delays,amps)
    return batch

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


def create_patterns_1d(amount,N,seq_length = 50):
    patterns = np.zeros((amount,N))
    number_in_each = np.random.randint(1,3,amount)
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
            #upper = min(configuration.IMG_X,position+round(sinc_len/2) - 1)
            #actual_len = upper - lower + 1
            #print(lower,upper,actual_len)
            #if lower == 0:
            #    to_apply = sinc[-actual_len:]
            #elif upper == configuration.IMG_X:
            #      to_apply = sinc[:actual_len - 1]
            #else:
            #      to_apply = sinc
            #patterns[i,lower:upper] += to_apply
    return torch.from_numpy(patterns)


class complex_normalize(object):
    def __init__(self) -> None:
        super().__init__()
    def fit(self,sample):
        self.real_min = np.real(sample).min()
        self.comp_min = np.imag(sample).min()
        self.sub = self.real_min + 1j * self.comp_min
        self.abs = np.abs(sample).max().max()
    
    def __call__(self,sample):
        return (sample - self.sub)/self.abs 
    def reverse(self,sample):
        return (sample * self.abs) + self.abs

class baseDataSet(Dataset):
    def __init__(self,indexes,source_data,seq_length = 50 ,csv_file = "C:/Users/drors/Desktop/code for thesis/data.csv",transforms = None, mode = 'both'):
        #self.data = pd.read_csv(csv_file,header = None, index_col = 0,skiprows = )
        self.data = source_data.iloc[indexes,:].copy()
        self.csv = csv_file
        self.measure = nn.CosineSimilarity()
        self.seq_length = seq_length
        self.mode = mode
        self.converters = {}
        self.transform_x = False
        self.transform_y = False
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        if self.mode == 'both':
            x = self.data.iloc[idx,:configuration.IMG_X]
            y = pd.concat([self.data.iloc[idx,configuration.IMG_X:configuration.IMG_X + 128] * 1e5,self.data.iloc[idx,configuration.IMG_X+128:]], axis = 0)
        else:
            x = self.data.iloc[idx,:configuration.IMG_X]
            y = self.data.iloc[idx,configuration.IMG_X:]
        # if len(y) == 2 * 128:
        #     y = torch.tensor(y.values).split(128,1)
        # else:
        #     y = torch.tensor(y.values)
        x = np.abs(x.values)
        x = torch.tensor(x)
        y = torch.tensor(y.values)
        if self.transform_x:
            x = self.transform_x.transform(x.reshape(1,-1)).reshape(-1)
            y = self.transform_y.transform(y.reshape(1,-1)).reshape(-1)
        return torch.tensor(x),torch.tensor(y)


    def test_get(self,idx):
        x = self.data.iloc[idx,:configuration.IMG_X]
        y = pd.concat([self.data.iloc[idx,configuration.IMG_X:configuration.IMG_X + 128] * 1e5,self.data.iloc[idx,configuration.IMG_X+128:]], axis = 0)
        return torch.tensor(x.values).reshape(-1,self.seq_length),torch.tensor(y.values)
    def get_data(self):
        return self.data
    def add_batch(self,new_data):
        #similarity = cosine_similarity(new_data,self.data)
        distance = distance_matrix(new_data.iloc[:,:configuration.IMG_X],self.data.iloc[:,:configuration.IMG_X])
        mask = np.all(distance > 1e-2, axis = 1)

        #print(new_data.shape,self.data.shape,similarity.shape,mask.shape)
        new_data = new_data.loc[mask,:]
        #new_df = pd.DataFrame(new_data,header = None)
        self.data = self.data.append(new_data,ignore_index = True)
        if len(self.data) > 10000:
            self.data = self.data.iloc[len(self.data) - 10000:,:]
            #self.data = pd.concat([self.data.iloc[:50,:],self.data.iloc[(len(self.data) - 1000):,:]]) 
        #new_df.to_csv(self.csv,mode = 'a', header = None)
    def set_transform(self,transform_x,transform_y):
        self.transform_x = transform_x
        self.transform_y = transform_y

class datasetManager():
    def __init__(self,seq_length = 50,csv_file = "C:/Users/drors/Desktop/code for thesis/data.csv",orig = "C:/Users/drors/Desktop/code for thesis/data original.csv"):
        self.eng = []
        for i in range(configuration.num_workers):
            self.eng.append(matlab.engine.start_matlab())
            self.eng[i].init_field(nargout=0)
        self.csv_file = csv_file
        if orig == configuration.path_combined:
            self.mode = 'both'
        else:
            self.mode = 'single'
        self.orig = orig
        self.train,self.val,self.test = self.create_datasets()
        self.seq_length = seq_length
        self.x_scalar = MinMaxScaler()
        self.y_scalar = MinMaxScaler()
        self.x_scalar.fit(self.train.data.iloc[:,:configuration.IMG_X])
        self.y_scalar.fit(self.train.data.iloc[:,configuration.IMG_X:])
        self.train.set_transform(self.x_scalar,self.y_scalar)
        self.test.set_transform(self.x_scalar,self.y_scalar)
        self.val.set_transform(self.x_scalar,self.y_scalar)
    def create_datasets(self):
        data = pd.read_csv(self.orig,header = None,index_col = None, engine = 'python') #.applymap(lambda x: np.complex(x.replace(" ", "").replace('i','j')))
        n = len(data)
        perm = np.random.permutation(n)
        train_idx = perm[:round(0.8*n)]
        val_idx = perm[round(0.8*n):round(0.9*n)]
        test_idx = perm[round(0.9*n):]
        train_dataset = baseDataSet(train_idx,data,mode = self.mode)
        val_dataset = baseDataSet(val_idx,data,mode = self.mode)
        test_dataset = baseDataSet(test_idx,data,mode = self.mode)
        return train_dataset,val_dataset,test_dataset
    def get_datasets(self):
        return self.train,self.val,self.test
    def reset(self):
        data = pd.read_csv(self.orig,header = None,index_col = None, engine = 'python') #.applymap(lambda x: np.complex(x.replace(" ", "").replace('i','j')))
        n = len(data)
        perm = np.random.permutation(n)
        train_idx = perm[:round(0.8*n)]
        val_idx = perm[round(0.8*n):round(0.9*n)]
        test_idx = perm[round(0.9*n):]
        train_dataset = baseDataSet(train_idx,data,mode = self.mode)
        val_dataset = baseDataSet(val_idx,data,mode = self.mode)
        test_dataset = baseDataSet(test_idx,data,mode = self.mode)
        self.train,self.val,self.test =  train_dataset,val_dataset,test_dataset
        self.x_scalar.fit(self.train.data.iloc[:,:configuration.IMG_X])
        self.y_scalar.fit(self.train.data.iloc[:,configuration.IMG_X:])
        self.train.set_transform(self.x_scalar,self.y_scalar)
        self.test.set_transform(self.x_scalar,self.y_scalar)
        self.val.set_transform(self.x_scalar,self.y_scalar)
        copyfile(self.orig,self.csv_file)
        
    def add_batch_to_data(self,batch, mode = 'create'):
        new_df = pd.DataFrame(batch).astype('float')
        n = len(new_df)
        perm = np.random.permutation(n)
        train_idx = perm[:round(0.8*n)]
        val_idx = perm[round(0.8*n):round(0.9*n)]
        test_idx = perm[round(0.9*n):]
        if mode == 'create':
            self.train = baseDataSet(train_idx,new_df) # .add_batch(new_df.iloc[train_idx,:])
            self.val = baseDataSet(val_idx,new_df)# .add_batch(new_df.iloc[val_idx,:])
            self.test = baseDataSet(test_idx,new_df)# .add_batch(new_df.iloc[test_idx,:])
            new_df.to_csv(self.csv_file, header = None,index = False)
            self.x_scalar.fit(self.train.data.iloc[:,:configuration.IMG_X])
            self.y_scalar.fit(self.train.data.iloc[:,configuration.IMG_X:])
            self.train.set_transform(self.x_scalar,self.y_scalar)
            self.test.set_transform(self.x_scalar,self.y_scalar)
            self.val.set_transform(self.x_scalar,self.y_scalar)
        else:
            self.train.add_batch(new_df.iloc[train_idx,:])
            self.val.add_batch(new_df.iloc[val_idx,:])
            self.test.add_batch(new_df.iloc[test_idx,:])
            new_df.to_csv(self.csv_file,mode = 'a', header = None,index = False)
    def generate_base_dataset(self,amount = configuration.create_amount, mode = 'create'):
        amount *= 5
        delays = torch.rand(amount,128) * 1e-5
        delays = torch.cat(delays,torch.ones(amount,128),1)
        with ThreadPoolExecutor() as executor:
            new_delay = torch.reshape(delays,(len(self.eng),-1))
            new_delay = new_delay
            results = executor.map(create_lines,self.eng,new_delay)
        res = np.zeros((amount,configuration.IMG_X))
        size_eng = amount / len(self.eng)
        for i,line in enumerate(results):
            res[round(size_eng * i):round(size_eng *(i+1)),:] = line
        self.add_batch_to_data(torch.cat((torch.tensor(res),delays),1),mode)
        torch.cat((torch.tensor(res),delays),1)
        
    def create_batch(self,net,amount = configuration.create_amount):
        patterns = create_patterns_1d(amount,configuration.IMG_X)
        net.eval()
        if isinstance(net,nets.lstmModel):
            h = net.init_hidden(amount)
            patterns_to_send = patterns.reshape(amount,-1,self.seq_length)
            delays,_ = net(patterns_to_send.float(),h)
        else:
            #patterns_to_send = patterns.reshape(amount,-1,self.seq_length)
            delays = net(patterns.float())
        
        delays[:,:128] *= 1e-5
        #while np.any(delays.detach().numpy() > 1e-5):
        #    delays *= 1e-1
        if torch.any(torch.isnan(delays)):
            print(delays)
        with ThreadPoolExecutor() as executor:
            new_delay = torch.reshape(delays,(configuration.num_workers,-1))
            results = executor.map(create_lines,self.eng,new_delay)
        res = np.zeros((amount,configuration.IMG_X))
        size_eng = amount / len(self.eng)
        for i,line in enumerate(results):
            res[round(size_eng * i):round(size_eng *(i+1)),:] = line
        return torch.from_numpy(res),delays,patterns

    def create_pressure_batch(self,delays,amount = 8):
        with ThreadPoolExecutor() as executor:
            #delays = delays * 1e-5
            new_delay = torch.reshape(delays,(configuration.num_workers,-1))
            results = executor.map(create_images,self.eng,new_delay)
        res = np.zeros((amount,configuration.IMG_Y,configuration.IMG_X))
        size_eng = amount / len(self.eng)
        for i,line in enumerate(results):
            res[round(size_eng * i):round(size_eng *(i+1)),:,:] = line
        return res

