
from cgi import test
from email.mime import base
import torch
from torch import chunk, dtype, nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import distance_matrix
import numpy as np
from zmq import device
import configuration
import nets
import json 
import os
import dask.dataframe as dd
from scipy.signal import find_peaks
import dask
from dask.distributed import Client
#import matlab

column_types_real = {str(i):'float64' for i in range(1,513)} #pd.SparseDtype(dtype='int8',fill_value=0) for i in range(1,513)}
column_types_real.update({str(i):'int8' for i in range(513,513 + 128)}) #pd.SparseDtype(dtype='int8',fill_value=1) for i in range(513,513 + 128)})
column_types_second = {str(i):'float64' for i in range(641,641 + 128)}
column_types_real.update(column_types_second)
column_names_no_amps = [str(i) for i in range(1,513)].append([str(i) for i in range(641,641 + 128)])
column_names_with_amps = [str(i) for i in range(1,641 + 128)]
column_types_synth = {str(i):'int8' for i in range(1,513)} #pd.SparseDtype(dtype='int8',fill_value=0) for i in range(1,513)}
column_types_synth.update({str(i):'int8' for i in range(513,513 + 128)}) #pd.SparseDtype(dtype='int8',fill_value=1) for i in range(513,513 + 128)})
column_types_second = {str(i):'float64' for i in range(641,641 + 128)}
column_types_synth.update(column_types_second)

column_types = column_types_real
BASE_FILE_SIZE = 1000
def split_data(path):
    os.mkdir(path + 'train data/')
    os.mkdir(path + 'val data/')
    os.mkdir(path + 'test data/')
    train_data = pd.read_parquet(path + 'train.parquet').astype(column_types)
    _ = [train_data.iloc[i * BASE_FILE_SIZE:i * BASE_FILE_SIZE + BASE_FILE_SIZE,:].to_parquet(path + 'train data/' + str(i) + '.parquet') for i in range(0,int(train_data.shape[0] / BASE_FILE_SIZE))]        
    val_data = pd.read_parquet(path + 'train.parquet').astype(column_types)
    _ = [val_data.iloc[i * BASE_FILE_SIZE:i * BASE_FILE_SIZE + BASE_FILE_SIZE,:].to_parquet(path + 'val data/' + str(i) + '.parquet') for i in range(0,int(val_data.shape[0] / BASE_FILE_SIZE))]        
    test_data = pd.read_parquet(path + 'train.parquet').astype(column_types)
    _ = [test_data.iloc[i * BASE_FILE_SIZE:i * BASE_FILE_SIZE + BASE_FILE_SIZE,:].to_parquet(path + 'test data/' + str(i) + '.parquet') for i in range(0,int(test_data.shape[0] / BASE_FILE_SIZE))]        

def test_collate_fn(batch):
    x_list, y_list = [], []
    for _x,_y in batch:
        x_list.append(_x)
        y_list.append(_y)
    
    new_x_list = torch.as_tensor(np.reshape(np.asarray(dask.compute(x_list),(configuration.batch_size, -1))))
    new_y_list = torch.as_tensor(np.reshape(np.asarray(dask.compute(y_list),(configuration.batch_size, -1))))
    if new_y_list.shape[1] == 128:
        new_y_list = F.pad(new_y_list,(128,0),'constant',1)

    return new_x_list, new_y_list

def get_last_step(base_path = configuration.path_to_checkpoints):
    dir = os.listdir(base_path)
    last_step = -1
    last_step_name = 'curriculum -1'
    for file in dir:
        if 'curriculum' in file and int(file[11:12]) > last_step:
            last_step = int(file[11:12])
            last_step_name = file
    return (last_step,last_step_name)

def convert_csv_to_parquet(path):
    #size = os.path.getsize(path + '.csv')
    #data = dd.read_csv(path +'.csv',header = None,index_col = None).to_parquet(path + '.csv')
    if 'curriculum' in path or configuration.mode == 'real':
        column_types = column_types_real
    else:
        column_types = column_types_synth
    column_types.update({'0':'int32'})
    if os.path.exists(path + '.parquet'):
        data = dd.read_parquet(path + '.parquet').astype(column_types)
    elif os.path.exists(path + '/base data/'):
        data = dd.read_parquet(path + '/base data/*.parquet').astype(column_types) #.set_index('0')
        print(data.columns,data.dtypes,'read data')
        try:
            data = data.set_index(data.columns[0]) #.astype(column_types)    
        except Exception as e: 
            print(e)
            data = data.set_index(data.columns[0]) #.astype(column_types)
        column_types.pop('0')
        print('passed index setting')
    else:
        data = dd.read_csv(path +'.csv',dtype =  column_types).set_index('0').astype(column_types)
    print(data.dtypes,data.index)
    n = len(data)
    perm = np.random.permutation(n)
    train_idx = np.sort(perm[:round(0.7*n)])
    val_idx = np.sort(perm[round(0.7*n):round(0.8*n)])
    test_idx = np.sort(perm[round(0.8*n):])
    train_data = data.loc[train_idx,:].to_parquet(path + '/train.parquet')
    val_data = data.loc[val_idx,:].to_parquet(path + '/val.parquet')
    test_data = data.loc[test_idx,:].to_parquet(path + '/test.parquet')
    #data.set_index('0',sorted = True).to_parquet(path + '.parquet')
    #data.to_pickle(path +'.gzip')

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
    Frequancy = 4.464e6
    
    batch_size = delays.shape[0]
    batch = np.zeros((batch_size,configuration.IMG_Y, 200))
    for i in range(batch_size):
        #delay = matlab.double(delays[i].tolist())
        transducer_phase = np.asarray(delays[i])
        transducer_phase = np.unwrap(transducer_phase)
        transducer_phase = transducer_phase-min(transducer_phase)
        transducer_phase = transducer_phase/max(transducer_phase)
        Transducer_delay_Wavelength = transducer_phase/(2*np.pi)
        Transducer_delay_Wavelength = Transducer_delay_Wavelength /Frequancy
        delay = matlab.double(Transducer_delay_Wavelength.tolist())
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
def extract_peaks(input):
    pass

def convulve_with_sinc(input):
    pitch = configuration.pitch
    x_aranged = np.arange(-256 * pitch ,256 * pitch,pitch)
    #sinc = np.abs(np.sinc(1045.61 * x_aranged)) ** 2
    sinc = np.abs(np.sinc(1045.61 * x_aranged))
    sinc[:256 - 50] = 0
    sinc[256 + 50:] = 0
    if input.ndim == 2:
        x_conv = np.apply_along_axis(np.convolve,1,input,sinc,mode = 'same')
    else:
        x_conv = np.convolve(input,sinc,mode = 'same')
    x_conv = np.abs(x_conv)
    return x_conv

class ModelManager():
    def __init__(self) -> None:
        try: 
            with open(configuration.path_to_prev_results,'r') as prev_res:
                self.prev_results = json.load(prev_res)
        except IOError:
            self.prev_results = {}
    
    def save_checkpoint(self,num_focus,net,optimizer,train_loss,val_loss, base_training_params):
        if configuration.mode == 'synth':
            if configuration.out_size == 128:
                name = f'{num_focus},delays'
            else:
                name = f'{num_focus},delays and amps'
            if name in self.prev_results and min(val_loss) > self.prev_results[name]:
                return
            self.prev_results[name] = min(val_loss)
            state = {
                'net' : net.state_dict(),
                'train loss': train_loss,
                'val loss': val_loss,
                'optimizer state': optimizer.state_dict(),
                'base training params' : base_training_params,
                'out_size' : configuration.out_size,
                'mode' :  configuration.mode
            }
            if configuration.out_size == 128:

                path_to_save = os.path.join(configuration.path_to_checkpoints,f'net for {num_focus} focuses only delays.pt')
            else:
                path_to_save = os.path.join(configuration.path_to_checkpoints,f'net for {num_focus} focuses amps and delays.pt')

            torch.save(state,path_to_save)
            with open(configuration.path_to_prev_results, "w") as write_file:
                json.dump(self.prev_results, write_file)
        else:
            step = base_training_params['step num']
            name = 'curriculum ' + str(step)

            if name in self.prev_results and min(val_loss) > self.prev_results[name]:
                return
            self.prev_results[name] = min(val_loss)
            state = {
                'net' : net.state_dict(),
                'train loss': train_loss,
                'val loss': val_loss,
                'optimizer state': optimizer.state_dict(),
                'base training params' : base_training_params,
                'out_size' : configuration.out_size,
                'step num' : step,
                'mode' :  configuration.mode
            }
            path_to_save = os.path.join(configuration.path_to_checkpoints,f'{name}.pt')
            torch.save(state,path_to_save)
            
            with open(configuration.path_to_prev_results, "w") as write_file:
                json.dump(self.prev_results, write_file)

    def load_checkpoint(self,num_focus = 10,step_num = 0):
        (step,name) = get_last_step()
        print(step,name,step_num)
        if configuration.mode == 'synth' or step_num == -1:
            if configuration.out_size == 128:
                name = f'{num_focus},delays'
            else:
                name = f'{num_focus},delays and amps'
            if name in self.prev_results:
                if configuration.out_size == 128:
                    path_to_load = os.path.join(configuration.path_to_checkpoints,f'net for {num_focus} focuses only delays jit.pt')
                    if not os.path.isfile(path_to_load):
                        path_to_load = os.path.join(configuration.path_to_checkpoints,f'net for {num_focus} focuses only delays.pt')
                else:
                    path_to_load = os.path.join(configuration.path_to_checkpoints,f'net for {num_focus} focuses amps and delays jit.pt')
                    if not os.path.isfile(path_to_load):
                        path_to_load = os.path.join(configuration.path_to_checkpoints,f'net for {num_focus} focuses amps and delays.pt')
                checkpoint = torch.load(path_to_load,map_location = configuration.device)
                print(path_to_load)
                return checkpoint
        else:
            step = step_num if step_num != -1 else step
            name = 'curriculum ' + str(step)
            path_to_load = os.path.join(configuration.path_to_checkpoints,f'{name}.pt')
            checkpoint = torch.load(path_to_load,map_location = configuration.device)
            print(path_to_load)
            return checkpoint
    
    def load_model(self,num_focus,step_num,drop = 0,k = 512, reduce_conv = True,expansion_factor = 3, num_blocks = 9):
        checkpoint = self.load_checkpoint(num_focus,step_num)
        net = nets.multiResNet(drop = drop, k = k, reduce_conv = reduce_conv, expansion_factor=expansion_factor,num_blocks=num_blocks)
        net.load_state_dict(checkpoint['net'], strict= False)
        return net
        
    def convert_model_to_onnx(self,num_focus):
        if configuration.out_size == 128:
            path_to_save = os.path.join(configuration.path_to_checkpoints,f'net for {num_focus} focuses only delays.onnx')
        else:
            path_to_save = os.path.join(configuration.path_to_checkpoints,f'net for {num_focus} focuses amps and delays.onnx')
        checkpoint = self.load_checkpoint(num_focus)
        torch_model = checkpoint['net']
        params = checkpoint['base training params']
        net = nets.multiResNet(drop = 0, k = params['k'], reduce_conv = params['reduce_conv'], expansion_factor=params['expansion_factor'],num_blocks=params['num_blocks'])
        net.load_state_dict(torch_model, strict= False)
        net = net.eval()
        x = torch.randn(configuration.batch_size,configuration.in_size,requires_grad=True)
        torch_out = net(x)
        torch.onnx.export(net,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  path_to_save,              # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input data'],   # the model's input names
                  output_names = ['output data'], # the model's output names
                  dynamic_axes={'input data' : {0 : 'batch_size'},    # variable length axes
                                'output data' : {0 : 'batch_size'}},
                verbose=True)


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



class dataSingleton():
    __instance = None
    @staticmethod 
    def getInstance():
      """ Static access method. """
      if dataSingleton.__instance == None:
        dataSingleton()
      return dataSingleton.__instance
    def __init__(self):
      """ Virtually private constructor. """
      if dataSingleton.__instance != None:
        raise Exception("This class is a singleton!")
      else:
        dataSingleton.__instance = self

    def load_data(self,base_path):
        if 'amps' in base_path:
            self.data_train = pd.read_parquet(base_path + '/train.parquet').astype(column_types)
            self.data_val = pd.read_parquet(base_path + '/val.parquet').astype(column_types)
            self.data_test = pd.read_parquet(base_path + '/train.parquet').astype(column_types)

        else:
            self.data_train = pd.read_parquet(base_path + '/train.parquet').astype(column_types)
            if not 'amps' in base_path:
                self.data_train = self.data_train.drop(columns=[str(i) for i in range(513,513 + 128)])      
            self.data_val = pd.read_parquet(base_path + '/val.parquet').astype(column_types)
            if not 'amps' in base_path:
                self.data_val = self.data_val.drop(columns=[str(i) for i in range(513,513 + 128)])
            self.data_test = pd.read_parquet(base_path + '/train.parquet').astype(column_types)
            if not 'amps' in base_path:
                self.data_test = self.data_test.drop(columns=[str(i) for i in range(513,513 + 128)])
    def get_len_data(self,type_of_data):
        if type_of_data == 'train':
            return len(self.data_train)
        elif type_of_data == 'val':
            return len(self.data_val)
        return len(self.data_test)
        
    def get_data(self,type_of_data):
        if type_of_data == 'train':
            return self.data_train
        elif type_of_data == 'val':
            return self.data_val
        return self.data_test
    def get_row(self,type_of_data,idx):
        if type_of_data == 'train':
            return self.data_train.iloc[idx,:].values
        elif type_of_data == 'val':
            return self.data_val.iloc[idx,:].values
        return self.data_test.iloc[idx,:].values

class baseDataSet(Dataset):
    def __init__(self,indexes = None,source_data = None,seq_length = 50 ,file_path = "C:/Users/drors/Desktop/code for thesis/data.csv",transforms = None, mode = 'both',
                    from_file = False, from_singleton = False,type_of_data = 'train', lazy = False,return_both = False,return_integers = False):

        #self.data = pd.read_csv(csv_file,header = None, index_col = 0,skiprows = )
        self.from_file = from_file
        self.return_integers = return_integers
        self.from_singleton = False
        self.lazy = lazy
        self.return_both = return_both
        if not from_file:
            self.data = source_data.loc[indexes,:].compute()
        elif from_singleton:
            self.from_singleton = from_singleton
            self.type_of_data = type_of_data
            self.singleton = dataSingleton.getInstance()
        elif from_file and not lazy:
            self.data = pd.read_parquet(file_path).astype(column_types)
            if not 'amps' in file_path:
                self.data = self.data.drop(columns=[str(i) for i in range(513,513 + 128)])
                self.droped = False 
        elif from_file and lazy:
            if not 'amps' in file_path:
                self.droped = False
        else:
            if True: #os.path.getsize(file_path) < 1e6 and False:
                self.data = dd.read_parquet(file_path).astype(column_types)
                if not 'amps' in file_path:
                    self.data = self.data.drop(columns=[str(i) for i in range(513,513 + 128)])
                    self.droped = True
                self.lookup_table = {i:j  for j,i in zip(self.data.index,range(len(self.data)))}
            else:
                self.data = None
                data = dd.read_parquet(file_path)
                self.num_rows_partition = np.asarray(data.map_partitions(len).compute())
                self.num_rows = len(data)
                self.chunk_size = self.num_rows_partition[0]
                #self.lookup = 


        #new_index = np.arange(self.data.shape[0])
        #self.data['0'] = pd.Series(new_index)
        #self.data = self.data.set_index('0')
        self.path = file_path
        self.measure = nn.CosineSimilarity()
        self.seq_length = seq_length
        self.mode = mode
        self.converters = {}
        self.transform_x = False
        self.transform_y = False


    def __len__(self):
        if self.from_singleton:
            return self.singleton.get_len_data(self.type_of_data)
        if self.lazy == True:
            return len(os.listdir(self.path)) * BASE_FILE_SIZE
        if type(self.data) != None:
            return len(self.data)
        else:
            return self.num_rows
    def __getitem__(self,idx):
        if self.from_singleton:
            row = self.singleton.get_row(self.type_of_data,idx)
        elif self.from_file and not self.lazy:
            row = self.data.iloc[idx,:].values
        elif self.from_file and self.lazy:
            file_name = self.path + str(idx // BASE_FILE_SIZE) + '.parquet'
            row = pd.read_parquet(file_name)
            #print(row.shape, idx % BASE_FILE_SIZE, idx // BASE_FILE_SIZE)
            row = row.iloc[idx % BASE_FILE_SIZE,:].values
        elif type(self.data) != None:
            idx = self.lookup_table[idx]
            row = self.data.loc[idx,:] #.compute()
        else:
            chunk_id = np.where(self.num_rows_partition )
            chunk_id = idx // self.chunk_size
            row_idx = idx % self.chunk_size
            if 'amps' in self.path:
                data = pd.read_parquet(self.path + '/' + str(chunk_id) + '.parquet')
            else:
                data = pd.read_parquet(self.path + '/part.' + str(chunk_id) + '.parquet', columns=column_names_no_amps)
            row = data.iloc[row_idx]

        #row = np.asarray(row.values)
        #row = torch.as_tensor(np.asarray(row)).flatten()
        if self.from_singleton or self.from_file:
            x = row[:configuration.in_size]
            if configuration.mode == 'synth':
                base = x
                x = convulve_with_sinc(x)
                #pass
            else:
                pass
                peaks,info = find_peaks(x,height=0.5)
                #print(x.dtype,np.zeros(x.shape).shape,info['peak_heights'],peaks,info)
                base = np.zeros(x.shape)
                if self.return_integers:
                    base[peaks] = 1
                else:
                    base[peaks] = info['peak_heights']
                x = convulve_with_sinc(base)
            x = torch.as_tensor(x)
            y = torch.as_tensor(row[configuration.in_size:])
            if self.droped:
                y = F.pad(y,(128,0),'constant',1)
        else:
            x = row[:configuration.in_size]
            y = row[configuration.in_size:]
        
        # x = self.data.iloc[idx,:configuration.in_size]
        # y = self.data.iloc[idx,configuration.in_size:]
        # if len(y) == 2 * 128:
        #     y = torch.tensor(y.values).split(128,1)
        # else:
        #     y = torch.tensor(y.values)
        # x = np.abs(x.values)
        # x = torch.tensor(x)
        # y = torch.tensor(y.values)
        # if self.transform_x:
        #     x = self.transform_x.transform(x.reshape(1,-1)).reshape(-1)
        #     y = self.transform_y.transform(y.reshape(1,-1)).reshape(-1)
        if self.return_both == True:
            return x,y,base
        return x,y


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
    def __init__(self,seq_length = 50,csv_file = "C:/Users/drors/Desktop/code for thesis/data.csv",orig = "C:/Users/drors/Desktop/code for thesis/data original.csv",
                    num_focuses = 0):
        #self.eng = []
        #for i in range(configuration.num_workers):
        #    self.eng.append(matlab.engine.start_matlab())
        #    self.eng[i].init_field(nargout=0)

        if num_focuses != 0:
            if configuration.out_size == 256:
                path = f'{num_focuses} focus data delays amps'
            else:
                path = f'{num_focuses} focus data delays'
            csv_file = os.path.join(configuration.base_path_datasets,path)
            orig = os.path.join(configuration.base_path_datasets,path)
        elif configuration.mode != 'synth':
            (step,name) = get_last_step(configuration.base_path_datasets)
            print(step,name)
            path = '10 focus data delays'
            if step != -1:
                path = name
            csv_file = os.path.join(configuration.base_path_datasets,path)
            orig = os.path.join(configuration.base_path_datasets,path)
            
                  

        self.csv_file = csv_file
        if orig == configuration.path_combined:
            self.mode = 'both'
        else:
            self.mode = 'single'
        self.orig = orig
        self.train,self.val,self.test = self.create_datasets()
        self.seq_length = seq_length
        #client = Client()  # start distributed scheduler locally.  Launch dashboard
        # self.svd = KernelPCA(n_components=configuration.in_size, kernel='poly')
        # self.train.data.iloc[:,:configuration.in_size] = self.svd.fit_transform(self.train.data.iloc[:,:configuration.IMG_X])
        # self.val.data.iloc[:,:configuration.in_size] = self.svd.transform(self.val.data.iloc[:,:configuration.IMG_X])
        # self.test.data.iloc[:,:configuration.in_size] = self.svd.transform(self.test.data.iloc[:,:configuration.IMG_X])
    # #     self.x_scalar = MinMaxScaler()
    #     self.y_scalar = MinMaxScaler()
    #     self.y_scalar = MinMaxScaler()
    #     self.x_scalar.fit(self.train.data.iloc[:,:configuration.IMG_X])
    #     self.y_scalar.fit(self.train.data.iloc[:,configuration.IMG_X:])
    #     self.train.set_transform(self.x_scalar,self.y_scalar)
    #     self.test.set_transform(self.x_scalar,self.y_scalar)
    #     self.val.set_transform(self.x_scalar,self.y_scalar)
    def create_datasets(self):
        if not os.path.exists(self.orig + '/train.parquet'):
            convert_csv_to_parquet(self.orig)
        #if not os.path.exists(self.orig + '/train data/0.parquet'):
        #    split_data(self.orig + '/')
        #singleton = dataSingleton.getInstance()
        #singleton.load_data(self.orig)
        #train_dataset = baseDataSet(from_singleton=True,type_of_data='train', from_file=True)
        #val_dataset = baseDataSet(from_singleton=True,type_of_data='val', from_file=True)
        #test_dataset = baseDataSet(from_singleton=True,type_of_data='test', from_file=True)
        train_dataset = baseDataSet(file_path=self.orig + '/train.parquet',from_file= True, lazy=False)
        val_dataset = baseDataSet(file_path=self.orig + '/val.parquet',from_file= True, lazy=False)
        test_dataset = baseDataSet(file_path=self.orig + '/test.parquet',from_file=True, lazy=False)
        #train_dataset = baseDataSet(train_idx,data,mode = self.mode)
        #val_dataset = baseDataSet(val_idx,data,mode = self.mode)
        #test_dataset = baseDataSet(test_idx,data,mode = self.mode)
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
        # self.x_scalar.fit(self.train.data.iloc[:,:configuration.IMG_X])
        # self.y_scalar.fit(self.train.data.iloc[:,configuration.IMG_X:])
        # self.train.set_transform(self.x_scalar,self.y_scalar)
        # self.test.set_transform(self.x_scalar,self.y_scalar)
        # self.val.set_transform(self.x_scalar,self.y_scalar)
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




from multiprocessing.reduction import ForkingPickler, AbstractReducer
import multiprocessing as torch_mp
class ForkingPickler2(ForkingPickler):
    def __init__(self, *args):
        if len(args) > 1:
            args[1] = 2
        else:
            args.append(2)
        super().__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=4):
        return ForkingPickler.dumps(obj, protocol)


def dump(obj, file, protocol=4):
    ForkingPickler2(file, protocol).dump(obj)


class Pickle2Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler2
    register = ForkingPickler2.register
    dump = dump



ctx = torch_mp.get_context()
ctx.reducer = Pickle2Reducer()
