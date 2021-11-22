import torch

#run time configurations
num_workers = 4
create_amount = 200
batch_size = 500
IMG_X = 128
pitch = 0.218e-3
frequency = 4.464e6
v = 1490
Wavelength = v/frequency
IMG_Y = 300
depth = 40e-3
dz = (80e-3 - 10e-3)/IMG_Y
SEQ_LENGTH = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_size = 256 
path_combined = "C:/Users/DrorSchein/Desktop/thesis/thesis/data advanced.csv"
path_gs = "C:/Users/DrorSchein/Desktop/thesis/thesis/data gs.csv"
path_gs_orig = "C:/Users/DrorSchein/Desktop/thesis/thesis/data gs - Copy.csv"