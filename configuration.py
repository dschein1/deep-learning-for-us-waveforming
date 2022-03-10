import torch

#run time configurations
num_workers = 4
create_amount = 200
batch_size = 100
IMG_X = 128
in_size = 512
pitch = 0.218e-3
frequency = 4.464e6
v = 1490
Wavelength = v/frequency
IMG_Y = 300
depth = 40e-3
dz = (80e-3 - 10e-3)/IMG_Y
SEQ_LENGTH = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_size = 128
base_path_datasets = "C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/"
path_to_checkpoints = 'C:/Users/DrorSchein/Desktop/thesis/thesis/checkpoints'
path_to_channel = 'C:/Users/DrorSchein/Desktop/thesis/thesis/py to matlab/'
path_to_prev_results = 'C:/Users/DrorSchein/Desktop/thesis/thesis/previous results.json'
path_double_focus_delays = "C:/Users/DrorSchein/Desktop/thesis/thesis/double focus only delays.csv"
path_triple_focus_delays = "C:/Users/DrorSchein/Desktop/thesis/thesis/3 focus data delays only.csv"
path_combined = "C:/Users/DrorSchein/Desktop/thesis/thesis/data advanced.csv"
path_gs = "C:/Users/DrorSchein/Desktop/thesis/thesis/data gs.csv"
path_gs_orig = "C:/Users/DrorSchein/Desktop/thesis/thesis/data gs - Copy.csv"
path_double_focus = "C:/Users/DrorSchein/Desktop/thesis/thesis/data advanced 2 focus.csv"
path_single_focus = "C:/Users/DrorSchein/Desktop/thesis/thesis/data small single focus.csv"
path_single_focus_only_delay_expanded = "C:/Users/DrorSchein/Desktop/thesis/thesis/single focus only delays.csv"
path_single_focus_only_delay = "C:/Users/DrorSchein/Desktop/thesis/thesis/single focus only delays small range.csv"

