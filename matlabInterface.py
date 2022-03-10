import nets
import helper
import configuration
import data
import numpy as np
import torch
from torch.utils.data import DataLoader
class manager():
    def __init__(self) -> None:
        self.num_focus = 10
        self.modelManager = data.ModelManager()
        self.checkpoint = self.modelManager.load_checkpoint(self.num_focus)
        params = self.checkpoint['base training params']
        net = nets.multiResNet(drop = 0, k = params['k'], reduce_conv = params['reduce_conv'], expansion_factor=params['expansion_factor'],num_blocks=params['num_blocks'])
        net.load_state_dict(self.checkpoint['net'], strict= False)
        self.model = net.to(configuration.device).eval()
        self.dataManager = data.datasetManager(num_focuses=self.num_focus)
        self.test_data = self.dataManager.get_datasets()[2]


    
    def evaluate(self,input):
        as_numpy = np.asarray(input)
        tensor = torch.as_tensor(as_numpy, device=configuration.device)
        output = self.model(tensor.float())
        output = helper.create_wave_for_propagation(output)
        return np.asarray(output.detach().abs().cpu().numpy())
        

    def get_batch_from_dataset(self,amount = 8):
        loader = DataLoader(self.test_data,batch_size=amount,shuffle=True)
        x,y = next(iter(loader))
        y = helper.create_wave_for_propagation(y)
        output_from_net = self.model(x.to(configuration.device).float())
        return (np.asarray(x.detach().abs().cpu().numpy()), np.asarray(output_from_net.detach().abs().cpu().numpy()), np.asarray(y.detach().abs().cpu().numpy()))

def create_interface():
    return manager()
    