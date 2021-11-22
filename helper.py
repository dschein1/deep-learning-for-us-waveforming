import numpy as np
import configuration
import torch
import torch.nn.functional as F
from torch import nn
#taken from pytorch forums
def unwrap(phi, dim=-1):
    assert dim is -1 #, unwrap only supports dim=-1 for now’
    dphi = diff(phi, same_size=True)
    dphi_m = ((dphi+np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m==-np.pi)&(dphi>0)] = np.pi
    phi_adj = dphi_m-dphi
    phi_adj[dphi.abs()<np.pi] = 0
    return phi + phi_adj.cumsum(dim)

def diff(x, dim=-1, same_size=True):
    assert dim is -1 #, ‘diff only supports dim=-1 for now’
    if same_size:
        return F.pad(x[:,1:]-x[:,:-1], (1,0))
    else:
        return x[:,1:]-x[:,:-1]

def fsp_x_near(source,dz = configuration.depth):
    source =  F.pad(source,(512 - 64,512 - 64),'constant',0)# torch.zeros(source.size()[0],configuration.IMG_X).to(configuration.device) 
    #new_source[:,int(configuration.IMG_X/2) - int(source.size()[1]/2):int(configuration.IMG_X/2) + int(source.size()[1]/2)] = source
    #print(new_source.shape)
    FFT_inp = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(source)))
    du=1/configuration.pitch/1024
    #xx = torch.arange(-int(configuration.IMG_X/2),int(configuration.IMG_X/2)) #.to(configuration.device)
    xx = torch.arange(-int(1024/2),int(1024/2), device= configuration.device)
    PS=torch.exp(torch.tensor(1j, device= configuration.device)*2*np.pi*dz/configuration.Wavelength*(torch.sqrt(1-torch.pow((xx*du*configuration.Wavelength),2))))
    source = torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(PS * FFT_inp)))
    abs_source= source.abs()
    return abs_source # / torch.max(abs_source,dim = 0).values

def loss_gs(source,target,base_loss = nn.MSELoss()):
    if source.size(1) == 256:
        source = source[:,:128] * torch.exp(1j * source[:,128:])
    new_source = fsp_x_near(source)
    if target.size(1) == 256:
        target = target[:,:128] * torch.exp(1j * target[:,128:])
        target = fsp_x_near(target)
    elif target.size(1) == 128:
        new_source = new_source[:,512 - 64:512 + 64]
    if configuration.device == 'cpu':
        #MSE
        loss = torch.mean((new_source - target) ** 2)
    else:
        loss = base_loss(new_source,target)
    return loss

#not implemented fully
def extract_amp_delays(transducer):
    amp = torch.abs(transducer)
    amp = amp / torch.max(amp,dim = 1)
    phase = torch.angle(transducer)