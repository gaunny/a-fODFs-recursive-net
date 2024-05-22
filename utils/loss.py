import torch
from dipy.core.ndindex import ndindex
from dipy.direction.peaks import peak_directions
from dipy.reconst.shm import sh_to_sf
import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
import math
from itertools import combinations
from dipy.reconst.recspeed import local_maxima

    
class Loss_smoothness(torch.nn.Module):
    def __init__(self, norm, sigma=None):
        super(Loss_smoothness, self).__init__()
        if norm not in ['cauchy']:
            raise NotImplementedError('Expected mse but got {}'.format(norm))
        self.norm = norm
        self.sigma = sigma
    def forward(self, fodf_shc, sh_degree):
        n_range_full = np.arange(0,sh_degree+1,dtype=int)
        n_list_full = np.repeat(n_range_full,n_range_full *2 +1)
        n_list_full = torch.from_numpy(n_list_full)
        odd_index = torch.where(n_list_full%2 != 0)[0]
        high_freq_degree = sh_degree/2 
       
        n_range_full_low = np.arange(0,high_freq_degree+1,dtype=int)
        n_list_full_low = np.repeat(n_range_full_low,n_range_full_low *2 +1)
        n_list_full_low = torch.from_numpy(n_list_full_low)
        odd_index_low_freq = torch.where(n_list_full_low%2 != 0)[0]
        odd_index_high_freq = odd_index[~torch.isin(odd_index,odd_index_low_freq)]
        
        if self.norm == 'cauchy':
            odd_fodf_shc_high = fodf_shc[:,:,odd_index_high_freq,:,:,:]
            full_zeros = torch.zeros(odd_fodf_shc_high.size()).cuda()
            loss_high = 2 * torch.log(1 + ((odd_fodf_shc_high - full_zeros)**2 / (2*self.sigma)))
            loss = loss_high 
        loss =loss.mean()
        return loss
    
class Loss_oddsensitive(torch.nn.Module):
    def __init__(self, norm):
        super(Loss_oddsensitive, self).__init__()
        if norm not in ['L1']:
            raise NotImplementedError('Expected mse but got {}'.format(norm))
        self.norm = norm
    
    def forward(self, fodf_shc, sh_degree):
        n_range_full = np.arange(0,sh_degree+1,dtype=int)
        n_list_full = np.repeat(n_range_full,n_range_full *2 +1)
        n_list_full = torch.from_numpy(n_list_full)
        high_freq_degree = sh_degree/2 
        n_range_full_low = np.arange(0,high_freq_degree+1,dtype=int)
        n_list_full_low = np.repeat(n_range_full_low,n_range_full_low *2 +1)
        n_list_full_low = torch.from_numpy(n_list_full_low)
        odd_index_low_freq = torch.where(n_list_full_low%2 != 0)[0]
        even_index_freq = torch.where(n_list_full_low%2 == 0)[0]
        
        if self.norm == 'L1':
            odd_fodf_shc_low = fodf_shc[:,:,odd_index_low_freq,:,:,:]
            even_fodf_shc = fodf_shc[:,:,even_index_freq,:,:,:]
            loss_low = torch.abs(torch.mean(torch.abs(even_fodf_shc)) - torch.mean(torch.abs(odd_fodf_shc_low)))
            loss = loss_low
        loss =loss.mean()
        return loss
    
class Loss(torch.nn.Module):
    def __init__(self, norm):
       
        super(Loss, self).__init__()
        if norm not in ['L2', 'L1']:
            raise NotImplementedError('Expected L1, L2 but got {}'.format(norm)) 
        self.norm = norm

    def forward(self, img1, img2, mask, wts=None):
        img1 = img1[mask>0]
        img2 = img2[mask>0]
        if self.norm == 'L2':
            out = (img1 - img2)**2
        elif self.norm == 'L1':
            out = torch.abs(img1 - img2)
        else:
            raise ValueError('Expected L1, L2, cauchy, welsh, geman but got {}'.format(self.norm))

        if wts is not None:
            out = out * wts
        loss = out.mean()
        return loss