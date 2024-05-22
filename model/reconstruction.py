import torch
import math
from .shutils import ShellComputeSignal
import numpy as np

class Reconstruction(torch.nn.Module):
    """Building Block for spherical harmonic convolution with a polar filter
    """

    def __init__(self, polar_filter_equi, polar_filter_inva, shellSampling,full_basis,sh_degree):
        """Initialization.
        Args:
            polar_filter (:obj:`torch.Tensor`): [in_channel x S x L] Polar filter spherical harmonic coefficients
            polar_filter_inva (:obj:`torch.Tensor`): [in_channel x S x 1] Polar filter spherical harmonic coefficients
            shellSampling (:obj:`sampling.ShellSampling`): Input sampling scheme
        """
        super(Reconstruction, self).__init__()
        if polar_filter_equi is None:
            self.equi = False
        else:
            self.conv_equi = IsoSHConv(polar_filter_equi,full_basis,sh_degree)
            self.equi = True

        if polar_filter_inva is None:
            self.inva = False
        else:
            self.conv_inv = IsoSHConv(polar_filter_inva,full_basis,sh_degree)
            self.inva = True

        self.get_signal = ShellComputeSignal(shellSampling)

    def forward(self, x_equi_shc, x_inva_shc):
        """Forward pass.
        Args:
            x_equi_shc (:obj:`torch.tensor`): [B x equi_channel x C x X x Y x Z] Signal spherical harmonic coefficients.
            x_inva_shc (:obj:`torch.tensor`): [B x inva_channel x 1 x X x Y x Z] Signal spherical harmonic coefficients.
        Returns:
            :obj:`torch.tensor`: [B x V x X x Y x Z] Reconstruction of the signal
        """
        x_convolved_equi, x_convolved_inva = 0, 0
        if self.equi:
            x_convolved_equi_shc = self.conv_equi(x_equi_shc) # B x equi_channel x S x C x X x Y x Z
            x_convolved_equi = self.get_signal(x_convolved_equi_shc) # B x equi_channel x V x X x Y x Z
            x_convolved_equi = torch.sum(x_convolved_equi, axis=1) # B x V x X x Y x Z
        if self.inva:
            x_convolved_inva_shc = self.conv_inv(x_inva_shc) # B x inva_channel x S x 1 x X x Y x Z
            x_convolved_inva = self.get_signal(x_convolved_inva_shc) # B x inva_channel x V x X x Y x Z
            x_convolved_inva = torch.sum(x_convolved_inva, axis=1) # B x V x X x Y x Z
        # Get reconstruction
        x_reconstructed =  x_convolved_equi + x_convolved_inva

        return x_reconstructed,x_convolved_equi_shc


class IsoSHConv(torch.nn.Module):
    """Building Block for spherical harmonic convolution with a polar filter
    """

    def __init__(self, polar_filter,full_basis,sh_degree):
        """Initialization.
        Args:
            polar_filter (:obj:`torch.Tensor`): [in_channel x S x L] Polar filter spherical harmonic coefficients
        where in_channel is the number of tissue, S is the number of shell and L is the number of odd spherical harmonic order
        C is the number of coefficients for the L odd spherical harmonic order
        """
        super().__init__()
        # We need to multiply each coefficient order by sqrt(4pi/(4l+1)) and copy it 2*l+1 times
        polar_filter = self.construct_filter(polar_filter,full_basis,sh_degree) # 1 x in_channel x S x C
        self.register_buffer("polar_filter", polar_filter)

    def construct_filter(self, filter,full_basis,sh_degree):
        """Reformate the polar filter (scale and extand the coefficients).
        Args:
            polar_filter (:obj:`torch.Tensor`): [in_channel x S x L] Polar filter spherical harmonic coefficients
        Returns:
            :obj:`torch.tensor`: [1 x in_channel x S x C] Extanded polar filter spherical harmonic coefficients
        """

        L = filter.shape[2]
        # Scale by sqrt(4*pi/4*l+1))
        scale = torch.Tensor([math.sqrt(4*math.pi/(4*l+1)) for l in range(L)])[None, None, :] # 1 x 1 x L
        filter = scale*filter # in_channel x S x L
        # Repeat each coefficient 4*l+1 times
        repeat = torch.Tensor([int(4*l+1) for l in range(L)]).type(torch.int64) # L
        # print('repeat',repeat)    
        filter = filter.repeat_interleave(repeat, dim=2) # in_channel x S x C
        # Add the first dimension for multiplication convenience
        filter = filter[None, :, :, :] # 1 x in_channel x S x C
        number = 0
        even_indices = []

        if full_basis == True:
            number = int((sh_degree +1)*(sh_degree +1))
            n_range_full = np.arange(0,sh_degree+1,dtype=int)
            n_list_full = np.repeat(n_range_full,n_range_full*2 +1)
            even_indices = np.where(n_list_full%2 ==0)[0].tolist()
        if full_basis == False:
      
            number = int((sh_degree +1)*((sh_degree +2)/2))
            n_range_hemi = np.arange(0,sh_degree+1,2,dtype=int)
            n_list_hemi = np.repeat(n_range_hemi,n_range_hemi*2 +1)
            even_indices = np.where(n_list_hemi%2 ==0)[0].tolist()
       
        filter_full = torch.zeros(filter.size()[0],filter.size()[1],filter.size()[2],number)
        filter_full[..., even_indices] = filter
  
        return filter_full

    def forward(self, x):
        """Forward pass.
        Args:
            x (:obj:`torch.tensor`): [B x in_channel x C x X x Y x Z] Signal spherical harmonic coefficients.
        Returns:
            :obj:`torch.tensor`: [B x in_channel x S x C x X x Y x Z] Spherical harmonic coefficient of the output
        """
        
        x = x[:, :, None, :] # B x in_channel x 1 x C x X x Y x Z
        x = x*self.polar_filter[:, :, :, :, None, None, None] # B x in_channel x S x C x X x Y x Z
        return x
