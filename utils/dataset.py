from torch.utils.data import Dataset
import nibabel as nib
import torch
import os
import numpy as np
from itertools import groupby

# Normalize the dmri data by dividing b0 data
def normalize(data_path,dmri_data,bvals,affine):
    if os.path.exists(f'{data_path}/normalized_data.nii.gz'):
        print(f"{f'{data_path}/normalized_data.nii.gz'} already exists.")
    else:
        print(f"{f'{data_path}/normalized_data.nii.gz'} doesn't exist.")
        index = np.where(bvals==0)[0]
        b0_data_all = dmri_data[:,:,:,index]
        b0_data = np.mean(b0_data_all,axis=-1)
        normalized_data = np.zeros((dmri_data.shape))
        for i in range(dmri_data.shape[-1]):
            normalized_data[:,:,:,i] = dmri_data[:,:,:,i] / b0_data
        # Avoid voxels of nan, inf, -inf in normalized images
        normalized_data = np.nan_to_num(normalized_data, nan=0)
        normalized_data = np.where(normalized_data == np.inf, 0, normalized_data)  
        normalized_data = np.where(normalized_data == -np.inf, 0, normalized_data) 
        normalized_data = normalized_data.astype(np.float32)
        new_img = nib.Nifti1Image(normalized_data, affine=affine)
        nib.save(new_img, f'{data_path}/normalized_data.nii.gz')


class DMRIDataset(Dataset):
    def __init__(self, data_path,dmri_path,mask_path, bvec_path, bval_path, patch_size, concatenate=False):
        self.patch_size = patch_size
        try:
            self.data = nib.load(dmri_path)
        except:
            self.data = nib.load(dmri_path+'.gz')
        self.affine = self.data.affine
        vectors = np.loadtxt(bvec_path)
        bvals = np.loadtxt(bval_path)
        self.normalized_data = nib.load(f'{data_path}/normalized_data.nii.gz')
        # Load the data with b!=0
        notb0_index = np.where(bvals!=0)[0]
        dmri_shell_data = self.normalized_data.get_fdata()[:,:,:,notb0_index]
        self.data = torch.Tensor(dmri_shell_data) # Load image X x Y x Z x V
        vectors = vectors[:,notb0_index]
        bvals = bvals[notb0_index]
        # Load mask
        if os.path.isfile(mask_path) or os.path.isfile(mask_path+'.gz'):
            try:
                self.mask = torch.Tensor(nib.load(mask_path).get_fdata()) # X x Y x Z
            except:
                self.mask =  torch.Tensor(nib.load(mask_path+'.gz').get_fdata())
        else:
            self.mask = torch.ones(self.data.shape[:-1])
        # 0-pad image and mask
        self.data = torch.nn.functional.pad(self.data, (0, 0, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2), 'constant', value=0)
        self.mask = torch.nn.functional.pad(self.mask, (patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2), 'constant', value=0)
        # Permute
        self.data = self.data.permute(3, 0, 1, 2)
        # Save the non-null index of the mask
        ind = np.arange(self.mask.nelement())[torch.flatten(self.mask) != 0]
        self.x, self.y, self.z = np.unravel_index(ind, self.mask.shape)
        self.N = len(self.x)
        
        if vectors.shape[0] == 3:
            vectors = vectors.T
        assert bvals.shape[0] == vectors.shape[0]
        assert vectors.shape[1] == 3
        vectors[:, 0] = -vectors[:, 0] # (should use the affine tranformation matrix instead)
        self.vec = np.unique(vectors[bvals!=0], axis=0)
        self.b = np.unique(bvals)
        self.normalization_value = np.ones(self.b.shape[0])
        self.patch_size_output = patch_size
        self.patch_size_input = patch_size
        self.concatenate = concatenate
        if concatenate:
            self.patch_size_output = 1

    def __len__(self):
        return int(self.N)

    def __getitem__(self, i):
        input = self.data[None, :, self.x[i] - (self.patch_size // 2):self.x[i] + (self.patch_size // 2) + (self.patch_size%2), self.y[i] - (self.patch_size // 2):self.y[i] + (self.patch_size // 2) + (self.patch_size%2), self.z[i] - (self.patch_size // 2):self.z[i] + (self.patch_size // 2) + (self.patch_size%2)] # 1 x V x P x P x P
        if self.concatenate:
            input = torch.flatten(input[0], start_dim=-3) # V x P*P*P
            input = input.permute(1, 0)[:, :, None, None, None] #  P*P*P x V x 1 x 1 x 1
        output = self.data[:, self.x[i] - (self.patch_size_output // 2):self.x[i] + (self.patch_size_output // 2) + (self.patch_size_output%2), self.y[i] - (self.patch_size_output // 2):self.y[i] + (self.patch_size_output // 2) + (self.patch_size_output%2), self.z[i] - (self.patch_size_output // 2):self.z[i] + (self.patch_size_output // 2) + (self.patch_size_output%2)] # V x P x P x P
        mask = self.mask[self.x[i] - (self.patch_size_output // 2):self.x[i] + (self.patch_size_output // 2) + (self.patch_size_output%2), self.y[i] - (self.patch_size_output // 2):self.y[i] + (self.patch_size_output // 2) + (self.patch_size_output%2), self.z[i] - (self.patch_size_output // 2):self.z[i] + (self.patch_size_output // 2) + (self.patch_size_output%2)] # P x P x P
        return {'sample_id': i, 'input': input, 'out': output, 'mask': mask}


def load_data(data_path,sh_degree,wm,gm,csf):
    data = nib.load(f'{data_path}/dmri_data.nii.gz').get_fdata()
    affine = nib.load(f'{data_path}/dmri_data.nii.gz').affine
    bvals = np.loadtxt(f'{data_path}/bvals.bvals')
    bvecs = np.loadtxt(f'{data_path}/bvecs.bvecs')
    normalize(data_path,data,bvals,affine)
    normalize_dmri_data_file = nib.load(f'{data_path}/normalized_data.nii.gz')
    normalize_dmri_data = normalize_dmri_data_file.get_fdata()
    normalize_dmri_file_affine = normalize_dmri_data_file.affine
    shell_not0_index = np.where(bvals!=0)[0]
    bvecs = bvecs[:,shell_not0_index]
    bvals = bvals[shell_not0_index]
    normalize_dmri_data = normalize_dmri_data[:,:,:,shell_not0_index]
    shell = np.unique(bvals)
    shellnumber = np.unique(bvals).shape[0]

    dmri_wm_gm_csf_data_allshell = []
    dmri_wm_gm_csf_data_bool_allshell = []
    expanded_mask_wmgm_csf_allshell = []
    dmri_wm_data_allshell=[]
    dmri_wm_data_bool_allshell = []
    wm_expanded_mask_allshell = []
   
    for m in range(shellnumber):
        bval_shell = shell[m]
        shellindex=np.where(bvals==bval_shell)[0]
        n = np.arange(0, sh_degree + 1, 2) 
        dmri_wm_gm_csf_data_bool = []
        expanded_mask_wmgm_csf = []
        mask_bool_wmgmcsf = []
        mask_wmgm_csf = []
        if wm:
            wm_mask_file = nib.load(f'{data_path}/wm_mask.nii.gz')  
            wm_mask = wm_mask_file.get_fdata()
            mask_wmgm_csf.append(wm_mask)
            wm_mask_bool = np.bool_(wm_mask)
            mask_bool_wmgmcsf.append(wm_mask_bool)
            dmri_shelldata = normalize_dmri_data[:,:,:,shellindex]
           
            dmri_wm_data_bool_all = normalize_dmri_data[wm_mask_bool]
            wm_expanded_mask_all = np.expand_dims(wm_mask, axis=-1)
            wm_expanded_mask_all = np.repeat(wm_expanded_mask_all, normalize_dmri_data.shape[-1], axis=-1)
            dmri_wm_gm_csf_data_bool.append(dmri_wm_data_bool_all) 
            expanded_mask_wmgm_csf.append(wm_expanded_mask_all)
            
            dmri_wm_data_bool_shell = dmri_shelldata[wm_mask_bool]
            dmri_wm_data_bool_shell = np.nan_to_num(dmri_wm_data_bool_shell, nan=0)
            dmri_wm_data_bool_shell = np.where(dmri_wm_data_bool_shell == np.inf, 0, dmri_wm_data_bool_shell)
            dmri_wm_data_bool_shell = np.where(dmri_wm_data_bool_shell == -np.inf, 0, dmri_wm_data_bool_shell)  #把inf和-inf改为0

            dmri_wm_data_bool_allshell.append(dmri_wm_data_bool_shell) 
            wm_expanded_mask_shell = np.expand_dims(wm_mask, axis=-1)
            wm_expanded_mask_shell = np.repeat(wm_expanded_mask_shell, dmri_shelldata.shape[-1], axis=-1)
            wm_expanded_mask_allshell.append(wm_expanded_mask_shell) 
            
            dmri_wm_data_shell = np.multiply(dmri_shelldata, wm_expanded_mask_shell)
           
            dmri_wm_data_allshell.append(dmri_wm_data_shell) 
            dmri_wm_gm_csf_data_allshell.append(dmri_wm_data_allshell) 
            dmri_wm_gm_csf_data_bool_allshell.append(dmri_wm_data_bool_allshell)
            expanded_mask_wmgm_csf_allshell.append(wm_expanded_mask_allshell)
          
        if gm:
            gm_mask_file = nib.load(f'{data_path}/gm_mask.nii.gz')
            gm_mask = gm_mask_file.get_fdata()
            mask_wmgm_csf.append(gm_mask)
            gm_expanded_mask = np.expand_dims(gm_mask, axis=-1)
            gm_expanded_mask = np.repeat(gm_expanded_mask, normalize_dmri_data.shape[-1], axis=-1)
            expanded_mask_wmgm_csf.append(gm_expanded_mask)
            gm_mask_bool = np.bool_(gm_mask)
            mask_bool_wmgmcsf.append(gm_mask_bool)
            dmri_gm_data_bool = normalize_dmri_data[gm_mask_bool]
            dmri_wm_gm_csf_data_bool.append(dmri_gm_data_bool)
          
        if csf:
            csf_mask_file = nib.load(f'{data_path}/csf_mask.nii.gz')
            csf_mask = csf_mask_file.get_fdata()
            mask_wmgm_csf.append(csf_mask)
            csf_expanded_mask = np.expand_dims(csf_mask, axis=-1)
            csf_expanded_mask = np.repeat(csf_expanded_mask, normalize_dmri_data.shape[-1], axis=-1)
            expanded_mask_wmgm_csf.append(csf_expanded_mask)
            csf_mask_bool = np.bool_(csf_mask)
            mask_bool_wmgmcsf.append(csf_mask_bool)
            dmri_csf_data_bool = normalize_dmri_data[csf_mask_bool]
            dmri_wm_gm_csf_data_bool.append(dmri_csf_data_bool)
            
    return n,dmri_wm_gm_csf_data_bool_allshell,dmri_wm_gm_csf_data_bool,normalize_dmri_file_affine,mask_bool_wmgmcsf,shellnumber,bvals,bvecs,dmri_wm_gm_csf_data_allshell,expanded_mask_wmgm_csf_allshell,mask_wmgm_csf
