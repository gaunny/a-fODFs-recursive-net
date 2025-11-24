import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.dataset import DMRIDataset
from utils.sampling import HealpixSampling, ShellSampling, BvecSampling
from utils.response import load_response_function
from model.model import Model
from utils.loss import Loss,Loss_smoothness,Loss_oddsensitive
from model.shutils import ComputeSignal
from utils.tools import get_latest_pth_file

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def spherical_model(data_path, batch_size, lr, n_epoch, kernel_sizeSph, kernel_sizeSpa, 
             filter_start, sh_degree, depth, n_side,
             response_p_wmgmcsf, wm, gm, csf,
             loss_fn_intensity, loss_fn_non_negativity,loss_fn_smoothness,loss_fn_oddsensitive,
             intensity_weight, nn_fodf_weight, pve_weight,smoothness_weight,smoothness_sigma,oddsensitive_weight,
             save_path, save_every, normalize, load_state, patch_size, graph_sampling, conv_name, isoSpa, concatenate, middle_voxel,full_basis) :
    
    dataset = DMRIDataset(data_path, f'{data_path}/normalized_data.nii.gz', f'{data_path}/wm_mask.nii.gz', f'{data_path}/bvecs.bvecs', f'{data_path}/bvals.bvals', patch_size, concatenate=concatenate)
    feature_in = 1
    if concatenate:
        feature_in = patch_size*patch_size*patch_size
        patch_size = 1
    bvec = dataset.vec
    
    if graph_sampling=='healpix':
        graphSampling = HealpixSampling(n_side, depth, patch_size,full_basis=full_basis, sh_degree=sh_degree, pooling_name=conv_name)
    elif graph_sampling=='bvec':
        graphSampling = BvecSampling(bvec, depth, image_size=patch_size, full_basis=full_basis,sh_degree=sh_degree, pooling_mode='average')
    else:
        raise NotImplementedError
    shellSampling = ShellSampling(full_basis,f'{data_path}/bvecs.bvecs', f'{data_path}/bvals.bvals', sh_degree=sh_degree, max_sh_degree=10)

    # Load the image and the mask
    print("Dataset size:", len(dataset))
    dataloader_train = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    n_batch = len(dataloader_train)
    n_shell=len(shellSampling.shell_values)

    # Load the Polar filter used for the deconvolution
    polar_filter_equi, polar_filter_inva = load_response_function(response_p_wmgmcsf, wm=wm, gm=gm, csf=csf, max_degree=sh_degree, n_shell=n_shell, norm=dataset.normalization_value)
    print('polar_filter_equi',polar_filter_equi)
    print('polar_filter_inva',polar_filter_inva)
    model = Model(polar_filter_equi, polar_filter_inva, shellSampling,graphSampling, filter_start, kernel_sizeSph, kernel_sizeSpa, normalize, conv_name, isoSpa, feature_in,full_basis,sh_degree)
    if load_state:
        print(load_state)
        model.load_state_dict(torch.load(load_state), strict=False)
    
    model = model.to(DEVICE)
   
    torch.save(model.state_dict(), os.path.join(save_path, 'history', 'epoch_0.pth'))

    # Loss
    intensity_criterion = Loss(loss_fn_intensity)
    non_negativity_criterion = Loss(loss_fn_non_negativity)
    oddsensitive_criterion = Loss_oddsensitive(loss_fn_oddsensitive)
    smoothness_criterion = Loss_smoothness(loss_fn_smoothness,smoothness_sigma)
    denseGrid_interpolate = ComputeSignal(torch.Tensor(graphSampling.sampling.SH2S))
    denseGrid_interpolate = denseGrid_interpolate.cuda()
    # Optimizer/Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, threshold=0.01, factor=0.1, patience=3, verbose=True)
    save_loss = {}
    save_loss['train'] = {}
    folder_path = f'{save_path}/history/'
    latest_pth_file = get_latest_pth_file(folder_path)
    if latest_pth_file:
        print("the last updated '.pth' file:", latest_pth_file)
        epoch = latest_pth_file.split('/')[-1].split('.')[0].split('_')[-1]
        print('epoch',epoch)
    else:
        print("There is no '.pth' file in the folder。")
    test_model = Model(polar_filter_equi, polar_filter_inva, shellSampling,graphSampling, filter_start, kernel_sizeSph, kernel_sizeSpa, normalize, conv_name, isoSpa, feature_in,full_basis,sh_degree)
    test_model.load_state_dict(torch.load(latest_pth_file),strict=False)
    # Load model in GPU
    test_model = test_model.to(DEVICE)
    
    test_model.eval()
    # Output initialization
    if middle_voxel:
        b_selected = 1
        b_start = patch_size//2
        b_end = b_start + 1
    else:
        b_selected = patch_size
        b_start = 0
        b_end = b_selected
    if full_basis == False:
        nb_coef = int((sh_degree + 1) * (sh_degree / 2 + 1))
    else:
        nb_coef = int((sh_degree + 1) * (sh_degree + 1))

    count = np.zeros((dataset.data.shape[1],
                    dataset.data.shape[2],
                    dataset.data.shape[3]))
    reconstruction_list = np.zeros((dataset.data.shape[1],
                                    dataset.data.shape[2],
                                    dataset.data.shape[3], len(shellSampling.vectors)))
    if wm:
        fodf_shc_wm_list = np.zeros((dataset.data.shape[1],
                                    dataset.data.shape[2],
                                    dataset.data.shape[3], nb_coef))
       
    if gm:
        fodf_shc_gm_list = np.zeros((dataset.data.shape[1],
                                    dataset.data.shape[2],
                                    dataset.data.shape[3], nb_coef)) 
    if csf:
        fodf_shc_csf_list = np.zeros((dataset.data.shape[1],
                                    dataset.data.shape[2],
                                    dataset.data.shape[3], 1))
    # Test on batch.
    for i, data in enumerate(dataloader_train):
        print(str(i * 100 / n_batch) + " %", end='\r', flush=True)
        # Load the data in the DEVICE
        input = data['input'].to(DEVICE)
        sample_id = data['sample_id']
        x_reconstructed,x_convolved_equi_shc, x_deconvolved_equi_shc, x_deconvolved_inva_shc = test_model(input)
      
        for j in range(len(input)):
            sample_id_j = sample_id[j]
            reconstruction_list[dataset.x[sample_id_j] - (b_selected // 2):dataset.x[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.y[sample_id_j] - (b_selected // 2):dataset.y[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.z[sample_id_j] - (b_selected // 2):dataset.z[sample_id_j] + (b_selected // 2) + (b_selected%2)] += x_reconstructed[j, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
            index = 0
            if wm:
                fodf_shc_wm_list[dataset.x[sample_id_j] - (b_selected // 2):dataset.x[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.y[sample_id_j] - (b_selected // 2):dataset.y[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.z[sample_id_j] - (b_selected // 2):dataset.z[sample_id_j] + (b_selected // 2) + (b_selected%2)] += x_deconvolved_equi_shc[j, index, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()  #[j, 0, :, b_start:b_end, b_start:b_end, b_start:b_end]
                index += 1
            if gm:
                fodf_shc_gm_list[dataset.x[sample_id_j] - (b_selected // 2):dataset.x[sample_id_j] + (b_selected // 2) + (b_selected%2),
                    dataset.y[sample_id_j] - (b_selected // 2):dataset.y[sample_id_j] + (b_selected // 2) + (b_selected%2),
                    dataset.z[sample_id_j] - (b_selected // 2):dataset.z[sample_id_j] + (b_selected // 2) + (b_selected%2)] += x_deconvolved_equi_shc[j, index, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
         
            if csf:
                fodf_shc_csf_list[dataset.x[sample_id_j] - (b_selected // 2):dataset.x[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.y[sample_id_j] - (b_selected // 2):dataset.y[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.z[sample_id_j] - (b_selected // 2):dataset.z[sample_id_j] + (b_selected // 2) + (b_selected%2)] += x_deconvolved_inva_shc[j, index, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
            count[dataset.x[sample_id_j] - (b_selected // 2):dataset.x[sample_id_j] + (b_selected // 2) + (b_selected%2),
                dataset.y[sample_id_j] - (b_selected // 2):dataset.y[sample_id_j] + (b_selected // 2) + (b_selected%2),
                dataset.z[sample_id_j] - (b_selected // 2):dataset.z[sample_id_j] + (b_selected // 2) + (b_selected%2)] += 1
    # Average patch
    try:
        reconstruction_list[count!=0] = reconstruction_list[count!=0] / count[count!=0, None]
        if wm:
            fodf_shc_wm_list[count!=0] = fodf_shc_wm_list[count!=0] / count[count!=0, None]
        if gm:
            fodf_shc_gm_list[count!=0] = fodf_shc_gm_list[count!=0] / count[count!=0, None]
        if csf:
            fodf_shc_csf_list[count!=0] = fodf_shc_csf_list[count!=0] / count[count!=0, None]
    except:
        print('Count failed')
    # Test directory
    
    test_path = f"{data_path}/test{'_middle'*middle_voxel}/epoch_{epoch}"
    if not os.path.exists(test_path):
        print('Create new directory: {0}'.format(test_path))
        os.makedirs(test_path)
    # Save the results
    #bs = 0
    bs = patch_size//2
    if bs>0:
        count = count[bs:-bs,bs:-bs,bs:-bs]
    #else:
    #    count = count[:dataset.X,:dataset.Y,:dataset.Z]
    count = np.array(count).astype(np.float32)
    img = nib.Nifti1Image(count, dataset.affine)
    nib.save(img, f"{data_path}/test{'_middle'*middle_voxel}/epoch_{epoch}/count.nii.gz")
    if bs>0:
        reconstruction_list = reconstruction_list[bs:-bs,bs:-bs,bs:-bs]
    #else:
    #    reconstruction_list = reconstruction_list[:dataset.X,:dataset.Y,:dataset.Z]
    reconstruction_list = np.array(reconstruction_list).astype(np.float32)
    img = nib.Nifti1Image(reconstruction_list, dataset.affine)
    nib.save(img, f"{data_path}/test{'_middle'*middle_voxel}/epoch_{epoch}/reconstruction.nii.gz")
    fodf_shc_wmgmcsf = []
    if wm:
        if bs>0:
            fodf_shc_wm_list = fodf_shc_wm_list[bs:-bs,bs:-bs,bs:-bs]
        #else:
        #    fodf_shc_wm_list = fodf_shc_wm_list[:dataset.X,:dataset.Y,:dataset.Z]
        fodf_shc_wm_list = np.array(fodf_shc_wm_list).astype(np.float32)
        
        # Retain data only if mask equals 1
        mask_img = nib.load(f'{data_path}/wmgm_mask.nii.gz')
        mask = mask_img.get_fdata()
        expanded_mask = np.expand_dims(mask, axis=-1)
        expanded_mask = np.repeat(expanded_mask, fodf_shc_wm_list.shape[-1], axis=-1)
        data_new = np.where(expanded_mask == 1, fodf_shc_wm_list, 0)
        wm_fodf = data_new
        fodf_shc_wmgmcsf.append(wm_fodf)
        img = nib.Nifti1Image(wm_fodf, dataset.affine)
        nib.save(img, f"{data_path}/test{'_middle'*middle_voxel}/epoch_{epoch}/a-fODFs_wm.nii.gz")
    if gm:
        if bs>0:
            fodf_shc_gm_list = fodf_shc_gm_list[bs:-bs,bs:-bs,bs:-bs]
        #else:
        #    fodf_shc_gm_list = fodf_shc_gm_list[:dataset.X,:dataset.Y,:dataset.Z]
        fodf_shc_gm_list = np.array(fodf_shc_gm_list).astype(np.float32)
        mask_img = nib.load(f'{data_path}/gm_mask.nii.gz')
        mask = mask_img.get_fdata()
        expanded_mask = np.expand_dims(mask, axis=-1)
        expanded_mask = np.repeat(expanded_mask, fodf_shc_gm_list.shape[-1], axis=-1)
        data_new = np.where(expanded_mask == 1, fodf_shc_gm_list, 0)
        gm_fodf = data_new
        fodf_shc_wmgmcsf.append(gm_fodf)
       
        img = nib.Nifti1Image(gm_fodf, dataset.affine)
        nib.save(img, f"{save_path}/test{'_middle'*middle_voxel}/epoch_{epoch}/a-fODFs_gm.nii.gz")
    if csf:
        if bs>0:
            fodf_shc_csf_list = fodf_shc_csf_list[bs:-bs,bs:-bs,bs:-bs]
        #else:
        #    fodf_shc_csf_list = fodf_shc_csf_list[:dataset.X,:dataset.Y,:dataset.Z]
        fodf_shc_csf_list = np.array(fodf_shc_csf_list).astype(np.float32)
        mask_img = nib.load(f'{data_path}/csf_mask.nii.gz')
        mask = mask_img.get_fdata()
        expanded_mask = np.expand_dims(mask, axis=-1)
        expanded_mask = np.repeat(expanded_mask, fodf_shc_gm_list.shape[-1], axis=-1)
        data_new = np.where(expanded_mask == 1, fodf_shc_gm_list, 0)
        csf_fodf = data_new
        fodf_shc_wmgmcsf.append(csf_fodf)
        img = nib.Nifti1Image(fodf_shc_csf_list, dataset.affine)
        nib.save(img, f"{save_path}/test{'_middle'*middle_voxel}/epoch_{epoch}/a-fODFs_csf.nii.gz")

    return fodf_shc_wmgmcsf
