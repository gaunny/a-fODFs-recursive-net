import os
import time
import numpy as np
import nibabel as nib
import pickle

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
    # Training loop
    for epoch in range(n_epoch):
        # TRAIN
        model.train()
        # Initialize loss to save and plot.
        loss_intensity_ = 0
        loss_smoothness_ = 0
        loss_non_negativity_fodf_ = 0
        loss_oddsensitive_ = 0
        loss_pve_equi_ = 0
        loss_pve_inva_ = 0
        # Train on batch.
        for batch, data in enumerate(dataloader_train):
            start = time.time()
            # Delete all previous gradients
            optimizer.zero_grad()
            to_print = ''
            # Load the data in the DEVICE
            input = data['input'].to(DEVICE)  
            mask = data['mask'].to(DEVICE)
            output = data['out'].to(DEVICE)
            x_reconstructed,x_convolved_equi_shc, x_deconvolved_equi_shc, x_deconvolved_inva_shc = model(input)
            if middle_voxel:
                x_reconstructed = x_reconstructed[:, :, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1]
                x_deconvolved_equi_shc = x_deconvolved_equi_shc[:, :, :, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1]
                x_deconvolved_inva_shc = x_deconvolved_inva_shc[:, :, :, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1]
                output = output[:, :, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1]
                mask = mask[:, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1]
                x_convolved_equi_shc = x_convolved_equi_shc[:, :, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1]
            # Loss
            # Intensity loss
            
            loss_intensity = intensity_criterion(x_reconstructed, output, mask[:, None].expand(-1, output.shape[1], -1, -1, -1))
            loss_intensity_ += loss_intensity.item()
            loss = intensity_weight * loss_intensity
            to_print += ', Intensity: {0:.10f}'.format(loss_intensity.item())
            index=0
            if not x_deconvolved_equi_shc  is None:
                x_deconvolved_equi = denseGrid_interpolate(x_deconvolved_equi_shc)
                # Non negativity loss
                fodf_neg = torch.min(x_deconvolved_equi, torch.zeros_like(x_deconvolved_equi))
                fodf_neg_zeros = torch.zeros(fodf_neg.shape).cuda()
                loss_non_negativity_fodf = non_negativity_criterion(fodf_neg, fodf_neg_zeros, mask[:, None, None].expand(-1, fodf_neg_zeros.shape[1], fodf_neg_zeros.shape[2], -1, -1, -1))
                loss_non_negativity_fodf_ += loss_non_negativity_fodf.item()
                loss += nn_fodf_weight * loss_non_negativity_fodf
                to_print += ', Equi NN: {0:.10f}'.format(loss_non_negativity_fodf.item())
                #oddsensitive loss
                loss_oddsensitive = oddsensitive_criterion(x_deconvolved_equi_shc, sh_degree)
                loss_oddsensitive_ += loss_oddsensitive.item()
                loss += oddsensitive_weight * loss_oddsensitive
                to_print += ', Equi oddsensitive: {0:.10f}'.format(loss_oddsensitive.item())
                # Smoothness loss
                loss_smoothness = smoothness_criterion(x_deconvolved_equi_shc, sh_degree)
                loss_smoothness_ += loss_smoothness.item()
                loss += smoothness_weight * loss_smoothness
                to_print += ', Equi Smoothness: {0:.10f}'.format(loss_smoothness.item())
                # Partial volume regularizer
                new_tensor = torch.unsqueeze(x_deconvolved_equi_shc[:, index,0], 1)  
                loss_pve_equi = 1/(torch.mean(new_tensor[mask[:, None]==1])*np.sqrt(4*np.pi) + 1e-16)
                loss_pve_equi_ += loss_pve_equi.item()
                loss += pve_weight * loss_pve_equi
                to_print += ', Equi regularizer: {0:.10f}'.format(loss_pve_equi.item())
                index+=1
            if gm:
                new_tensor = torch.unsqueeze(x_deconvolved_equi_shc[:, index,0], 1)
                loss_pve_equi = 1/(torch.mean(new_tensor[mask[:, None]==1])*np.sqrt(4*np.pi) + 1e-16)
                loss_pve_equi_ += loss_pve_equi.item()
                loss += pve_weight * loss_pve_equi
                to_print += ', Equi regularizer GM: {0:.10f}'.format(loss_pve_equi.item())
            if not x_deconvolved_inva_shc is None:
               # Partial volume regularizer
               loss_pve_inva = 1/torch.mean(x_deconvolved_inva_shc[:, :, 0][mask[:, None].expand(-1, x_deconvolved_inva_shc.shape[1], -1, -1, -1)==1])*np.sqrt(4*np.pi)
               loss_pve_inva_ += loss_pve_inva.item()
               loss += pve_weight * loss_pve_inva
               to_print += ', Inva regularizer: {0:.10f}'.format(loss_pve_inva.item())
          
            if csf:
                loss_pve_inva = 1/(torch.mean(x_deconvolved_inva_shc[:, index, 0][mask==1])*np.sqrt(4*np.pi) + 1e-16)
                loss_pve_inva_ += loss_pve_inva.item()
                loss += pve_weight * loss_pve_inva
                to_print += ', Inva regularizer CSF: {0:.10f}'.format(loss_pve_inva.item())
            
            # Loss backward
            loss = loss
            loss.backward()
            optimizer.step()
            # To print loss
            end = time.time()
            to_print += ', Elapsed time: {0} s'.format(end - start)
            to_print = 'Epoch [{0}/{1}], Iter [{2}/{3}]: Loss: {4:.10f}'.format(epoch + 1, n_epoch, batch + 1, n_batch, loss.item()) + to_print
            print(to_print, end="\r")
            if (batch + 1) % 500 == 0:
                torch.save(model.state_dict(), os.path.join(save_path, 'history', 'epoch_{0}.pth'.format(epoch + 1)))
        # Save and print mean loss for the epoch
        print("")
        to_print = ''
        loss_ = 0
        # Mean results of the last epoch
        save_loss['train'][epoch] = {}
        save_loss['train'][epoch]['loss_intensity'] = loss_intensity_ / n_batch
        save_loss['train'][epoch]['weight_loss_intensity'] = intensity_weight
        loss_ += intensity_weight * loss_intensity_
        to_print += ', Intensity: {0:.10f}'.format(loss_intensity_ / n_batch)
       
        save_loss['train'][epoch]['loss_smoothness'] = loss_smoothness_ / n_batch
        save_loss['train'][epoch]['weight_loss_smoothness'] = smoothness_weight
        loss_ += smoothness_weight * loss_smoothness_
        to_print += ', Smoothness: {0:.10f}'.format(loss_smoothness_ / n_batch)

        save_loss['train'][epoch]['loss_non_negativity_fodf'] = loss_non_negativity_fodf_ / n_batch
        save_loss['train'][epoch]['weight_loss_non_negativity_fodf'] = nn_fodf_weight
        loss_ += nn_fodf_weight * loss_non_negativity_fodf_
        to_print += ', WM fODF NN: {0:.10f}'.format(loss_non_negativity_fodf_ / n_batch)

        save_loss['train'][epoch]['loss_oddsensitive'] = loss_oddsensitive_ / n_batch
        save_loss['train'][epoch]['weight_loss_oddsensitive'] = oddsensitive_weight
        loss_ += oddsensitive_weight * loss_oddsensitive_
        to_print += ', Oddsensitive: {0:.10f}'.format(loss_oddsensitive_/ n_batch)
      
        save_loss['train'][epoch]['loss_pve_equi'] = loss_pve_equi_ / n_batch
        save_loss['train'][epoch]['weight_loss_pve_equi'] = pve_weight
        loss_ += pve_weight * loss_pve_equi_
        to_print += ', Equi regularizer: {0:.10f}'.format(loss_pve_equi_ / n_batch)
        save_loss['train'][epoch]['loss_pve_inva'] = loss_pve_inva_ / n_batch
        save_loss['train'][epoch]['weight_loss_pve_inva'] = pve_weight
        loss_ += pve_weight * loss_pve_inva_
        to_print += ', Inva regularizer: {0:.10f}'.format(loss_pve_inva_ / n_batch)
        save_loss['train'][epoch]['loss'] = loss_ / n_batch
        to_print = 'Epoch [{0}/{1}], Train Loss: {2:.10f}'.format(epoch + 1, n_epoch, loss_ / n_batch) + to_print
        print(to_print)
       
        ###############################################################################################
        # VALIDATION  loss_ < min_loss * 0.9999: 
        scheduler.step(loss_ / n_batch)
        if epoch == 0:
            min_loss = loss_
            epochs_no_improve = 0
            n_epochs_stop = 1
            early_stop = False
        elif loss_ < min_loss * 0.9999:   
            epochs_no_improve = 0
            min_loss = loss_
        else:
            epochs_no_improve += 1
        if epoch > 1 and epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            early_stop = True
        ###############################################################################################
        # Save the loss and model
        with open(os.path.join(save_path, 'history', 'loss.pkl'), 'wb') as f:
            pickle.dump(save_loss, f)
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'history', 'epoch_{0}.pth'.format(epoch + 1)))
        if early_stop:
            print("Stopped")
            break
    #test
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
    test_path = f"{save_path}/test{'_middle'*middle_voxel}/epoch_{epoch}"
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
    nib.save(img, f"{save_path}/test{'_middle'*middle_voxel}/epoch_{epoch}/count.nii.gz")
    if bs>0:
        reconstruction_list = reconstruction_list[bs:-bs,bs:-bs,bs:-bs]
    #else:
    #    reconstruction_list = reconstruction_list[:dataset.X,:dataset.Y,:dataset.Z]
    reconstruction_list = np.array(reconstruction_list).astype(np.float32)
    img = nib.Nifti1Image(reconstruction_list, dataset.affine)
    nib.save(img, f"{save_path}/test{'_middle'*middle_voxel}/epoch_{epoch}/reconstruction.nii.gz")
    fodf_shc_wmgmcsf = []
    if wm:
        if bs>0:
            fodf_shc_wm_list = fodf_shc_wm_list[bs:-bs,bs:-bs,bs:-bs]
        #else:
        #    fodf_shc_wm_list = fodf_shc_wm_list[:dataset.X,:dataset.Y,:dataset.Z]
        fodf_shc_wm_list = np.array(fodf_shc_wm_list).astype(np.float32)
        
        # Retain data only if mask equals 1
        mask_img = nib.load(f'{data_path}/wm_mask.nii.gz')
        mask = mask_img.get_fdata()
        expanded_mask = np.expand_dims(mask, axis=-1)
        expanded_mask = np.repeat(expanded_mask, fodf_shc_wm_list.shape[-1], axis=-1)
        data_new = np.where(expanded_mask == 1, fodf_shc_wm_list, 0)
        wm_fodf = data_new
        fodf_shc_wmgmcsf.append(wm_fodf)
        img = nib.Nifti1Image(wm_fodf, dataset.affine)
        nib.save(img, f"{save_path}/test{'_middle'*middle_voxel}/epoch_{epoch}/a-fODFs_wm.nii.gz")
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