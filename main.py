import argparse
import os
import time
import json
import numpy as np
import nibabel as nib

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.reconst.shm import lazy_index
from dipy.core.ndindex import ndindex
from dipy.direction.peaks import peak_directions
from dipy.reconst.shm import sf_to_sh
from dipy.core.geometry import vec2vec_rotmat
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sh_descoteaux_from_index
from dipy.reconst.csdeconv import AxSymShResponse
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.direction.peaks import PeaksAndMetrics
from dipy.direction.peaks import reshape_peaks_for_visualization

from utils.dataset import load_data
from spherical_model import spherical_model
from utils.tools import create_dir,odfsh_sf,pam_from_attrs,generate_fatRF


def main(data_path,full_basis,sh_degree,n_peaks,iteration,sphere, global_max, relative_peak_threshold,min_separation_angle,peak_thr,convergence,
         batch_size, lr, n_epoch, kernel_sizeSph, kernel_sizeSpa, 
         filter_start, depth, n_side, wm, gm, csf,
         loss_fn_intensity, loss_fn_non_negativity,loss_fn_smoothness,loss_fn_oddsensitive,
         intensity_weight, nn_fodf_weight, pve_weight,smoothness_weight,smoothness_sigma,oddsensitive_weight,
         save_every, normalize, load_state, patch_size, graph_sampling, conv_name, isoSpa, concatenate, middle_voxel):
    save_path = create_dir(os.path.join(data_path, 'results'))
    save_path = os.path.join(save_path, time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime()))
    print('Save path: {0}'.format(save_path))
    save_path += f'_{conv_name}_{patch_size}_{filter_start}_{depth}_{sh_degree}_{wm}_{gm}_{csf}'
    n,dmri_wm_gm_csf_data_bool_allshell,dmri_wm_gm_csf_data_bool,dmri_file_affine,mask_bool_wmgmcsf,shellnumber,bvals,bvecs,dmri_wm_gm_csf_data_allshell,expanded_mask_wmgm_csf_allshell,mask_wmgm_csf = load_data(data_path,sh_degree,wm,gm,csf)
    
    # Generate the fat RF
    response_p_wmgmcsf = generate_fatRF(data_path)
    print('response_p_wmgmcsf',response_p_wmgmcsf)
    print('response_p_wmgmcsf',response_p_wmgmcsf.shape)
    for iter in range(iteration):
        print('***********************iterations:',iter,'**********************')
        save_path += f'_{iter}'
        history_path = create_dir(os.path.join(save_path, 'history'))
        response_path = create_dir(os.path.join(save_path, 'response'))
        with open(os.path.join(save_path, 'args.txt'), 'w') as file:
            json.dump(args.__dict__, file, indent=2)
        r_sh_all_allshell = None
        for m_shell in range(shellnumber):
            r_sh_all = np.zeros(len(n))
            wmgmcsf_shape = []
            qa_array_wmgmcsf=[]
            peak_dirs_wmgmcsf = []
            peak_values_wmgmcsf = []
            peak_indices_wmgmcsf = []
            if wm:
                wm_shape = dmri_wm_gm_csf_data_bool[0].shape[:-1]
                wmgmcsf_shape.append(wm_shape)
                wm_mask = np.ones(wm_shape,dtype='bool')
                qa_array_wm = np.zeros((wm_shape + (n_peaks,)))
                qa_array_wmgmcsf.append(qa_array_wm)
                peak_dirs_wm = np.zeros((wm_shape + (n_peaks, 3)))
                peak_dirs_wmgmcsf.append(peak_dirs_wm)
                peak_values_wm = np.zeros((wm_shape + (n_peaks,)))
                peak_values_wmgmcsf.append(peak_values_wm)
                peak_indices_wm = np.zeros((wm_shape + (n_peaks,)), dtype=np.int32)
                peak_indices_wm.fill(-1)
                peak_indices_wmgmcsf.append(peak_indices_wm)
            if gm:
                gm_shape = dmri_wm_gm_csf_data_bool_allshell[1][m_shell].shape[:-1]
                wmgmcsf_shape.append(gm_shape)
                qa_array_gm = np.zeros((gm_shape + (n_peaks,)))
                qa_array_wmgmcsf.append(qa_array_gm)
                peak_dirs_gm = np.zeros((gm_shape + (n_peaks, 3)))
                peak_dirs_wmgmcsf.append(peak_dirs_gm)
                peak_values_gm = np.zeros((gm_shape + (n_peaks,)))
                peak_values_wmgmcsf.append(peak_values_gm)
                peak_indices_gm = np.zeros((gm_shape + (n_peaks,)), dtype=np.int32)
                peak_indices_gm.fill(-1)
                peak_indices_wmgmcsf.append(peak_indices_gm)
            if csf:
                csf_shape = dmri_wm_gm_csf_data_bool_allshell[2][m_shell].shape[:-1]
                wmgmcsf_shape.append(csf_shape)
                qa_array_csf = np.zeros((csf_shape + (n_peaks,)))
                qa_array_wmgmcsf.append(qa_array_csf)
                peak_dirs_csf = np.zeros((csf_shape + (n_peaks, 3)))
                peak_dirs_wmgmcsf.append(peak_dirs_csf)
                peak_values_csf = np.zeros((csf_shape + (n_peaks,)))
                peak_values_wmgmcsf.append(peak_values_csf)
                peak_indices_csf = np.zeros((csf_shape + (n_peaks,)), dtype=np.int32)
                peak_indices_csf.fill(-1)
                peak_indices_wmgmcsf.append(peak_indices_csf)

        fodf_shc_wmgmcsf = spherical_model(data_path, batch_size, lr, n_epoch, kernel_sizeSph, kernel_sizeSpa, 
             filter_start, sh_degree, depth, n_side,
             response_p_wmgmcsf, wm, gm, csf,
             loss_fn_intensity, loss_fn_non_negativity,loss_fn_smoothness,loss_fn_oddsensitive,
             intensity_weight, nn_fodf_weight, pve_weight,smoothness_weight,smoothness_sigma,oddsensitive_weight,
             save_path, save_every, normalize, load_state, patch_size, graph_sampling, conv_name, isoSpa, concatenate, middle_voxel,full_basis) 
        
        if full_basis==True:
            n_shm_coeff = int((sh_degree + 1) * (sh_degree + 1) )
        else: 
            n_shm_coeff = int((sh_degree + 2) * (sh_degree + 1) // 2)
        if wm:
            response_p_wm = response_p_wmgmcsf[0]
            fodf_shc_wm = fodf_shc_wmgmcsf[0]
            odf_sphere = odfsh_sf(odf_sh=fodf_shc_wm,mask_bool=mask_bool_wmgmcsf[0],sh_degree=sh_degree,full_basis=full_basis)
            shm_coeff = np.zeros((wm_shape + (n_shm_coeff,)))
            B, invB = sh_to_sf_matrix(sphere, sh_degree, None, full_basis=full_basis,return_inv=True)
            shape = dmri_wm_gm_csf_data_bool[0].shape[:-1]
            for idx in ndindex(shape):  #wm_shape
                odf_sphere_ = odf_sphere[idx]
                shm_coeff[idx] = np.dot(odf_sphere_, invB)
                direction, pk, ind = peak_directions(odf_sphere_, sphere,relative_peak_threshold,min_separation_angle,is_symmetric=True)
                # Calculate peak metrics
                if pk.shape[0] != 0:
                    global_max = max(global_max, pk[0])
                    m = min(n_peaks, pk.shape[0])
                    qa_array_wm[idx][:m] = pk[:m] - odf_sphere_.min()
                    peak_dirs_wm[idx][:m] = direction[:m]
                    peak_indices_wm[idx][:m] = ind[:m]
                    peak_values_wm[idx][:m] = pk[:m]
            gfa_array = None
            odf_array = None
            peaks = pam_from_attrs(PeaksAndMetrics,sphere,peak_indices_wm,peak_values_wm,peak_dirs_wm,gfa_array,qa_array_wm,shm_coeff,B,odf_array)
            reshape_peak = reshape_peaks_for_visualization(peaks)
            
            indices = np.where(mask_bool_wmgmcsf[0])
            print('----indices',np.count_nonzero(mask_bool_wmgmcsf[0]))
            wm_mask_file = nib.load(f'{data_path}/wm_mask.nii.gz') 
            wm_mask = wm_mask_file.get_fdata()
            data_orgin = np.zeros((wm_mask.shape[0],wm_mask.shape[1],wm_mask.shape[2],n_peaks*3))
            
            data_orgin[indices] = reshape_peak
           
            peak_img = nib.Nifti1Image(data_orgin,affine = dmri_file_affine)
            nib.save(peak_img,os.path.join(response_path, 'peaks_wm.nii.gz'))
            single_peak_mask = (peak_values_wm[:, 1] / peak_values_wm[:, 0]) < peak_thr
            
            data_wm_new = dmri_wm_gm_csf_data_bool[0][single_peak_mask]

            response_allshell = None
            for n_shell in range(shellnumber):
                bval_shell=np.unique(bvals)[n_shell]
                shellindex=np.where(bvals==bval_shell)[0]
                gtab = gradient_table(bvals[shellindex], bvecs[:,shellindex])
                where_dwi = lazy_index(~gtab.b0s_mask)
                data_wm_new_shell = dmri_wm_gm_csf_data_bool_allshell[0][n_shell][single_peak_mask]
                dirs = peak_dirs_wm[single_peak_mask]
                r_sh_all = np.zeros(len(n))
                for num_vox in range(data_wm_new_shell.shape[0]): 
                    rotmat = vec2vec_rotmat(dirs[num_vox, 0], np.array([0, 0, 1]))
                    rot_gradients = np.dot(rotmat, gtab.gradients.T).T
                    
                    x, y, z = rot_gradients[where_dwi].T
                    r, theta, phi = cart2sphere(x, y, z)
                    # for the gradient sphere
                    B_dwi = real_sh_descoteaux_from_index(
                        0, n, theta[:, None], phi[:, None])
                    r_sh_all += np.linalg.lstsq(B_dwi, data_wm_new_shell[num_vox, where_dwi].astype('float32'),rcond=-1)[0]
                response = r_sh_all / data_wm_new_shell.shape[0]
                if n_shell == 0:
                    response_allshell = response
                else:
                    response_allshell=np.vstack((response_allshell,response))
            change = abs((response_p_wm - response_allshell) / response_p_wm)
            all_less_than_convergence = (change < convergence).all().item()
            if all_less_than_convergence:
                break
            response_p_wm = response_allshell
            if np.isnan(response_p_wm).any()==True:
                break
            else:
                np.savetxt(os.path.join(response_path,'recursive_RF_wm_{}.txt'.format(iter)),response_p_wm)
            gtab_all = gradient_table(bvals, bvecs)
            res_obj = AxSymShResponse(data_wm_new[:, gtab_all.b0s_mask].mean(), response)
            rec_response_signal = res_obj.on_sphere(sphere)
            rec_response_signal_sh = sf_to_sh(rec_response_signal,sphere,sh_order=sh_degree,basis_type=None,full_basis=full_basis)
            # transform our data from 1D to 4D
            rec_response_signal_sh = rec_response_signal_sh[None, None, None, :]
            data_new_img = nib.Nifti1Image(rec_response_signal_sh, affine=dmri_file_affine)
            nib.save(data_new_img, os.path.join(response_path,'rec_response_signal_sh_wm_{}.nii.gz'.format(iter)))
        
        if gm:
            response_p_gm = response_p_wmgmcsf[1]
            fodf_shc_gm = fodf_shc_wmgmcsf[1]
            odf_sphere = odfsh_sf(odf_sh=fodf_shc_gm,mask_bool=mask_bool_wmgmcsf[1],sh_degree=1,full_basis=full_basis)
            shm_coeff = np.zeros((gm_shape + (n_shm_coeff,)))
            B, invB = sh_to_sf_matrix(sphere, sh_degree, None,full_basis=full_basis, return_inv=True)
            for idx in ndindex(gm_shape):
                odf_sphere_ = odf_sphere[idx]
                shm_coeff[idx] = np.dot(odf_sphere_, invB)
                direction, pk, ind = peak_directions(odf_sphere_, sphere,relative_peak_threshold,min_separation_angle,is_symmetric=True)
                # Calculate peak metrics
                if pk.shape[0] != 0:
                    global_max = max(global_max, pk[0])
                    m = min(n_peaks, pk.shape[0])
                    qa_array_gm[idx][:m] = pk[:m] - odf_sphere_.min()
                    peak_dirs_gm[idx][:m] = direction[:m]
                    peak_indices_gm[idx][:m] = ind[:m]
                    peak_values_gm[idx][:m] = pk[:m]
            gfa_array = None
            odf_array = None
            peaks = pam_from_attrs(PeaksAndMetrics,sphere,peak_indices_gm,peak_values_gm,peak_dirs_gm,gfa_array,qa_array_gm,shm_coeff,B,odf_array)
            reshape_peak = reshape_peaks_for_visualization(peaks)
            # 获取 mask 中 True 的索引
            indices = np.where(mask_bool_wmgmcsf[1])
            gm_mask_file = nib.load(f'{data_path}/gm_mask.nii.gz')  
            gm_mask = gm_mask_file.get_fdata()
            data_orgin = np.zeros((gm_mask.shape[0],gm_mask.shape[1],gm_mask.shape[2],n_peaks*3))
            # 将 data 中的元素放回到 data_orgin 数组的相应位置
            data_orgin[indices] = reshape_peak
            peak_img = nib.Nifti1Image(data_orgin,affine = dmri_file_affine)
            nib.save(peak_img,os.path.join(response_path, 'peaks_gm.nii.gz'))
            single_peak_mask = (peak_values_gm[:, 1] / peak_values_gm[:, 0]) < peak_thr
            data_gm_new = dmri_wm_gm_csf_data_bool[1][single_peak_mask]
            dirs = peak_dirs_gm[single_peak_mask]
            for num_vox in range(data_gm_new.shape[0]):
                rotmat = vec2vec_rotmat(dirs[num_vox, 0], np.array([0, 0, 1]))
                rot_gradients = np.dot(rotmat, gtab.gradients.T).T
                x, y, z = rot_gradients[where_dwi].T
                r, theta, phi = cart2sphere(x, y, z)
                # for the gradient sphere
                B_dwi = real_sh_descoteaux_from_index(
                    0, n, theta[:, None], phi[:, None])
                r_sh_all += np.linalg.lstsq(B_dwi, data_gm_new[num_vox, where_dwi].astype('float32'),rcond=-1)[0]
            response = r_sh_all / data_gm_new.shape[0]
            change = abs((response_p_gm - response) / response_p_gm)
            if all(change < convergence):
                break
            response_p_gm = response
            res_obj = AxSymShResponse(data_gm_new[:, gtab.b0s_mask].mean(), response)
            # print('*****response_p_gm*****',response_p_gm)
            if np.isnan(response_p_gm).any()==True:
                break
            else:
                np.savetxt(os.path.join(response_path,'recursive_RF_gm_{}.txt'.format(iter)),response_p_wm)
            rec_response_signal = res_obj.on_sphere(sphere)
            rec_response_signal_sh = sf_to_sh(rec_response_signal,sphere,sh_order=sh_degree,basis_type=None,full_basis=full_basis)
            # transform our data from 1D to 4D
            rec_response_signal_sh = rec_response_signal_sh[None, None, None, :]
            data_new_img = nib.Nifti1Image(rec_response_signal_sh, affine=dmri_file_affine)
            nib.save(data_new_img, os.path.join(response_path,'rec_response_signal_sh_gm_{}.nii.gz'.format(iter)))
        if csf:
            response_p_csf = response_p_wmgmcsf[2]
            fodf_shc_csf = fodf_shc_wmgmcsf[2]
            odf_sphere = odfsh_sf(odf_sh=fodf_shc_csf,mask_bool=mask_bool_wmgmcsf[1],sh_degree=1,full_basis=full_basis)
            shm_coeff = np.zeros((csf_shape + (n_shm_coeff,)))
            B, invB = sh_to_sf_matrix(sphere, sh_degree, None, return_inv=True)
            for idx in ndindex(csf_shape):
                odf_sphere_ = odf_sphere[idx]
                shm_coeff[idx] = np.dot(odf_sphere_, invB)
                direction, pk, ind = peak_directions(odf_sphere_, sphere,relative_peak_threshold,min_separation_angle,is_symmetric=True)
                # Calculate peak metrics
                if pk.shape[0] != 0:
                    global_max = max(global_max, pk[0])
                    m = min(n_peaks, pk.shape[0])
                    qa_array_csf[idx][:m] = pk[:m] - odf_sphere_.min()
                    peak_dirs_csf[idx][:m] = direction[:m]
                    peak_indices_csf[idx][:m] = ind[:m]
                    peak_values_csf[idx][:m] = pk[:m]
            gfa_array = None
            odf_array = None
            peaks = pam_from_attrs(PeaksAndMetrics,sphere,peak_indices_csf,peak_values_csf,peak_dirs_csf,gfa_array,qa_array_csf,shm_coeff,B,odf_array)
            reshape_peak = reshape_peaks_for_visualization(peaks)
            indices = np.where(mask_bool_wmgmcsf[2])
            csf_mask_file = nib.load(f'{data_path}/csf_mask.nii.gz')  
            csf_mask = csf_mask_file.get_fdata()
            data_orgin = np.zeros((csf_mask.shape[0],csf_mask.shape[1],csf_mask.shape[2],n_peaks*3))
            data_orgin[indices] = reshape_peak
            peak_img = nib.Nifti1Image(data_orgin,affine = dmri_file_affine)
            nib.save(peak_img,os.path.join(response_path, 'peaks_csf.nii.gz'))
            single_peak_mask = (peak_values_csf[:, 1] / peak_values_csf[:, 0]) < peak_thr
            data_csf_new = dmri_wm_gm_csf_data_bool[2][single_peak_mask]
            dirs = peak_dirs_csf[single_peak_mask]
            for num_vox in range(data_csf_new.shape[0]):
                rotmat = vec2vec_rotmat(dirs[num_vox, 0], np.array([0, 0, 1]))
                rot_gradients = np.dot(rotmat, gtab.gradients.T).T
                x, y, z = rot_gradients[where_dwi].T
                r, theta, phi = cart2sphere(x, y, z)
                # for the gradient sphere
                B_dwi = real_sh_descoteaux_from_index(
                    0, n, theta[:, None], phi[:, None])
                r_sh_all += np.linalg.lstsq(B_dwi, data_csf_new[num_vox, where_dwi].astype('float32'),rcond=-1)[0]
            response = r_sh_all / data_csf_new.shape[0]
            change = abs((response_p_csf - response) / response_p_csf)
            if all(change < convergence):
                break
            response_p_csf = response
            res_obj = AxSymShResponse(data_csf_new[:, gtab.b0s_mask].mean(), response)
            if np.isnan(response_p_csf).any()==True:
                break
            else:
                np.savetxt(os.path.join(response_path,'recursive_RF_csf_{}.txt'.format(iter)),response_p_csf)
            rec_response_signal = res_obj.on_sphere(sphere)
            rec_response_signal_sh = sf_to_sh(rec_response_signal,sphere,sh_order=sh_degree,basis_type=None,full_basis=full_basis)
            # transform our data from 1D to 4D
            rec_response_signal_sh = rec_response_signal_sh[None, None, None, :]
            data_new_img = nib.Nifti1Image(rec_response_signal_sh, affine=dmri_file_affine)
            nib.save(data_new_img, os.path.join(response_path,'rec_response_signal_sh_csf_{}.nii.gz'.format(iter)))
        response_p_wmgmcsf = []
        if wm:  
            response_p_wmgmcsf.append(response_p_wm)
        if gm:
            response_p_wmgmcsf.append(response_p_gm)
        if csf:
            response_p_wmgmcsf.append(response_p_csf)
        print('response_p_wmgmcsf_iter',response_p_wmgmcsf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        default='datasets/marmoset',
        help='Root path of the data',
        type=str
    )
    parser.add_argument(
        '--load_state',
        help='Load a saved model (default: None)',
        type=str
    )
    parser.add_argument(
        '--n_peaks',
        default=3,
        help='number of peaks',
        type=int
    )
    parser.add_argument(
        '--iteration',
        default=5,
        help='number of recursive iterations',
        type=int
    )
    parser.add_argument(
        '--batch_size',
        default=64,
        help='Batch size (default: 64)',
        type=int
    )
    parser.add_argument(
        '--lr',
        default=0.0017,
        help='Learning rate (default: 0.0017)',
        type=float
    )
    parser.add_argument(
        '--epoch',
        default=5,
        help='Epoch (default: 5)',
        type=int
    )
    parser.add_argument(
        '--filter_start',
        help='Number of filters for the first convolution (default: 2)',
        default=1,
        type=int
    )
    parser.add_argument(
        '--sh_degree',
        help='Max spherical harmonic order (default: 10)',
        default=10,
        type=int
    )
    parser.add_argument(
        '--kernel_sizeSph',
        help='Spherical kernel size (default: 5)',
        default=5,
        type=int
    )
    parser.add_argument(
        '--kernel_sizeSpa',
        help='Spatial kernel size (default: 3)',
        default=3,
        type=int
    )
    parser.add_argument(
        '--depth',
        help='Graph subsample depth (default: 5)',
        default=5,
        type=int
    )
    parser.add_argument(
        '--n_side',
        help='Healpix resolution (default: 16)',
        default=16,
        type=int
    )
    parser.add_argument(
        '--save_every',
        help='Saving periodicity (default: 1)',
        default=1,
        type=int
    )
    parser.add_argument(
        '--loss_intensity',
        choices=('L1', 'L2'),
        default='L2',
        help='Objective function (default: L2)',
        type=str
    )
    parser.add_argument(
        '--intensity_weight',
        default=1.,
        help='Intensity weight (default: 1.)',
        type=float
    )
    parser.add_argument(
        '--loss_oddsensitive',
        default='L1',
        help='Objective function (default: L1)',
        type=str
    )
    parser.add_argument(
        '--oddsensitive_weight',
        default=1e-3,
        help='oddsensitive weight (default: 1e-3)',
        type=float
    )
    parser.add_argument(
        '--loss_smoothness',
        choices=('cauchy'),
        default='cauchy',
        help='Objective function (default: cauchy)',
        type=str
    )
    parser.add_argument(
        '--smoothness_weight',
        default=1e-04,
        help='Smoothness weight (default: 1e-04)',
        type=float
    )
    parser.add_argument(
        '--sigma_smoothness',
        default=1e-05,
        help='Sigma for odd smoothness (default: 1e-05)',
        type=float
    )

    parser.add_argument(
        '--pve_weight',
        default=1e-5,
        help='PVE regularizer weight (default: 1e-5)',
        type=float
    )

    parser.add_argument(
        '--loss_non_negativity',
        choices=('L1', 'L2'),
        default='L2',
        help='Objective function (default: L2)',
        type=str
    )
    parser.add_argument(
        '--nn_fodf_weight',
        default=1.,
        help='Non negativity fODF weight (default: 1.)',
        type=float
    )
    
    parser.add_argument(
        '--wm',
        action='store_true',
        default=True,
        help='Estimate white matter fODF (default: False)',
    )
    parser.add_argument(
        '--gm',
        action='store_true',
        help='Estimate grey matter fODF (default: False)',
    )
    parser.add_argument(
        '--csf',
        action='store_true',
        help='Estimate CSF fODF (default: False)',
    )
    parser.add_argument(
        '--full_basis',
        action='store_true',
        help='True to use a SH basis containing even and odd order SH functions.Else, use a SH basis consisting only of even order SH functions.'
    )
    parser.add_argument(
        '--patch_size',
        default=3,
        help='Patch size (default: 3)',
        type=int
    )
    parser.add_argument(
        '--graph_sampling',
        default='healpix',
        choices=('healpix', 'bvec'),
        help='Sampling used for the graph convolution, healpix or bvec (default: healpix)',
        type=str
    )
    parser.add_argument(
        '--conv_name',
        default='mixed',
        choices=('mixed', 'spherical'),
        help='Convolution name (default: mixed)',
        type=str
    )
    parser.add_argument(
        '--anisoSpa',
        action='store_true',
        help='Use anisotropic spatial filter (default: False)',
    )
    parser.add_argument(
        '--concatenate',
        action='store_true',
        help='Concatenate spherical features (default: False)',
    )
    parser.add_argument(
        '--project',
        default='default',
        help='Project name',
        type=str
    )
    parser.add_argument(
        '--expname',
        default='default',
        help='Exp name',
        type=str
    )
    parser.add_argument(
        '--middle_voxel',
        action='store_true',
        help='Concatenate spherical features (default: False)',
    )
    parser.add_argument(
        '--normalize',
        default=True,
        action='store_true',
        help='Norm the partial volume sum to be 1 (default: False)',
    )
    args = parser.parse_args()
    data_path = args.data_path
    expname = args.expname
    middle_voxel = args.middle_voxel
    sh_degree = args.sh_degree
    n_peaks = args.n_peaks
    iteration = args.iteration
    # Train properties
    batch_size = args.batch_size
    lr = args.lr
    n_epoch = args.epoch
    full_basis = args.full_basis
    full_basis = bool(full_basis)
    # Model architecture properties
    filter_start = args.filter_start
    sh_degree = args.sh_degree
    kernel_sizeSph = args.kernel_sizeSph
    kernel_sizeSpa = args.kernel_sizeSpa
    depth = args.depth
    n_side = args.n_side
    normalize = args.normalize
    patch_size = args.patch_size
    graph_sampling = args.graph_sampling
    conv_name = args.conv_name
    isoSpa = not args.anisoSpa
    concatenate = args.concatenate

    # Saving parameters
    save_every = args.save_every

    # Intensity loss
    loss_fn_intensity = args.loss_intensity
    intensity_weight = args.intensity_weight
   
    # Odd smoothness/sparisty loss
    loss_fn_smoothness = args.loss_smoothness
    smoothness_weight =args.smoothness_weight
    smoothness_sigma = args.sigma_smoothness
    # Non-negativity loss
    loss_fn_non_negativity = args.loss_non_negativity
    nn_fodf_weight = args.nn_fodf_weight
    # PVE loss
    pve_weight = args.pve_weight
    #Odd sensitivity loss
    loss_fn_oddsensitive = args.loss_oddsensitive
    oddsensitive_weight = args.oddsensitive_weight

    # Load pre-trained model and response functions
    load_state = args.load_state
    wm = args.wm
    gm = args.gm
    csf = args.csf

    sphere = get_sphere('symmetric362')
    global_max = -np.inf
    relative_peak_threshold = 0.2 
    min_separation_angle = 15 
    
    peak_thr = 0.2  
    convergence = 0.001
 
    main(data_path,full_basis,sh_degree,n_peaks,iteration,sphere, global_max, relative_peak_threshold,min_separation_angle,peak_thr,convergence,
         batch_size, lr, n_epoch, kernel_sizeSph, kernel_sizeSpa, 
         filter_start, depth, n_side, wm, gm, csf,
         loss_fn_intensity, loss_fn_non_negativity,loss_fn_smoothness,loss_fn_oddsensitive,
         intensity_weight, nn_fodf_weight, pve_weight,smoothness_weight,smoothness_sigma,oddsensitive_weight,
         save_every, normalize, load_state, patch_size, graph_sampling, conv_name, isoSpa, concatenate, middle_voxel)
    
    