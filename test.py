import argparse
import os
import time
import json
import numpy as np
from utils.dataset import load_data
from test_model import spherical_model
from utils.tools import create_dir,generate_fatRF

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
    response_p_wmgmcsf = generate_fatRF(data_path)
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
    
    
