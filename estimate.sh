#!/bin/bash
# data_path should contain :dmri_data.nii.gz bvals.bvals bvecs.bvecs
python main.py \
    --data_path /data1/esd/fibercup/ \
    --sh_degree 8 \
    --n_peaks 3 \
    --iteration 5 \
    --batch_size 64 \
    --lr 0.0017 \
    --epoch 5 \
    --filter_start 2 \
    --kernel_sizeSph 5 \
    --kernel_sizeSpa 3 \
    --depth 4 \
    --n_side 16 \
    --save_every 1 \
    --loss_intensity L2 \
    --intensity_weight 1 \
    --loss_oddsensitive L1 \
    --oddsensitive_weight 1e-04 \
    --loss_smoothness cauchy \
    --smoothness_weight 1e-04 \
    --sigma_smoothness 1e-05\
    --loss_non_negativity L2  \
    --nn_fodf_weight 1 \
    --wm \
    --full_basis \
    --patch_size 3 \
    --graph_sampling healpix\
    --conv_name mixed \
    --pve_weight 1e-11
    
