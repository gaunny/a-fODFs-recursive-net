import numpy as np
import nibabel as nib
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
from dipy.tracking import utils
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
import time

# =========================================================
# 1. Configuration & Paths
# =========================================================
fodf_path   = "fodf_asym.nii.gz"
peaks_path  = "asym_peaks.nii.gz"
mask_path   = "wm_mask.nii.gz"
asi_path    = "asi_map.nii.gz"
out_tck     = "streamlines.tck"

# Tracking parameters
step_size_vox  = 0.5    # Step size in voxel units
min_length_mm  = 20.0   # Minimum streamline length in mm
max_length_mm  = 250.0  # Maximum streamline length in mm
theta_global   = 70.0   # Maximum turning angle in degrees
seed_density   = 2      # Number of seeds per voxel side
use_asi        = True   # Use Asymmetric Index for adaptive theta
deterministic  = False  # False for Probabilistic (FODF), True for Deterministic (Peaks)

def asi_to_theta(theta_global_deg, asi_value):
    """ Scales the max angle based on ASI: ASI 0 -> 0.5*theta, ASI 1 -> 1.0*theta """
    return theta_global_deg * (0.5 + 0.5 * np.clip(asi_value, 0, 1))

# =========================================================
# 2. Data Loading & Initialization
# =========================================================
print("Loading datasets...")
fodf_img = nib.load(fodf_path)
affine   = fodf_img.affine
# Get physical voxel dimensions for mm-length calculations
voxel_size = np.mean(fodf_img.header.get_zooms()[:3])

fodf_data  = fodf_img.get_fdata()
peaks_data = nib.load(peaks_path).get_fdata()
mask_data  = nib.load(mask_path).get_fdata().astype(bool)
asi_data   = nib.load(asi_path).get_fdata() if use_asi else None
X, Y, Z    = mask_data.shape

# Setup sphere for FODF sampling
sphere = get_sphere('repulsion724')
n_dirs = len(sphere.vertices)

# Voxel-space alignment: ISMRM datasets often require X-axis flip for FODF alignment
v_flipped = sphere.vertices * np.array([-1, 1, 1])

print("Converting SH coefficients to SF (Spherical Function)...")
fodf_sf = sh_to_sf(fodf_data, sphere, sh_order=6, full_basis=True)
fodf_sf = np.clip(fodf_sf, 0, None)
fodf_sf /= np.maximum(fodf_sf.max(axis=-1, keepdims=True), 1e-6)

# Generate seeds in Voxel Space (affine=Identity)
seeds_v = utils.seeds_from_mask(mask_data, affine=np.eye(4), density=seed_density)

# =========================================================
# 3. Tracking Engine (Bidirectional)
# =========================================================
start_time = time.time()
final_streamlines_v = []
# Calculate max steps for half-length to ensure bidirectional total <= max_length
max_steps_half = int(max_length_mm / (step_size_vox * voxel_size * 2))

print(f"Tracking on {len(seeds_v)} seeds...")

for s_idx, seed in enumerate(seeds_v):
    tracks = []
    # Bidirectional loop: grow forward and backward from seed
    for direction_sign in [1, -1]:
        pos = seed.copy()
        prev_dir = None
        sl_dir = [pos.copy()]

        for _ in range(max_steps_half):
            vi, vj, vk = np.round(pos).astype(int)
            
            # Boundary & Mask check
            if not (0 <= vi < X and 0 <= vj < Y and 0 <= vk < Z) or not mask_data[vi, vj, vk]:
                break

            # Adaptive theta calculation
            current_asi = asi_data[vi, vj, vk] if use_asi else 1.0
            theta_limit = np.deg2rad(asi_to_theta(theta_global, current_asi))

            if deterministic:
                # DETERMINISTIC: Follow Peaks
                voxel_peaks = peaks_data[vi, vj, vk]
                if prev_dir is None:
                    new_dir = voxel_peaks[0] * direction_sign
                else:
                    cos_sims = np.dot(voxel_peaks, prev_dir)
                    best_idx = np.argmax(np.abs(cos_sims))
                    new_dir = voxel_peaks[best_idx]
                    # Ensure direction continuity
                    if np.dot(new_dir, prev_dir) < 0:
                        new_dir = -new_dir
                    # Angle constraint check
                    if np.dot(new_dir, prev_dir) < np.cos(theta_limit):
                        break
            else:
                # PROBABILISTIC: Sample from FODF PMF
                pmf = fodf_sf[vi, vj, vk].copy()
                if prev_dir is not None:
                    cos_sim = np.dot(v_flipped, prev_dir)
                    # Mask out directions exceeding theta_limit or in the back hemisphere
                    pmf[np.arccos(np.clip(np.abs(cos_sim), -1, 1)) > theta_limit] = 0
                    pmf = pmf * np.clip(cos_sim, 0, 1)
                
                if pmf.sum() <= 0: 
                    break
                    
                pmf /= pmf.sum()
                idx = np.random.choice(n_dirs, p=pmf)
                new_dir = v_flipped[idx]

            # Step forward in voxel units
            pos += new_dir * step_size_vox
            sl_dir.append(pos.copy())
            prev_dir = new_dir
        
        tracks.append(sl_dir)

    # Merge backward (reversed) and forward tracks
    full_sl_v = np.array(tracks[0][::-1][:-1] + tracks[1])
    
    # Filter by physical length (mm)
    if len(full_sl_v) * step_size_vox * voxel_size >= min_length_mm:
        final_streamlines_v.append(full_sl_v)

    if (s_idx + 1) % 1000 == 0:
        print(f"Processed {s_idx + 1}/{len(seeds_v)} seeds...")

# =========================================================
# 4. Coordinate Transformation & Saving
# =========================================================
print(f"Total streamlines found: {len(final_streamlines_v)}")

if final_streamlines_v:
    final_streamlines_ras = []
    # Convert Voxel streamlines to RASMM space (World Coordinates)
    # Formula: P_ras = Affine * [P_voxel, 1]^T
    for sl_v in final_streamlines_v:
        homo_v = np.hstack([sl_v, np.ones((len(sl_v), 1))])
        ras_coords = np.dot(affine, homo_v.T).T[:, :3]
        final_streamlines_ras.append(ras_coords.astype(np.float32))

    # Initialize StatefulTractogram to link streamlines with NIfTI header info
    sft = StatefulTractogram(final_streamlines_ras, fodf_img, Space.RASMM)
    save_tractogram(sft, out_tck)
    print(f"Successfully saved tracking results to: {out_tck}")
else:
    print("No streamlines generated.")
