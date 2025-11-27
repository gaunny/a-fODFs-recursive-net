import numpy as np
import nibabel as nib
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from nibabel.streamlines import Tractogram
import time

fodf_path   = "fodf_asym.nii.gz"
peaks_path  = "asym_peaks_optimized.nii.gz"
mask_path   = "wm_mask.nii.gz"
asi_path    = "asi_map.nii.gz"
out_tck     = "streamlines.tck"

step_size   = 0.5      # mm
min_length  = 10       # mm
max_length  = 250      # mm
theta_global= 70       # degree
seed_density= 2
use_asi     = True
deterministic = False  # True -> use peaks direction

def asi_to_theta(theta_global_deg, asi_value):
    """
    ASI 0 -> 0.5*theta_global, ASI 1 -> 1.0*theta_global
    """
    return theta_global_deg * (0.5 + 0.5 * np.clip(asi_value, 0, 1))

fodf_img  = nib.load(fodf_path)
peaks_img = nib.load(peaks_path)
mask_img  = nib.load(mask_path)
asi_img   = nib.load(asi_path) if use_asi else None

fodf  = fodf_img.get_fdata()
peaks = peaks_img.get_fdata()  # shape: (X,Y,Z,n_peaks,3)
mask  = mask_img.get_fdata().astype(bool)
asi   = asi_img.get_fdata() if use_asi else None
affine = fodf_img.affine
inv_aff = np.linalg.inv(affine)
X, Y, Z = mask.shape

# ================== Sphere ==================
sphere = get_sphere('repulsion724')
n_dirs = len(sphere.vertices)

# ================== Stop ==================
stopping_criterion = BinaryStoppingCriterion(mask)

# ================== Seed ==================
seeds = utils.seeds_from_mask(mask, affine=affine, density=seed_density)
if len(seeds) == 0:
    raise RuntimeError("No seeds found!")

# ================== FODF -> spherical values ==================
fodf_sf = sh_to_sf(fodf, sphere, sh_order=6, full_basis=True)  # shape = (X,Y,Z,n_dirs)
fodf_sf = np.clip(fodf_sf, 0, None)
fodf_sf /= np.maximum(fodf_sf.max(axis=-1, keepdims=True), 1e-6)

# ================== Tracking ==================
start_time = time.time()
streamlines = []

for s_idx, seed in enumerate(seeds):
    pos = seed[:3].copy()
    prev_dir = None
    sl = [pos.copy()]

    for step in range(int(max_length / step_size)):
        voxel_f = nib.affines.apply_affine(inv_aff, pos)
        vi, vj, vk = np.round(voxel_f).astype(int)
        vi = np.clip(vi, 0, X - 1)
        vj = np.clip(vj, 0, Y - 1)
        vk = np.clip(vk, 0, Z - 1)

        if not mask[vi, vj, vk]:
            break

        pmf = fodf_sf[vi, vj, vk].copy()
        if pmf.sum() <= 0:
            break

        # Calculate the ASI adaptive Angle threshold
        theta_local = np.deg2rad(theta_global)
        if use_asi:
            theta_local = np.deg2rad(asi_to_theta(theta_global, asi[vi, vj, vk]))

        # Limit the Angle with the previous direction
        if prev_dir is not None:
            cos_sim = np.dot(sphere.vertices, prev_dir)
            cos_sim = np.clip(cos_sim, -1, 1)
            angles = np.arccos(np.abs(cos_sim))
            pmf[angles > theta_local] = 0
            pmf = pmf * np.clip(cos_sim, 0, 1)

        if pmf.sum() <= 0:
            break

        pmf /= pmf.sum()

        if deterministic:
            # deterministic, asym peaks
            peak_dirs = peaks[vi, vj, vk]  # shape: (n_peaks,3)
            if prev_dir is not None:
                cos_sim_peak = np.dot(peak_dirs, prev_dir)
                idx = np.argmax(np.abs(cos_sim_peak))
                new_dir = peak_dirs[idx]
                if np.dot(new_dir, prev_dir) < 0:
                    new_dir = -new_dir
            else:
                new_dir = peak_dirs[0]
        else:
            # probabilistic, FODF
            idx = np.random.choice(n_dirs, p=pmf)
            new_dir = sphere.vertices[idx]
            if prev_dir is not None and np.dot(new_dir, prev_dir) < 0:
                new_dir = -new_dir

        pos = pos + new_dir * step_size
        sl.append(pos.copy())
        prev_dir = new_dir

        if len(sl) * step_size >= max_length:
            break

    if len(sl) * step_size >= min_length:
        streamlines.append(np.array(sl))

    if (s_idx + 1) % 50 == 0:
        print(f"Processed {s_idx + 1}/{len(seeds)} seeds, streamlines so far: {len(streamlines)}")

elapsed = time.time() - start_time
print(f"Tracking done in {elapsed:.1f}s, total streamlines: {len(streamlines)}")

# ================== save TCK ==================
if len(streamlines) > 0:
    tractogram = Tractogram(streamlines=streamlines, affine_to_rasmm=affine)
    nib.streamlines.save(tractogram, out_tck)
    print("Saved:", out_tck)
else:
    print("No streamlines generated.")

