import glob
import os
import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf,sh_to_rh,sph_harm_ind_list,real_sh_descoteaux_from_index,lazy_index
from dipy.core.ndindex import ndindex
from dipy.sims.voxel import single_tensor
from dipy.core.gradients import gradient_table
from dipy.core.geometry import cart2sphere

def get_latest_pth_file(folder_path):
    pth_files = glob.glob(os.path.join(folder_path, '*.pth'))
    pth_files.sort(key=os.path.getmtime, reverse=True)
    if pth_files:
        return pth_files[0]
    else:
        return None
    
def fa_trace_to_lambdas(fa=0.08, trace=0.0021):
    lambda1 = (trace / 3.) * (1 + 2 * fa / (3 - 2 * fa ** 2) ** (1 / 2.))
    lambda2 = (trace / 3.) * (1 - fa / (3 - 2 * fa ** 2) ** (1 / 2.))
    evals = np.array([lambda1, lambda2, lambda2])
    return evals

def create_dir(path):
    if not os.path.exists(path):
        print('Create new directory: {0}'.format(path))
        os.makedirs(path)
    return path

def odfsh_sf(odf_sh,mask_bool,sh_degree,full_basis):
    odf_sh = odf_sh[mask_bool]
    sphere = get_sphere('symmetric362')
    odf_sphere = np.zeros((odf_sh.shape[0],362)) 
    for idx in ndindex(odf_sh.shape[:-1]):
        if sh_degree>1:
            odf_sf = sh_to_sf(odf_sh[idx],sphere,sh_degree,basis_type=None,full_basis=full_basis, legacy=True)
            odf_sphere[idx] = odf_sf
    return odf_sphere

def pam_from_attrs(klass, sphere, peak_indices, peak_values, peak_dirs,
                    gfa, qa, shm_coeff, B, odf):
    this_pam = klass()
    this_pam.sphere = sphere
    this_pam.peak_dirs = peak_dirs
    this_pam.peak_values = peak_values
    this_pam.peak_indices = peak_indices
    this_pam.gfa = gfa
    this_pam.qa = qa
    this_pam.shm_coeff = shm_coeff
    this_pam.B = B
    this_pam.odf = odf
    return this_pam

def estimate_response(gtab, evals, S0):
    """ Estimate single fiber response function
    """
    evecs = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])

    return single_tensor(gtab, S0, evals, evecs, snr=None)

def generate_fatRF(data_path):
    bvecs = np.loadtxt(f'{data_path}/bvecs.bvecs')
    bvals = np.loadtxt(f'{data_path}/bvals.bvals')
    shell_not0_index = np.where(bvals!=0)[0]
    bvecs = bvecs[:,shell_not0_index]
    bvals = bvals[shell_not0_index]
    shellnumber = np.unique(bvals).shape[0]
    r_rh_all = []
    for m in range(shellnumber):
        bval_shell=np.unique(bvals)[m]  
        shellindex=np.where(bvals==bval_shell)[0]
        bvals_shell = bvals[shellindex]
        bvecs_shell = bvecs[:,shellindex]
        gtab = gradient_table(bvals_shell, bvecs_shell)
        evals = fa_trace_to_lambdas(0.08, 0.0021) 
        origin_res_obj = (evals, 1.)
        S_r = estimate_response(gtab, origin_res_obj[0],origin_res_obj[1])
        sh_order_max = 10
        m_values, l_values = sph_harm_ind_list(sh_order_max)
        _where_dwi = lazy_index(~gtab.b0s_mask)
        x, y, z = gtab.gradients[_where_dwi].T
        r, theta, phi = cart2sphere(x, y, z)
        B_dwi = real_sh_descoteaux_from_index(m_values, l_values, theta[:, None], phi[:, None])
        r_sh = np.linalg.lstsq(B_dwi, S_r[_where_dwi],rcond=-1)[0]
        l_response = l_values
        m_response = m_values
        r_rh = sh_to_rh(r_sh, m_response, l_response)
        r_rh_all.append(r_rh)
    r_rh_all_= np.expand_dims(np.array(r_rh_all), axis=0)
    return r_rh_all_