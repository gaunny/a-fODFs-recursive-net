import numpy as np
import torch

def load_response_function(response_p_wmgmcsf, wm, gm, csf, max_degree, n_shell, norm):
    """Response function loader.
    Args:
        rf_path (str): reponse function folder path
        wm (bool): Load white matter rf
        gm (bool): Load grey matter rf
        csf (bool): Load csf rf
        max_degree (int): Spherical harmonic degree of the sampling
    """
    # Load response functions
    filter_equi = None
    filter_inva = None
    if wm:
        rf_wm = response_p_wmgmcsf[0]
        if len(rf_wm.shape) == 1:
            rf_wm = rf_wm.reshape(1, len(rf_wm))
        if n_shell != rf_wm.shape[0]:  
            print("WM response function and shells doesn't match: ")
            print("WM rf: ", rf_wm.shape[0])
            print("Shell: ", n_shell)
            raise NotImplementedError
        if max_degree // 2 + 1 > rf_wm.shape[1]:
            print("WM response function doesn't have enough coefficients: ")
            print("WM rf: ", rf_wm.shape[1])
            print("Max order: ", max_degree // 2 + 1)
            k = max_degree // 2 + 1 - rf_wm.shape[1]
            rf_wm = np.hstack((rf_wm, np.zeros((rf_wm.shape[0], k))))
        rf_wm = rf_wm[:, :max_degree // 2 + 1] 
        filter_equi=torch.Tensor(rf_wm[None])
    
    if gm or csf:
        filter_inva = []
        if gm:
            rf_gm = response_p_wmgmcsf[1]
            if rf_gm.shape == ():
                rf_gm = np.array([rf_gm])
            if len(rf_gm.shape) != 1:
                print("GM response function has too many dimension: ")
                print("GM rf: ", rf_gm.shape)
                print("Should be: ", 1)
                raise NotImplementedError
            if n_shell != rf_gm.shape[0]:
                print("Response function and shells doesn't match: ")
                print("GM rf: ", rf_gm.shape[0])
                print("Should be: ", n_shell)
                raise NotImplementedError
            rf_gm = rf_gm.reshape(n_shell, 1) / norm[:, None]
            print('GM rf shape: ', rf_gm.shape)
            print(rf_gm)
            filter_inva.append(torch.Tensor(rf_gm[None]))
        if csf:
            rf_csf = response_p_wmgmcsf[2]
            if rf_csf.shape == ():
                rf_csf = np.array([rf_csf])
            if len(rf_csf.shape) != 1:
                print("CSF response function has too many dimension: ")
                print("CSF rf: ", rf_csf.shape)
                print("Should be: ", 1)
                raise NotImplementedError
            if n_shell != rf_csf.shape[0]:
                print("Response function and shells doesn't match: ")
                print("CSF rf: ", rf_csf.shape[0])
                print("Should be: ", n_shell)
                raise NotImplementedError
            rf_csf = rf_csf.reshape(n_shell, 1) / norm[:, None]
            filter_inva.append(torch.Tensor(rf_csf[None]))
        filter_inva = torch.cat(filter_inva)
    return filter_equi, filter_inva