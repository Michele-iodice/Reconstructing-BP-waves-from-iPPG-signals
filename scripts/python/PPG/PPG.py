import numpy as np


def signals_to_rppg(sig, method, params={}):
    """
    Transform an input RGB signal in a PPG signal using a rPPG
    method (see PPG.methods).
    This method must use and execute on CPU.
    You can pass also non-RGB signal but the method used must handle its shape.

    Args:
        sig (float32 ndarray): RGB Signal as float32 ndarray with shape  [num_estimators, rgb_channels, num_frames].
            You can pass also a generic signal but the method used must handle its shape and type.
        method: a method that comply with the fucntion signature documented
            in PPG.methods. This method must use Numpy.
        params (dict): dictionary of usefull parameters that will be passed to the method.

    Returns:
        float32 ndarray: rPPG signal as float32 ndarray with shape [num_estimators, num_frames].
    """
    if sig.shape[0] == 0:
        return np.zeros((0, sig.shape[2]), dtype=sig.dtype)
    cpu_sig = np.array(sig)
    if len(params) > 0:
        r_ppgs = method(cpu_sig, **params)
    else:
        r_ppgs = method(cpu_sig)
    return r_ppgs


def RGB_sig_to_rPPG(windowed_sig, fps, method=None, params={}):
    """
    Transform an input RGB windowed signal in a rPPG windowed signal using a rPPG method (see PPG.methods).
    You can pass also non-RGB signal but the method used must handle its shape.

    Args:
        windowed_sig (np.array): RGB windowed signal as a list of length num_windows of np.ndarray with shape [num_estimators, rgb_channels, num_frames].
        fps (float): frames per seconds. You can pass also a generic signal but the method used must handle its shape and type.
        method: a method that comply with the fucntion signature documented
            in pyVHR.BVP.methods. This method must use Numpy if the 'device_type' is 'cpu', Torch if the 'device_type' is 'torch', and Cupy
            if the 'device_type' is 'cuda'.
        params(dict): dictionary of usefull parameters that will be passed to the method. If the method needs fps you can set {'fps':'adaptive'}
            in the dictionary to use the 'fps' input variable.

    Returns:
        a list of lenght num_windows of rPPG signals as np.ndarray with shape [num_estimators, num_frames];
        if no rPPG can be found in a window, then the np.ndarray has num_estimators == 0.
    """

    if 'fps' in params and params['fps'] == 'adaptive':
        params['fps'] = np.float32(fps)

    r_ppgs = []
    for sig in windowed_sig:
        copy_signal = np.copy(sig)
        r_ppg = signals_to_rppg(copy_signal, method, params)

        # check for nan
        r_ppg_nonan = []
        for i in range(r_ppg.shape[0]):
           if not np.isnan(r_ppg[i]).any():
              r_ppg_nonan.append(r_ppg[i])
        if len(r_ppg_nonan) == 0:            # if empty
           r_ppgs.append(np.zeros((0, 1), dtype=np.float32))
        else:
           r_ppgs.append(np.array(r_ppg_nonan, dtype=np.float32))

    return np.array(r_ppgs)