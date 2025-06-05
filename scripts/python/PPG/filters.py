import numpy as np
from scipy import interpolate, sparse
from scipy.sparse.linalg import spsolve


"""
This module contains a collection of filter methods.

FILTER METHOD SIGNATURE
A Filter method must accept theese parameters:
    > signal -> RGB signal as float32 ndarray with shape [num_estimators, rgb_channels, num_frames],
                or BVP signal as float32 ndarray with shape [num_estimators, num_frames].
    > **kargs [OPTIONAL] -> usefull parameters passed to the filter method.
It must return a filtered signal with the same shape as the input signal.
"""


def apply_ppg_filter(windowed_sig, filter_func, fps = None, params={}):
    """
    Apply a filter method to a windowed rPPG signal.

    Args:
        windowed_sig: list of length num_window of rPPG signal as float32 ndarray with shape [num_estimators, num_frames]
        filter_func: filter method that accept a 'windowed_sig' (implements some filters in PPG.filters).
        params (dict): usefull parameters passed to the filter method.

    Returns:
        A filtered signal with the same shape as the input signal.
    """

    if 'fps' in params and params['fps'] == 'adaptive' and fps is not None:
        params['fps'] = np.float32(fps)
    filtered_windowed_sig = []
    for idx in range(len(windowed_sig)):
        transform = False
        sig = np.copy(windowed_sig[idx])
        if len(sig.shape) == 2:
            transform = True
            sig = np.expand_dims(sig, axis=1)
        if params == {}:
            filt_temp = filter_func(sig)
        else:
            filt_temp = filter_func(sig, **params)
        if transform:
            filt_temp = np.squeeze(filt_temp, axis=1)

        filtered_windowed_sig.append(filt_temp)

    return filtered_windowed_sig


# ------------------------------------------------------------------------------------- #
#                                     FILTER METHODS                                    #
# ------------------------------------------------------------------------------------- #


def interpolation(sig, **kargs):
    fps = kargs['fps']
    time = np.linspace(0, (len(sig) - 1) / fps, int(len(sig) * (100 / fps)))
    x = np.linspace(0, (len(sig) - 1) / fps, len(sig))
    if len(x) != len(sig):
        min_len = min(len(x), len(sig))
        x = x[:min_len]
        sig = sig[:min_len]

    interp_func = interpolate.interp1d(x, sig, kind='linear')
    signal = interp_func(time)

    return signal

def detrend(sig):

    lambda_ = 470  # Smoothing parameter
    T = len(sig)

    # Identity matrix (sparse)
    I = sparse.eye(T)

    # Second-order difference matrix D (sparse)
    data = [np.ones(T), -2 * np.ones(T), np.ones(T)]
    offsets = [0, 1, 2]
    D2 = sparse.diags(data, offsets, shape=(T - 2, T))

    # Solve (I + λ^2 * D^T D) * z = signal
    H = I + lambda_ ** 2 * D2.T @ D2
    z = spsolve(H, sig)
    signal = sig - z

    return signal