import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt
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


def apply_filter(windowed_sig, filter_func, fps = None, params={}):
    """
    Apply a filter method to a windowed RGB signal.

    Args:
        windowed_sig: list of length num_window of RGB signal as float32 ndarray with shape [num_estimators, rgb_channels, num_frames],
                      or BVP signal as float32 ndarray with shape [num_estimators, num_frames].
        filter_func: filter method that accept a 'windowed_sig' (pyVHR implements some filters in pyVHR.BVP.filters).
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

def BPfilter(sig, **kargs):
    """
    Band Pass filter (using BPM band) for RGB signal.

    The dictionary parameters are: {'minHz':float, 'maxHz':float, 'fps':float, 'order':int}
    """
    x = np.array(sig)
    b, a = butter(kargs['order'], Wn=[kargs['minHz'],
                                      kargs['maxHz']], fs=kargs['fps'], btype='bandpass')
    y = filtfilt(b, a, x, axis=0, padlen=(x.shape[0]-1))
    return y


def zscore(sig):
    """
    Z-score filter for RGB signal.
    """
    x = np.array(sig)
    y = stats.zscore(x, axis=2)
    return y


def zscorerange(sig, **kargs):
    """
    Z-score filter for RGB signal.
    """
    x = np.array(sig)
    y = stats.zscore(x, axis=2)
    y = np.clip(y, kargs['minR'], kargs['maxR'])
    return y


def zeromean(X):
    """
    Zero Mean filter for RGB signal.
    """
    M = np.mean(X, axis=2)
    return X - np.expand_dims(M, axis=2)


def squaring(sig):
    """
    Squaring filter for RGB signal.
    """
    x = np.array(sig)
    y = np.square(x)
    return y


def clipping(sig):
    """
    clipping filter for RGB signal.
    """
    x = np.array(sig)
    clipped_signal = np.clip(x, 0, np.inf)
    y = clipped_signal
    return y
