import numpy as np
from numba import prange, njit

"""
This module defines classes or methods used for signal extraction.
"""


class SignalProcessingParams:
    """
        This class contains usefull parameters used by this module.

        RGB_LOW_TH (numpy.int32): RGB low-threshold value.

        RGB_HIGH_TH (numpy.int32): RGB high-threshold value.
    """
    RGB_LOW_TH = np.int32(55)
    RGB_HIGH_TH = np.int32(200)


@njit(['float32[:,:](uint8[:,:,:], int32, int32)', ], parallel=True, fastmath=True, nogil=True)
def holistic_mean(im, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method computes the RGB-Mean Signal excluding 'im' pixels
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).

    Args: 
        im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
        RGB_LOW_TH (numpy.int32): RGB low threshold value.
        RGB_HIGH_TH (numpy.int32): RGB high threshold value.
    
    Returns:
        RGB-Mean Signal as float32 ndarray with shape [1,3], where 1 is the single estimator,
        and 3 are r-mean, g-mean and b-mean.
    """
    mean = np.zeros((1, 3), dtype=np.float32)
    mean_r = np.float32(0.0)
    mean_g = np.float32(0.0)
    mean_b = np.float32(0.0)
    num_elems = np.float32(0.0)
    for x in prange(im.shape[0]):
        for y in prange(im.shape[1]):
            if not((im[x, y, 0] <= RGB_LOW_TH and im[x, y, 1] <= RGB_LOW_TH and im[x, y, 2] <= RGB_LOW_TH)
                    or (im[x, y, 0] >= RGB_HIGH_TH and im[x, y, 1] >= RGB_HIGH_TH and im[x, y, 2] >= RGB_HIGH_TH)):
                mean_r += im[x, y, 0]
                mean_g += im[x, y, 1]
                mean_b += im[x, y, 2]
                num_elems += 1.0
    if num_elems > 1.0:
        mean[0, 0] = mean_r / num_elems
        mean[0, 1] = mean_g / num_elems
        mean[0, 2] = mean_b / num_elems
    else:
        mean[0, 0] = mean_r
        mean[0, 1] = mean_g
        mean[0, 2] = mean_b 
    return mean