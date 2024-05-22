import cv2
import numpy as np


class MagicLandmarks:
    """
    This class contains usefull lists of landmarks identification numbers.
    """

    # more specific areas
    forehead_left = [21, 71, 68, 54, 103, 104, 63, 70,
                     53, 52, 65, 107, 66, 108, 69, 67, 109, 105]
    forehead_center = [10, 151, 9, 8, 107, 336, 285, 55, 8]
    forehoead_right = [338, 337, 336, 296, 285, 295, 282,
                       334, 293, 301, 251, 298, 333, 299, 297, 332, 284]
    cheek_left_bottom = [215, 138, 135, 210, 212, 57, 216, 207, 192]
    cheek_right_bottom = [435, 427, 416, 364,
                          394, 422, 287, 410, 434, 436]
    cheek_left_top = [116, 111, 117, 118, 119, 100, 47, 126, 101, 123,
                      137, 177, 50, 36, 209, 129, 205, 147, 177, 215, 187, 207, 206, 203]
    cheek_right_top = [349, 348, 347, 346, 345, 447, 323,
                       280, 352, 330, 371, 358, 423, 426, 425, 427, 411, 376]
    # dense zones used for convex hull masks
    left_eye = [157, 144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55, 56, 189, 190, 63, 65, 66, 70, 221, 222,
                223, 225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124]
    right_eye = [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 413, 293, 296, 300, 441, 442, 445, 446,
                 449, 451, 334, 463, 336, 464, 467, 339, 341, 342, 353, 381, 373, 249, 253, 255]
    mounth = [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181, 313, 314, 186, 57, 315, 61, 321, 73, 76,
              335, 83, 85, 90, 106]


def get_magic_landmarks():
    """ returns high_priority and mid_priority list of landmarks identification number """
    return [*MagicLandmarks.forehead_center, *MagicLandmarks.cheek_left_bottom, *MagicLandmarks.cheek_right_bottom], [
        *MagicLandmarks.forehoead_right, *MagicLandmarks.forehead_left, *MagicLandmarks.cheek_left_top,
        *MagicLandmarks.cheek_right_top]


def sig_windowing(sig, wsize, stride, fps):
    """
    This method is used to divide a RGB signal into overlapping windows.

    Args:
        sig (float32 ndarray): ndarray with shape [num_frames, num_estimators, rgb_channels].
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        A list of ndarray (float32) with shape [num_estimators, rgb_channels, window_frames],
        an array (float32) of times in seconds (win centers)
    """
    N = sig.shape[0]
    block_idx, timesES = sliding_straded_win_idx(N, wsize, stride, fps)
    block_signals = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(sig[st_frame: end_frame + 1])
        wind_signal = np.swapaxes(wind_signal, 0, 1)
        wind_signal = np.swapaxes(wind_signal, 1, 2)
        block_signals.append(wind_signal)
    return block_signals, timesES

    """
    This method is used to divide a Raw signal into overlapping windows.

    Args:
        sig (float32 ndarray): ndarray of images with shape [num_frames, rows, columns, rgb_channels].
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        windowed signal as a list of length num_windows of float32 ndarray with shape [num_frames, rows, columns, 
        rgb_channels],
        and a 1D ndarray of times in seconds,where each one is the center of a window.
    """
    N = raw_signal.shape[0]
    block_idx, timesES = sliding_straded_win_idx(N, wsize, stride, fps)
    block_signals = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(raw_signal[st_frame: end_frame + 1])
        # check for zero traces
        sum_wind = np.sum(wind_signal, axis=(1, 2))
        zero_idx = np.argwhere(sum_wind == 0).squeeze()
        est_idx = np.ones(wind_signal.shape[0], dtype=bool)
        est_idx[zero_idx] = False
        # append traces
        block_signals.append(wind_signal[est_idx])
    return block_signals, timesES


def sliding_straded_win_idx(N, wsize, stride, fps):
    """
    This method is used to compute the indices for creating an overlapping windows signal.

    Args:
        N (int): length of the signal.
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        List of ranges, each one contains the indices of a window, and a 1D ndarray of times in seconds, where each one
        is the center of a window.
    """
    wsize_fr = wsize * fps
    stride_fr = stride * fps
    idx = []
    timesES = []
    num_win = int((N - wsize_fr) / stride_fr) + 1
    s = 0
    for i in range(num_win):
        idx.append(np.arange(s, s + wsize_fr))
        s += stride_fr
        timesES.append(wsize / 2 + stride * i)
    return idx, np.array(timesES, dtype=np.float32)


def get_fps(videoFileName):
    """
    This method returns the fps of a video file name or path.
    """
    vidcap = cv2.VideoCapture(videoFileName)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return fps


def extract_frames_yield(videoFileName):
    """
    This method yield the frames of a video file name or path.
    """
    vidcap = cv2.VideoCapture(videoFileName)
    success, image = vidcap.read()
    while success:
        yield image
        success, image = vidcap.read()
    vidcap.release()