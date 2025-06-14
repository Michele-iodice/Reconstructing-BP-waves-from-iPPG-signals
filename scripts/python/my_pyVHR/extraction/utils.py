import cv2
import numpy as np
from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
    cos, diag, dot, eye, float32, float64, matrix, multiply, ndarray, newaxis, \
    sign, sin, sqrt, zeros


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
    N = sig.shape[1]
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

def raw_windowing(raw_signal, wsize, stride, fps):
    """
    This method is used to divide a Raw signal into overlapping windows.

    Args:
        raw_signal (float32 ndarray): ndarray of images with shape [num_frames, rows, columns, rgb_channels].
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

def ppg_sig_windowing(sig, wsize, stride, fps):
    """
    This method is used to divide a rPPG signal into overlapping windows.

    Args:
        sig (float32 ndarray): ndarray with shape [num_estimators, num_frames].
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        A list of ndarray (float32) with shape [num_estimators, window_frames],
        an array (float32) of times in seconds (win centers)
    """
    N = sig.shape[1]
    block_idx, timesES = sliding_straded_win_idx(N, wsize, stride, fps)
    block_signals = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(sig[:, st_frame:end_frame + 1])
        block_signals.append(wind_signal.astype(np.float32))
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


def extract_frames_yield(videoFileName, frame_interval=10):
    vidcap = cv2.VideoCapture(videoFileName)
    frame_count = 0
    success, image = vidcap.read()

    while success:
        if frame_count % frame_interval == 0:
            yield image
        success, image = vidcap.read()
        frame_count += 1

    vidcap.release()


def jadeR(X, m=None, verbose=True):
    """
    Blind separation of real signals with JADE.
    jadeR implements JADE, an Independent Component Analysis (ICA) algorithm
    developed by Jean-Francois Cardoso. See http://www.tsi.enst.fr/~cardoso/guidesepsou.html , and papers cited
    at the end of the source file.

    Translated into NumPy from the original Matlab Version 1.8 (May 2005) by
    Gabriel Beckers, http://gbeckers.nl .

    Parameters:
        X -- an nxT data matrix (n sensors, T samples). May be a numpy array or
             matrix.
        m -- output matrix B has size mxn so that only m sources are
             extracted.  This is done by restricting the operation of jadeR
             to the m first principal components. Defaults to None, in which
             case m=n.
        verbose -- print info on progress. Default is True.

    Returns:
        An m*n matrix B (NumPy matrix type), such that Y=B*X are separated
        sources extracted from the n*T data matrix X. If m is omitted, B is a
        square n*n matrix (as many sources as sensors). The rows of B are
        ordered such that the columns of pinv(B) are in order of decreasing
        norm; this has the effect that the `most energetically significant`
        components appear first in the rows of Y=B*X.

    Quick notes (more at the end of this file):
    o This code is for REAL-valued signals.  A MATLAB implementation of JADE
    for both real and complex signals is also available from
    http://sig.enst.fr/~cardoso/stuff.html
    o This algorithm differs from the first released implementations of
    JADE in that it has been optimized to deal more efficiently
    1) with real signals (as opposed to complex)
    2) with the case when the ICA model does not necessarily hold.
    o There is a practical limit to the number of independent
    components that can be extracted with this implementation.  Note
    that the first step of JADE amounts to a PCA with dimensionality
    reduction from n to m (which defaults to n).  In practice m
    cannot be `very large` (more than 40, 50, 60... depending on
    available memory)
    o See more notes, references and revision history at the end of
    this file and more stuff on the WEB
    http://sig.enst.fr/~cardoso/stuff.html
    o For more info on NumPy translation, see the end of this file.
    o This code is supposed to do a good job!  Please report any
    problem relating to the NumPY code gabriel@gbeckers.nl

    Copyright original Matlab code : Jean-Francois Cardoso <cardoso@sig.enst.fr>
    Copyright Numpy translation : Gabriel Beckers <gabriel@gbeckers.nl>
    """

    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for X.

    assert isinstance(X, ndarray), \
        "X (input data matrix) is of the wrong type (%s)" % type(X)
    origtype = X.dtype  # remember to return matrix B of the same type
    X = matrix(X.astype(float64))
    assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
    assert (verbose == True) or (verbose == False), \
        "verbose parameter should be either True or False"

    [n, T] = X.shape  # GB: n is number of input signals, T is number of samples

    if m == None:
        m = n  # Number of sources defaults to # of sensors
    assert m <= n, \
        "jade -> Do not ask more sources (%d) than sensors (%d )here!!!" % (m, n)

    if verbose:
        print("jade -> Looking for %d sources" % m)
        print("jade -> Removing the mean value")
    X -= X.mean(1)

    # whitening & projection onto signal subspace
    # ===========================================
    if verbose:
        print("jade -> Whitening the data")
    [D, U] = np.linalg.eig((X * X.T) / float(T))  # An eigen basis for the sample covariance matrix
    k = D.argsort()
    Ds = D[k]  # Sort by increasing variances
    PCs = arange(n - 1, n - m - 1, -1)  # The m most significant princip. comp. by decreasing variance

    # --- PCA  ----------------------------------------------------------
    B = U[:, k[PCs]].T  # % At this stage, B does the PCA on m components

    # --- Scaling  ------------------------------------------------------
    scales = sqrt(Ds[PCs])  # The scales of the principal components .
    B = diag(1. / scales) * B  # Now, B does PCA followed by a rescaling = sphering
    # B[-1,:] = -B[-1,:] # GB: to make it compatible with octave
    # --- Sphering ------------------------------------------------------
    X = B * X  # %% We have done the easy part: B is a whitening matrix and X is white.

    del U, D, Ds, k, PCs, scales

    # NOTE: At this stage, X is a PCA analysis in m components of the real data, except that
    # all its entries now have unit variance.  Any further rotation of X will preserve the
    # property that X is a vector of uncorrelated components.  It remains to find the
    # rotation matrix such that the entries of X are not only uncorrelated but also `as
    # independent as possible".  This independence is measured by correlations of order
    # higher than 2.  We have defined such a measure of independence which
    #   1) is a reasonable approximation of the mutual information
    #   2) can be optimized by a `fast algorithm"
    # This measure of independence also corresponds to the `diagonality" of a set of
    # cumulant matrices.  The code below finds the `missing rotation " as the matrix which
    # best diagonalizes a particular set of cumulant matrices.

    # Estimation of the cumulant matrices.
    # ====================================
    if verbose:
        print("jade -> Estimating cumulant matrices")

    # Reshaping of the data, hoping to speed up things a little bit...
    X = X.T
    dimsymm = int((m * (m + 1)) / 2)  # Dim. of the space of real symm matrices
    nbcm = dimsymm  # number of cumulant matrices
    CM = matrix(zeros([m, m * nbcm], dtype=float64))  # Storage for cumulant matrices
    R = matrix(eye(m, dtype=float64))
    Qij = matrix(zeros([m, m], dtype=float64))  # Temp for a cum. matrix
    Xim = zeros(m, dtype=float64)  # Temp
    Xijm = zeros(m, dtype=float64)  # Temp
    # Uns = numpy.ones([1,m], dtype=numpy.uint32)    # for convenience
    # GB: we don't translate that one because NumPy doesn't need Tony's rule

    # I am using a symmetry trick to save storage.  I should write a short note one of these
    # days explaining what is going on here.
    Range = arange(m)  # will index the columns of CM where to store the cum. mats.

    for im in range(m):
        Xim = X[:, im]
        Xijm = multiply(Xim, Xim)
        # Note to myself: the -R on next line can be removed: it does not affect
        # the joint diagonalization criterion
        Qij = multiply(Xijm, X).T * X / float(T) \
              - R - 2 * dot(R[:, im], R[:, im].T)
        CM[:, Range] = Qij
        Range = Range + m
        for jm in range(im):
            Xijm = multiply(Xim, X[:, jm])
            Qij = sqrt(2) * multiply(Xijm, X).T * X / float(T) \
                  - R[:, im] * R[:, jm].T - R[:, jm] * R[:, im].T
            CM[:, Range] = Qij
            Range = Range + m

    # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big m x m*nbcm array.

    V = matrix(eye(m, dtype=float64))

    Diag = zeros(m, dtype=float64)
    On = 0.0
    Range = arange(m)
    for im in range(nbcm):
        Diag = diag(CM[:, Range])
        On = On + (Diag * Diag).sum(axis=0)
        Range = Range + m
    Off = (multiply(CM, CM).sum(axis=0)).sum(axis=0) - On

    seuil = 1.0e-6 / sqrt(T)  # % A statistically scaled threshold on `small" angles
    encore = True
    sweep = 0  # % sweep number
    updates = 0  # % Total number of rotations
    upds = 0  # % Number of rotations in a given seep
    g = zeros([2, nbcm], dtype=float64)
    gg = zeros([2, 2], dtype=float64)
    G = zeros([2, 2], dtype=float64)
    c = 0
    s = 0
    ton = 0
    toff = 0
    theta = 0
    Gain = 0

    # Joint diagonalization proper

    if verbose:
        print("jade -> Contrast optimization by joint diagonalization")

    while encore:
        encore = False
        if verbose:
            print("jade -> Sweep #%3d" % sweep)
        sweep = sweep + 1
        upds = 0
        Vkeep = V

        for p in range(m - 1):
            for q in range(p + 1, m):

                Ip = arange(p, m * nbcm, m)
                Iq = arange(q, m * nbcm, m)

                # computation of Givens angle
                g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                gg = dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff))
                Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0

                # Givens update
                if abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = cos(theta)
                    s = sin(theta)
                    G = matrix([[c, -s], [s, c]])
                    pair = array([p, q])
                    V[:, pair] = V[:, pair] * G
                    CM[pair, :] = G.T * CM[pair, :]
                    CM[:, concatenate([Ip, Iq])] = \
                        append(c * CM[:, Ip] + s * CM[:, Iq], -s * CM[:, Ip] + c * CM[:, Iq], \
                               axis=1)
                    On = On + Gain
                    Off = Off - Gain

        if verbose:
            print("completed in %d rotations" % upds)
        updates = updates + upds
    if verbose:
        print("jade -> Total of %d Givens rotations" % updates)

    # A separating matrix
    # ===================

    B = V.T * B

    # Permute the rows of the separating matrix B to get the most energetic components first.
    # Here the **signals** are normalized to unit variance.  Therefore, the sort is
    # according to the norm of the columns of A = pinv(B)

    if verbose:
        print("jade -> Sorting the components")

    A = np.linalg.pinv(B)
    keys = array(argsort(multiply(A, A).sum(axis=0)[0]))[0]
    B = B[keys, :]
    B = B[::-1, :]  # % Is this smart ?

    if verbose:
        print("jade -> Fixing the signs")
    b = B[:, 0]
    signs = array(sign(sign(b) + 0.1).T)[0]  # just a trick to deal with sign=0
    B = diag(signs) * B

    return B.astype(origtype)