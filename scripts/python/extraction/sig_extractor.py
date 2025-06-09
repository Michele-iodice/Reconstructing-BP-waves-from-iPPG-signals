from importlib import import_module
import cv2
from my_pyVHR.extraction.sig_extraction_methods import SignalProcessingParams
from my_pyVHR.extraction.skin_extraction_methods import SkinExtractionFaceParsing, SkinProcessingParams
from my_pyVHR.extraction.sig_processing import SignalProcessing
from my_pyVHR.extraction.utils import get_fps, sig_windowing
from BP.filters import *
from PPG.filters import *
from scipy import interpolate, sparse
from scipy.sparse.linalg import spsolve
from PPG.PPG import RGB_sig_to_rPPG
from extraction.signal_to_cwt import signal_to_cwt


def extract_Sig(videoFileName, conf, verb=True, method='cpu_POS'):
    """the method extract and pre-processing a rgb signal from a video or path"""

    winsize= np.float32(conf.sigdict['winsize'])
    stride = np.float32(conf.sigdict['stride'])
    if get_winsize(videoFileName)<30:
        winsize=get_winsize(videoFileName)

    roi_method=conf.sigdict['method']
    roi_approach=conf.sigdict['approach']

    sig_processing = SignalProcessing()
    target_device=conf.sigdict['target_device']
    # 1. apply the Face parsing extractor method
    sig_processing.set_skin_extractor(SkinExtractionFaceParsing(target_device))
    # 2. set sig-processing and skin-processing params
    SignalProcessingParams.RGB_LOW_TH = np.int32(conf.sigdict['RGB_LOW_TH'])
    SignalProcessingParams.RGB_HIGH_TH = np.int32(conf.sigdict['RGB_HIGH_TH'])
    SkinProcessingParams.RGB_LOW_TH = np.int32(conf.sigdict['Skin_LOW_TH'])
    SkinProcessingParams.RGB_HIGH_TH = np.int32(conf.sigdict['Skin_HIGH_TH'])


    fps = get_fps(videoFileName)
    sig_processing.set_total_frames(30*fps) # set to 0 to process the whole video
    # 3. ROI selection
    if verb:
        print('\nProcessing Video ' + videoFileName)
        print('\nRoi processing...')

    sig = []
    # SIG extraction with holistic approach
    sig_extract = sig_processing.extract_holistic(videoFileName, scale_percent=30, frame_interval=2)
    sig_extract = np.transpose(sig_extract, (1, 2, 0))
    sig.append(sig_extract)
    print(sig[0].shape)
    print(sig[0])
    if len(sig) <= 0:
        print('\nError:No signal extracted.')
        return None

    if verb:
        print(' - Extraction approach: ' + roi_approach)
        print(' - Extraction method: ' + roi_method)

    # 4. PRE FILTERING
    if verb:
        print('\nPre filtering...')
    filtered_sig = sig

    minHz = np.float32(conf.sigdict['minHz'])  # min heart frequency in Hz
    maxHz = np.float32(conf.sigdict['maxHz'])  # max heart frequency in Hz
    module = import_module('BP.filters')
    method_to_call = getattr(module, 'BPfilter')
    filtered_bp_sig = apply_filter(filtered_sig,
                                   method_to_call,
                                   fps=fps,
                                   params={'minHz': minHz,
                                           'maxHz': maxHz,
                                           'fps': 'adaptive',
                                           'order': 2})

    print(filtered_bp_sig[0].shape)
    print(filtered_bp_sig[0])

    if verb:
        print(f' - Pre-filter applied: {method_to_call.__name__}')

    filter_range=[-1, 1]  # constant range of normalization
    method_to_call = getattr(module, 'zscorerange')
    filtered_normal_sig = apply_filter(filtered_bp_sig,
                                       method_to_call,
                                       params={'minR': filter_range[0],
                                               'maxR': filter_range[1]})

    print(filtered_normal_sig[0].shape)
    print(filtered_normal_sig[0])

    if verb:
        print(f' - Pre-filter applied: {method_to_call.__name__}')

    # 5. rPPG extraction
    if verb:
        print("\nPPG extraction...")
        print(" - Extraction method: " + method)

    module = import_module('PPG.methods')
    method_to_call = getattr(module, method)

    if 'POS' in method:
        pars = {'fps': 'adaptive'}
    elif 'PCA' in method or 'ICA' in method:
        pars = {'component': 'all_comp'}
    else:
        pars = {}

    r_ppgs_win = RGB_sig_to_rPPG(filtered_normal_sig,
                                 fps,
                                 method=method_to_call,
                                 params=pars)

    print(r_ppgs_win[0].shape)

    # 6. POST FILTERING
    module = import_module('PPG.filters')
    method_to_call = getattr(module, 'interpolation')
    fps = np.int32(conf.uNetdict['frameRate'])
    r_ppgs_interp = apply_ppg_filter(r_ppgs_win,
                                     method_to_call,
                                     fps=fps,
                                     params={'fps': fps})
    if verb:
        print(f' - Post-filter applied: {method_to_call.__name__}')

    print(r_ppgs_interp[0].shape)

    method_to_call = getattr(module, 'detrend')
    r_ppgs_detrend = apply_ppg_filter(r_ppgs_interp,
                                      method_to_call,
                                      fps=np.int32(conf.uNetdict['frameRate']))
    print(r_ppgs_detrend[0].shape)
    if verb:
        print(f' - Post-filter applied: {method_to_call.__name__}')

    # 7. Sig Windowing
    windowed_sig, timesES = sig_windowing(r_ppgs_detrend[0], winsize, stride, fps)
    if len(windowed_sig) <= 0:
        print('\nError:No windowed signal.')
        return None

    if verb:
        print(f' - Number of windows: {len(windowed_sig)}')
        print(' - Win size: (#Estimators, #Channels, #Frames) = ', windowed_sig[0].shape)


    return windowed_sig, timesES


def get_winsize(videoFileName):
    """
    This method returns the duration of a video file name or path in sec.
    """
    cap = cv2.VideoCapture(videoFileName)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    winsize = frame_count / fps
    cap.release()

    return winsize

def post_filtering(signal, detrend, fps, verbose=False):
    """
    This method post-filters the signal using a pre-filtering method.
    :param signal: full iPPG or BP signal (sampling frequency=fps)
    :param detrend: 0 for no detrending (BP) 1 for detrending (iPPG)
    :param fps: sampling frequency of the signal.
    :return: signal after post-filtering.
    """

    if verbose:
        print("Post-filtering...")

    time = np.linspace(0, (len(signal) - 1) / fps, int(len(signal) * (100 / fps)))
    x = np.linspace(0, (len(signal) - 1) / fps, len(signal))
    if len(x) != len(signal):
        min_len = min(len(x), len(signal))
        x = x[:min_len]
        signal = signal[:min_len]

    interp_func = interpolate.interp1d(x, signal, kind='linear')
    signal = interp_func(time)

    # DETRENDING (Tarvainen et al., 2002)
    if detrend:
        if verbose:
            print("-post-filter applied: Detrending")

        lambda_ = 470  # Smoothing parameter
        T = len(signal)

        # Identity matrix (sparse)
        I = sparse.eye(T)

        # Second-order difference matrix D (sparse)
        data = [np.ones(T), -2 * np.ones(T), np.ones(T)]
        offsets = [0, 1, 2]
        D2 = sparse.diags(data, offsets, shape=(T - 2, T))

        # Solve (I + λ^2 * D^T D) * z = signal
        H = I + lambda_ ** 2 * D2.T @ D2
        z = spsolve(H, signal)
        signal = signal - z

    return signal