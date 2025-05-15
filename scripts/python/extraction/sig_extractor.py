from importlib import import_module
import cv2
from my_pyVHR.extraction.sig_extraction_methods import SignalProcessingParams
from my_pyVHR.extraction.skin_extraction_methods import SkinExtractionFaceParsing, SkinProcessingParams
from my_pyVHR.extraction.sig_processing import SignalProcessing
from my_pyVHR.extraction.utils import get_fps, sig_windowing
from BP.filters import *


def extract_Sig(videoFileName, conf):
    """the method extract and pre processing an rgb signal from a video or path"""

    winsize= np.int32(conf.sigdict['winsize'])
    if get_winsize(videoFileName)<30:
        winsize=get_winsize(videoFileName)

    roi_method=conf.sigdict['method']
    roi_approach=conf.sigdict['approach']

    sig_processing = SignalProcessing()
    target_device=conf.sigdict['target_device']
    # apply the Face parsing extractor method
    sig_processing.set_skin_extractor(SkinExtractionFaceParsing(target_device))
    # set sig-processing and skin-processing params
    SignalProcessingParams.RGB_LOW_TH = np.int32(conf.sigdict['RGB_LOW_TH'])
    SignalProcessingParams.RGB_HIGH_TH = np.int32(conf.sigdict['RGB_HIGH_TH'])
    SkinProcessingParams.RGB_LOW_TH = np.int32(conf.sigdict['Skin_LOW_TH'])
    SkinProcessingParams.RGB_HIGH_TH = np.int32(conf.sigdict['Skin_HIGH_TH'])

    print('\nProcessing Video ' + videoFileName)
    fps = get_fps(videoFileName)
    sig_processing.set_total_frames(30*fps) # set to 0 to process the whole video
    # 3. ROI selection
    print('\nRoi processing...')
    sig = []
    # SIG extraction with holistic approach
    sig = sig_processing.extract_holistic(videoFileName, scale_percent=30, frame_interval=2)
    print(' - Extraction approach: ' + roi_approach)
    print(' - Extraction method: ' + roi_method)

    # 4. sig windowing
    windowed_sig, timesES = sig_windowing(sig, winsize, 1, fps)
    print(f' - Number of windows: {len(windowed_sig)}')
    print(' - Win size: (#ROI, #landmarks, #frames) = ', windowed_sig[0].shape)

    # 5. PRE FILTERING
    print('\nPre filtering...')
    filtered_windowed_sig = windowed_sig

    minHz = np.float32(conf.sigdict['minHz'])  # min heart frequency in Hz
    maxHz = np.float32(conf.sigdict['maxHz'])  # max heart frequency in Hz
    module = import_module('BP.filters')
    method_to_call = getattr(module, 'BPfilter')
    filtered_bp_sig = apply_filter(filtered_windowed_sig,
                                   method_to_call,
                                   fps=fps,
                                   params={'minHz': minHz,
                                           'maxHz': maxHz,
                                           'fps': 'adaptive',
                                           'order': 2})

    print(f' - Pre-filter applied: {method_to_call.__name__}')

    filter_range=[-1, 1]  # constant range of normalization
    method_to_call = getattr(module, 'zscorerange')
    filtered_normal_sig = apply_filter(filtered_bp_sig,
                                       method_to_call,
                                       params={'minR': filter_range[0],
                                               'maxR': filter_range[1]})

    print(f' - Pre-filter applied: {method_to_call.__name__}')

    return filtered_normal_sig


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
