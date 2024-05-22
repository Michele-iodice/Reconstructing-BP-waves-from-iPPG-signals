import numpy as np
import pandas as pd
from my_pyVHR.datasets.dataset import datasetFactory
from scipy.signal import find_peaks
from dataset.bp4d import BP4D
from extraction.ptt_detector import rgb_sig_to_ptt
from extraction.sig_extractor import extract_Sig
import matplotlib.pyplot as plt


def getSubjectId(videoFilename):
    """
    retrieve the id of the subject in the videoFilename
    :param videoFilename: the path of the video
    :return: Id of the Subject
    """
    pattern = videoFilename[86:90]

    # if found a match, return the respectively string
    if len(pattern) > 0:
        return pattern
    else:
        return None


def getSex(videoFilename):
    """
    retrieve the sex of the subject considered 0 if female, 1 if male
    :param videoFilename: the path of the video
    :return: sex of the subject 0 if female, 1 if male
    """
    # Definisci il pattern
    pattern = videoFilename[86:87]

    # Verifica se c'è una corrispondenza e stampa il risultato
    if pattern == "F":
        return 0  # sex equal female
    elif pattern == "M":
        return 1  # sex equal male
    else:
        return -1  # not found


def calculate_parameters(bp_signal, sampling_rate):
    """
    compute some values in the bp signal
    :param bp_signal: the signal to compute
    :param sampling_rate: the frequency of the signal registration
    :return: Tc = average cycle width ,
             Ts = time from start of cycle to systolic peak,
             Td = time from systolic peak to cycle end,
             AUCc = areas under the curve of average cycle width,
             AUCs = areas under the curve of time from start of cycle to systolic peak,
             AUCd = areas under the curve of time from systolic peak to cycle end.
    """
    # Find peaks
    peaks, _ = find_peaks(bp_signal)
    # Find valleys by inverting the signal and finding peaks
    inverted_signal = -bp_signal
    valleys, _ = find_peaks(inverted_signal)

    # Check the shapes or lengths of the arrays
    valleys_shape = np.shape(valleys)
    peaks_shape = np.shape(peaks)
    # Adjust the arrays to make them compatible if needed
    if valleys_shape[0] != peaks_shape[0]:
        # Perform necessary operations to make them compatible
        if valleys_shape[0] > peaks_shape[0]:
            valleys = valleys[:peaks_shape[0]]
        else:
            peaks = peaks[:valleys_shape[0]]

    # Calculate Tc, Ts, Td
    Tc = np.mean(np.diff(peaks) / sampling_rate)
    Ts = np.mean((valleys - peaks) / sampling_rate)
    Td = np.mean((peaks - valleys) / sampling_rate)

    # Calculate AUCc, AUCs, AUCd
    AUCc = np.sum(bp_signal[peaks] / sampling_rate)
    AUCs = np.sum(bp_signal[peaks[:-1]] / sampling_rate)
    AUCd = np.sum(bp_signal[valleys] / sampling_rate)

    return Tc, Ts, Td, AUCc, AUCs, AUCd


def extract_feature_on_dataset(conf):
    """
    this function extract the data feature from the dataset.
    :return: the dataframe 'data' with columns: PTT, Tc, Ts, Td, AUCc, AUCs, AUCd, sex, BP ground truth, and subject_id
    """
    datasetName = conf.datasetdict['dataset']
    path = conf.datasetdict['path']
    videodataDIR = conf.datasetdict['videodataDIR']
    BVPdataDIR = conf.datasetdict['BVPdataDIR']
    dataset = datasetFactory(datasetName,
                             videodataDIR,
                             BVPdataDIR,
                             path)
    dataset_len = dataset.len_dataset()
    print('dataset len: ', dataset_len)
    df = pd.DataFrame(columns=['PTT', 'Tc', 'Ts', 'Td', 'AUCc', 'AUCs', 'AUCd', 'sex', 'BP', 'subject_id'])
    ptt_for_subjects = []
    for idx in range(0, dataset_len):

            fname = dataset.getSigFilename(idx)
            sigGT = dataset.readSigfile(fname)
            bpGTsignal = sigGT.getSig()
            bpGT = sigGT.getSigValue()
            videoFileName = dataset.getVideoFilename(idx)
            subjectId = getSubjectId(videoFileName)
            sex = getSex(videoFileName)
            Tc, Ts, Td, AUCc, AUCs, AUCd = calculate_parameters(bpGTsignal[0], sigGT.getSigFps())
            sigEX = extract_Sig(videoFileName, conf)
            ptt, ptt_for_subject = rgb_sig_to_ptt(sigEX, np.int32(conf.sigdict['frameRate']), conf)
            ptt_for_subjects.append(ptt_for_subject)
            newLine = pd.DataFrame({'PTT': ptt, 'Tc': Tc, 'Ts': Ts, 'Td': Td,
                                    'AUCc': AUCc, 'AUCs': AUCs, 'AUCd': AUCd,
                                    'sex': sex, 'BP': bpGT, 'subject_id': subjectId}, index=[idx])
            df = pd.concat([df, newLine], ignore_index=True)

    # display the PTT per subject
    plt.boxplot(np.asarray(ptt_for_subjects), vert=True, patch_artist=True)
    # Adding labels and title
    subjects = np.asarray(df.filter(like='subject_id'))
    plt.xticks(list(range(1, len(subjects) + 1)), subjects)
    plt.xlabel('Subject')
    plt.ylabel('PTT [ms]')
    plt.title('PTT boxplot per subject')
    plt.legend()
    # Display the plot
    plt.show()

    return df


def extract_feature_on_video(video, bp, conf):
    """
        this function extract the data feature from a video.
        :param: video: the path of the video to compute
        :return: the dataframe 'data' with columns: PTT, Tc, Ts, Td, AUCc, AUCs, AUCd, sex, BP ground truth, and subject_id
        """
    df = pd.DataFrame(columns=['PTT', 'Tc', 'Ts', 'Td', 'AUCc', 'AUCs', 'AUCd', 'sex', 'BP', 'subject_id'])

    fname = video
    sigGT = BP4D.readSigfile(BP4D, bp)
    bpGTsignal = sigGT.getSig()
    bpGT = sigGT.getSigValue()
    subjectId = getSubjectId(fname)
    sex = getSex(fname)
    Tc, Ts, Td, AUCc, AUCs, AUCd = calculate_parameters(bpGTsignal[0], sigGT.getSigFps())
    sigEX = extract_Sig(fname, conf)
    ptt = rgb_sig_to_ptt(sigEX, np.int32(conf.sigdict['frameRate']), conf)
    newLine = pd.DataFrame({'PTT': ptt, 'Tc': Tc, 'Ts': Ts, 'Td': Td,
                            'AUCc': AUCc, 'AUCs': AUCs, 'AUCd': AUCd,
                            'sex': sex, 'BP': bpGT, 'subject_id': subjectId}, index=[0])
    df = pd.concat([df, newLine], ignore_index=True)

    return df
