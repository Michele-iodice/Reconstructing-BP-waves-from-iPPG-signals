import numpy as np
import pandas as pd
from my_pyVHR.datasets.dataset import datasetFactory
from dataset.bp4d import BP4D
from extraction.signal_to_cwt import signal_to_cwt
from extraction.sig_extractor import extract_Sig


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


def extract_feature_on_dataset(conf):
    """
    this function extract the data feature from the dataset.
    :return: the dataframe 'data' with columns: CWT, sex, BP ground truth, and subject_id
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
    df = pd.DataFrame(columns=['CWT', 'sex', 'BP', 'subject_id'])
    for idx in range(0, dataset_len):

            fname = dataset.getSigFilename(idx)
            sigGT = dataset.readSigfile(fname)
            bpGT = sigGT.getSigValue()
            cwt_BP = signal_to_cwt(bpGT, overlap=50, norm=0, detrend=0, recover=1,
                                   fps=np.int32(conf.uNetdict['frameRate']))
            videoFileName = dataset.getVideoFilename(idx)
            subjectId = getSubjectId(videoFileName)
            sex = getSex(videoFileName)
            sigEX = extract_Sig(videoFileName, conf)
            cwt_ippg = signal_to_cwt(sigEX, overlap=50, norm=1, detrend=1, recover=0,
                                     fps=np.int32(conf.uNetdict['frameRate']))
            newLine = pd.DataFrame({'CWT': cwt_ippg, 'sex': sex, 'BP': cwt_BP, 'subject_id': subjectId}, index=[0])
            df = pd.concat([df, newLine], ignore_index=True)

    return df


def extract_feature_on_video(video, bp, conf):
    """
        this function extract the data feature from a video.
        :param: video: the path of the video to compute
        :return: the dataframe 'data' with columns: CWT, sex, BP ground truth, and subject_id
        """
    df = pd.DataFrame(columns=['CWT', 'CWT_BP', 'sex', 'BP', 'original', 'subject_id'])

    fname = video
    sigGT = BP4D.readSigfile(BP4D, bp)
    bpGT = sigGT.getSigValue()
    cwt_BP, scales, time= signal_to_cwt(bpGT, overlap=50, norm=0, detrend=0,
                                        recover=1, fps=np.int32(conf.uNetdict['frameRate']))
    subjectId = getSubjectId(fname)
    sex = getSex(fname)
    sigEX = extract_Sig(fname, conf)
    cwt_ippg, scales2, time2 = signal_to_cwt(sigEX, overlap=50, norm=1, detrend=1,
                                             recover=0, fps=np.int32(conf.uNetdict['frameRate']))
    newLine = pd.DataFrame({'CWT': cwt_ippg, 'CWT_BP': cwt_BP, 'sex': sex, 'BP': bpGT,
                            'original': sigEX, 'subject_id': subjectId}, index=[0])
    df = pd.concat([df, newLine], ignore_index=True)

    return df

# Example usage
# data = extract_feature_on_dataset(config)
# data.to_csv(data_path, index=False)