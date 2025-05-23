import re

import h5py
import numpy as np
import pandas as pd

from extraction.signal_to_cwt import signal_to_cwt
from my_pyVHR.datasets.dataset import datasetFactory
from dataset.bp4d import BP4D
from extraction.sig_extractor import extract_Sig, post_filtering


def getSubjectId(videoFilename):
    """
    retrieve the id of the subject in the videoFilename
    :param videoFilename: the path of the video.
    :return: Id of the Subject
    """

    # if found a match, return the respectively string
    match = re.search(r"\\([FM]\d{3})\\", videoFilename)
    if match:
        s = match.group(1)
        return s
    else:
        return "Unknown"


def getSex(subjectId):
    """
    retrieve the sex of the subject considered 0 if female, 1 if male
    :param videoFilename: the path of the video
    :return: sex of the subject 0 if female, 1 if male
    """
    # Definisci il pattern
    pattern = subjectId[0]

    # Verifica se c'è una corrispondenza e stampa il risultato
    if pattern == "F":
        return 0  # sex equal female
    elif pattern == "M":
        return 1  # sex equal male
    else:
        return -1  # not found


def save_subject_data(f, group_id, subject_id, sex, sig_ippg, sig_bp, ippg, bp, ippg_cwt, bp_cwt):
    grp = f.create_group(str(group_id))
    grp.attrs["sex"] = int(sex)
    grp.attrs["subject_id"] = str(subject_id)

    grp.create_dataset("original_signal", data=sig_ippg.astype(np.float32), compression="gzip")
    grp.create_dataset("GT_bp", data=sig_bp.astype(np.float32), compression="gzip")
    grp.create_dataset("ippg", data=ippg.astype(np.float32), compression="gzip")
    grp.create_dataset("bp", data=bp.astype(np.float32), compression="gzip")
    grp.create_dataset("ippg_cwt", data=ippg_cwt.astype(np.float32), compression="gzip")
    grp.create_dataset("bp_cwt", data=bp_cwt.astype(np.float32), compression="gzip")


def extract_feature_on_dataset(conf,dataset_path):
    """
    this function extract the data feature from the dataset and load it into .h5 file.
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

    with h5py.File(dataset_path, "a") as f:
        for idx in range(0, dataset_len):
            fname = dataset.getSigFilename(idx)
            sigGT = dataset.readSigfile(fname)
            bpGT = sigGT.getSig()
            sig_bp = post_filtering(bpGT[0], detrend=1, fps=np.int32(conf.uNetdict['frameRate']))
            cwt_bp, sig_bp_windows = signal_to_cwt(sig_bp, range_freq=[0.6, 4.5], num_scales=256, overlap=50, norm=0, recover=1)
            videoFileName = dataset.getVideoFilename(idx)
            print('videoFileName: ', videoFileName)
            subjectId = getSubjectId(videoFileName)
            sex = getSex(subjectId)
            sigEX = extract_Sig(videoFileName, conf)
            if sigEX is None:
                print('\nError:No signal extracted.')
                print('\nDiscarded video.')
                continue

            green_signal = np.concatenate([segment[0, 1, :] for segment in sigEX])
            sig_ippg = post_filtering(green_signal, detrend=1, fps=np.int32(conf.uNetdict['frameRate']),verbose=True)
            cwt_ippg, sig_ippg_windows = signal_to_cwt(sig_ippg,range_freq=[0.6, 4.5],num_scales=256, overlap=50, norm=1, recover=0, verbose=True)

            for i in range(min(len(cwt_ippg), len(cwt_bp))):
                group_id = f"{subjectId}_{idx}_{i}"
                print("cwt_ippg: ", cwt_ippg[i].shape)
                print("cwt_bp: ", cwt_bp[i].shape)
                save_subject_data(f, group_id, subjectId, sex, sig_ippg, sig_bp, sig_ippg_windows[i], sig_bp_windows[i], cwt_ippg[i], cwt_bp[i])


def extract_feature_on_video(video, bp, dataset_path, conf):
    """
        this function extract the data feature from a video.
        :param: video: the path of the video to compute
        :return: the dataframe 'data' with columns: CWT, sex, BP ground truth, and subject_id
        """

    with h5py.File(dataset_path, "a") as f:
        fname = video
        sigGT = BP4D.readSigfile(BP4D, bp)
        bpGT = sigGT.getSig()
        sig_bp = post_filtering(bpGT, detrend=1, fps=np.int32(conf.uNetdict['frameRate']))
        cwt_bp = signal_to_cwt(sig_bp, range_freq=[0.1, 10], num_scales=256, overlap=50, norm=0, recover=1)
        subjectId = getSubjectId(fname)
        sex = getSex(subjectId)
        sigEX = extract_Sig(fname, conf)
        green_signal = np.concatenate([segment[0, 1, :] for segment in sigEX])
        sig_ippg = post_filtering(green_signal, detrend=1, fps=np.int32(conf.uNetdict['frameRate']), verbose=True)
        cwt_ippg = signal_to_cwt(sig_ippg,range_freq=[0.6, 4.5],num_scales=256, overlap=50, norm=1, recover=0, verbose=True)

        for i in range(min(len(cwt_ippg), len(cwt_bp))):
            group_id = f"{subjectId}_{i}"
            save_subject_data(f, group_id, subjectId, sex, sig_ippg, sig_bp, cwt_ippg, cwt_bp)


    return True

# Example usage
# data = extract_feature_on_dataset(config)
# data.to_csv(data_path, index=False)

# signal_to_cwt(green_signal, overlap=50, norm=1, recover=0)