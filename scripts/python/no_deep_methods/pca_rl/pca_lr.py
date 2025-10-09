import joblib
import numpy as np
import scipy.signal as sps
import cv2
from config import Configuration
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from my_pyVHR.datasets.dataset import datasetFactory
from my_pyVHR.extraction.sig_extraction_methods import SignalProcessingParams
from my_pyVHR.extraction.skin_extraction_methods import SkinExtractionFaceParsing, SkinProcessingParams
from my_pyVHR.extraction.sig_processing import SignalProcessing
from my_pyVHR.extraction.utils import get_fps
import pandas as pd
import ast

# -------------------------------
# Funzioni di estrazione e processing
# -------------------------------

def get_winsize(videoFileName):
    cap = cv2.VideoCapture(videoFileName)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    winsize = frame_count / fps
    cap.release()
    return winsize

def extract_Sig(videoFileName, conf, verb=True, method='cpu_POS'):
    roi_method = conf.sigdict['method']
    roi_approach = conf.sigdict['approach']

    sig_processing = SignalProcessing()
    target_device = conf.sigdict['target_device']
    sig_processing.set_skin_extractor(SkinExtractionFaceParsing(target_device))

    SignalProcessingParams.RGB_LOW_TH = np.int32(conf.sigdict['RGB_LOW_TH'])
    SignalProcessingParams.RGB_HIGH_TH = np.int32(conf.sigdict['RGB_HIGH_TH'])
    SkinProcessingParams.RGB_LOW_TH = np.int32(conf.sigdict['Skin_LOW_TH'])
    SkinProcessingParams.RGB_HIGH_TH = np.int32(conf.sigdict['Skin_HIGH_TH'])

    fps = get_fps(videoFileName)
    sig_processing.set_total_frames(30 * fps)

    if verb:
        print('\nProcessing Video ' + videoFileName)
        print('\nRoi processing...')

    sig = []
    sig_extract = sig_processing.extract_holistic(videoFileName, scale_percent=30, frame_interval=1)
    sig_extract = np.transpose(sig_extract, (1, 2, 0))
    sig.append(sig_extract)

    if len(sig) <= 0:
        print('\nError: No signal extracted.')
        return None

    if verb:
        print(' - Extraction approach: ' + roi_approach)
        print(' - Extraction method: ' + roi_method)

    return np.array(sig)

def extract_rppg(signal, n_components=10):
    red = np.nan_to_num(signal[0])
    red_frames = red.T
    n_comp = min(n_components, red_frames.shape[1], red.shape[1])
    if n_comp < 1:
        raise ValueError("Segnale troppo corto o vuoto per PCA")
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(red_frames)
    recon = pca.inverse_transform(transformed)
    rppg = (recon - red_frames).mean(axis=1)
    rppg = (rppg - np.mean(rppg)) / (np.std(rppg) + 1e-8)
    return rppg

def bandpass_filter(signal, fs=30, lowcut=0.5, highcut=5.0):
    if len(signal) < 21:
        print("Segnale troppo corto per bandpass, restituisco il segnale originale")
        return signal
    b, a = sps.butter(3, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    return sps.filtfilt(b, a, signal)

def select_best_segment(rppg, fs=30, discard_sec=5, segment_sec=10):
    discard_fr = int(discard_sec * fs)
    segment_fr = int(segment_sec * fs)
    rppg_clean = rppg[discard_fr : -discard_fr] if len(rppg) > 2*discard_fr else rppg
    window_len = min(segment_fr, len(rppg_clean))
    max_var = 0
    best_segment = rppg_clean[:window_len]
    best_start = 0
    for i in range(len(rppg_clean)-window_len):
        seg = rppg_clean[i:i+window_len]
        if np.var(seg) > max_var:
            max_var = np.var(seg)
            best_segment = seg
            best_start = i
    return best_segment, best_start + discard_fr

# -----------------------
# Feature extraction + SBP/DBP
# -----------------------
def extract_features(rppg, bp_segment, fs=30):
    peaks, _ = sps.find_peaks(rppg, distance=fs * 0.5)
    if len(peaks) < 2:
        intervals = [0]
        hr = 0
        amp = 0
        slope = 0
        time_rise = 0
        time_fall = 0
        area = 0
    else:
        intervals = np.diff(peaks) / fs
        hr = 60 / np.mean(intervals)
        amp = np.mean(rppg[peaks]) - np.min(rppg)
        slope = np.max(np.diff(rppg)) if len(rppg) > 1 else 0
        time_rise = np.mean([np.argmax(rppg[peaks[i]:peaks[i + 1]]) / fs
                             for i in range(len(peaks) - 1)]) if len(peaks) > 1 else 0
        time_fall = np.mean([np.argmin(rppg[peaks[i]:peaks[i + 1]]) / fs
                             for i in range(len(peaks) - 1)]) if len(peaks) > 1 else 0
        area = np.mean([np.sum(rppg[peaks[i]:peaks[i + 1]]) for i in range(len(peaks) - 1)]) if len(peaks) > 1 else 0

    if len(bp_segment) < 2:
        SBP = np.max(bp_segment) if len(bp_segment) > 0 else 0
        DBP = np.min(bp_segment) if len(bp_segment) > 0 else 0
    else:
        bp_peaks, _ = sps.find_peaks(bp_segment, distance=fs * 0.5)
        sbp_values = bp_segment[bp_peaks] if len(bp_peaks) > 0 else [np.max(bp_segment)]
        dbp_values = [np.min(bp_segment[bp_peaks[i]:bp_peaks[i + 1]]) for i in range(len(bp_peaks) - 1)] \
            if len(bp_peaks) > 1 else [np.min(bp_segment)]
        SBP = np.mean(sbp_values)
        DBP = np.mean(dbp_values)

    features = [np.mean(intervals), hr, amp, slope, time_rise, time_fall, area]
    return features, SBP, DBP

# -----------------------
# Metrics
# -----------------------
def evaluate_bp(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse

def check_aami(y_true, y_pred):
    mae, rmse = evaluate_bp(y_true, y_pred)
    errors = y_true - y_pred
    me = np.mean(errors)
    sde = np.std(errors, ddof=1)
    compliant = me <= 5 and sde <= 8
    return compliant, me, sde, mae, rmse

def check_bhs(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    p5 = np.mean(diff < 5) * 100
    p10 = np.mean(diff < 10) * 100
    p15 = np.mean(diff < 15) * 100
    if p5 >= 60 and p10 >= 85 and p15 >= 95:
        grade = "A"
    elif p5 >= 50 and p10 >= 75 and p15 >= 90:
        grade = "B"
    elif p5 >= 40 and p10 >= 65 and p15 >= 85:
        grade = "C"
    else:
        grade = "D"
    return grade, p5, p10, p15

def evaluate_metrics(y_true, y_pred):
    aami_compliant, aami_me, aami_sde, mae, rmse = check_aami(y_true, y_pred)
    grade, p5, p10, p15 = check_bhs(y_true, y_pred)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'AAMI': "Pass" if aami_compliant else "Fail",
        'ME': aami_me,
        'SDE': aami_sde,
        'Perc_<5': p5,
        'Perc_<10': p10,
        'Perc_<15': p15,
        'BHS': grade
    }

# -----------------------
# Main Pipeline
# -----------------------
def execute(conf, data_path):
    X_feats, y_sbp, y_dbp = [], [], []

    # --- Load dataset CSV ---
    df = pd.read_csv(data_path)
    df['rppg'] = df['rppg'].apply(ast.literal_eval)
    df['bp_segment'] = df['bp_segment'].apply(ast.literal_eval)
    rppg_list = [np.array(sig) for sig in df['rppg'].tolist()]
    bp_list = [np.array(bp) for bp in df['bp_segment'].tolist()]

    fs = 25
    for rppg, bp in zip(rppg_list, bp_list):
        feats, SBP, DBP = extract_features(rppg, bp, fs)
        X_feats.append(feats)
        y_sbp.append(SBP)
        y_dbp.append(DBP)

    X_feats = np.array(X_feats)
    Y = np.column_stack([y_sbp, y_dbp])  # Multi-output target

    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X_feats)

    cv_folds = min(9, len(X_feats))

    # --- Modello unico multi-output ---
    lr = LinearRegression()
    lr.fit(X_poly, Y)
    y_pred = cross_val_predict(lr, X_poly, Y, cv=cv_folds)

    # --- Metriche separate ---
    metrics_sbp = evaluate_metrics(Y[:,0], y_pred[:,0])
    metrics_dbp = evaluate_metrics(Y[:,1], y_pred[:,1])

    results = pd.DataFrame([
        {'Target': 'SP', **metrics_sbp},
        {'Target': 'DP', **metrics_dbp}
    ])
    results.to_csv('pcalr_results.csv', index=False)
    print(results)

    # --- Salvataggio modelli ---
    joblib.dump(poly, "pca_lr_poly.pkl")
    joblib.dump(lr, "pca_lr_model.pkl")


if __name__ == "__main__":
    data_path = "pca_lr_dataset.csv"
    config = Configuration('C:/Users/Utente/Documents/GitHub/Reconstructing-BP-waves-from-iPPG-signals/scripts/python/config.cfg')
    execute(config, data_path)
