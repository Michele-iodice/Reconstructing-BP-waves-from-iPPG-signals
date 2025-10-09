import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import find_peaks
from collections import defaultdict
import h5py
from statistics import mode
from scipy.signal import butter, filtfilt

# -----------------------------------
# SUPPORT VECTOR REGRESSOR METHOD
# -----------------------------------

def bandpass_filter(signal, fs=30):
    cutoff = 5  # Hz
    b, a = butter(2, cutoff / (fs / 2), btype='low')
    ppg_filtered = filtfilt(b, a, signal)
    return ppg_filtered

def extract_features(ppg_segment, bp_segment):
    features = {}
    features['ppg_max'] = np.max(ppg_segment)
    features['ppg_min'] = np.min(ppg_segment)
    features['ppg_range'] = features['ppg_max'] - features['ppg_min']

    rise_idx = np.argmax(ppg_segment)
    features['ppg_rise_time'] = rise_idx / len(ppg_segment)
    features['ppg_fall_time'] = (len(ppg_segment) - rise_idx) / len(ppg_segment)

    d1 = np.diff(ppg_segment)
    features['d1_max'] = np.max(d1)
    features['d1_min'] = np.min(d1)
    d2 = np.diff(d1)
    features['d2_max'] = np.max(d2)
    features['d2_min'] = np.min(d2)

    # Label
    features['SP'] = np.max(bp_segment)
    features['DP'] = np.min(bp_segment)

    return features


def segment_beats(ppg, bp):
    peaks, _ = find_peaks(ppg, distance=50)
    segments = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        segments.append((ppg[start:end], bp[start:end]))
    return segments


def create_feature_dataset(ppgs,bps):
    all_features = []
    end=min(len(ppgs),len(bps))
    for idx in range(0,end):
        ppg = ppgs[idx]
        bp = bps[idx]
        segments = segment_beats(ppg, bp)
        for seg_ppg, seg_bp in segments:
            feat = extract_features(seg_ppg, seg_bp)
            all_features.append(feat)
    feature_df = pd.DataFrame(all_features)
    X = feature_df.drop(columns=['SP', 'DP'])
    y_sp = feature_df['SP'].values
    y_dp = feature_df['DP'].values
    return X, y_sp, y_dp



def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    errors = y_true - y_pred
    me = np.mean(errors)
    sde = np.std(errors, ddof=1)
    abs_errors = np.abs(y_true - y_pred)

    # AAMI
    aami_mae = me < 5
    aami_sde = sde < 8
    aami_compliance = aami_mae and aami_sde

    # BHS grading
    p5 = np.mean(abs_errors < 5) * 100
    p10 = np.mean(abs_errors < 10) * 100
    p15 = np.mean(abs_errors < 15) * 100

    if p5 >= 60 and p10 >= 85 and p15 >= 95:
        grade = "A"
    elif p5 >= 50 and p10 >= 75 and p15 >= 90:
        grade = "B"
    elif p5 >= 40 and p10 >= 65 and p15 >= 85:
        grade = "C"
    else:
        grade = "D"

    return {
        "MAE": mae,
        "RMSE": rmse,
        "AAMI": "Pass" if aami_compliance else "Fail",
        "ME": me,
        "SDE": sde,
        "BHS": grade,
        "Perc_<5": p5,
        "Perc_<10": p10,
        "Perc_<15": p15
    }


# Funzione per training SVR con K-Fold CV
def train_svr_cv(X, y, C=1.1, kernel='rbf', gamma=0.1, epsilon=0.05, n_splits=5, random_state=123):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_metrics = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = SVR(C=C, kernel=kernel, gamma=gamma, epsilon=epsilon)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_metrics(y_test, y_pred)
        all_metrics.append(metrics)

    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if m[key] is not None]

        if isinstance(values[0], (int, float, np.float64, np.float32)):
            avg_metrics[key] = float(np.mean(values))
        elif isinstance(values[0], str):
            avg_metrics[key] = mode(values)  # maggioranza
        else:
            continue

    return avg_metrics, model


def create_results_table(metrics_sp, metrics_dp):
    results = pd.DataFrame([
        {"Target": "SP", **metrics_sp},
        {"Target": "DP", **metrics_dp}
    ])
    return results

def execute(data_path):
    with h5py.File(data_path, "r") as f:

        subject_to_groups = defaultdict(list)
        for group_id in f:
            subject_id = f[group_id].attrs["subject_id"]
            subject_to_groups[subject_id].append(group_id)

        subjects = list(subject_to_groups.keys())
        def get_group_ids(subject_list):
            return [gid for subj in subject_list for gid in subject_to_groups[subj]]

        ids = get_group_ids(subjects)

        def load_data(group_ids):
            X, Y = [], []
            for gid in group_ids:
                ippg_cwt = f[gid]["ippg"][:]
                bp_cwt = f[gid]["bp"][:]
                X.append(ippg_cwt)
                Y.append(bp_cwt)

            return np.array(X), np.array(Y)

        x, y = load_data(ids)

    X, y_sp, y_dp = create_feature_dataset(x,y)
    metrics_sp, svr_sp = train_svr_cv(X, y_sp)
    metrics_dp, svr_dp = train_svr_cv(X, y_dp)
    results = create_results_table(metrics_sp, metrics_dp)
    results.to_csv("svr_results.csv", index=False)
    print(results)

    joblib.dump(svr_sp, "svr_sp_model.pkl")
    joblib.dump(svr_dp, "svr_dp_model.pkl")


if __name__ == "__main__":
    data_path ="D:/iPPGtoBP/dataset_extracted/data_POS2.h5"
    execute(data_path)