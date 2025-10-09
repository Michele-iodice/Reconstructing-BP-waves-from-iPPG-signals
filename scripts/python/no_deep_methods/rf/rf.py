import joblib
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from collections import defaultdict
import h5py
from scipy.signal import butter, filtfilt
from sklearn.multioutput import MultiOutputRegressor

# ---------------------------------
# PARAMETERS
# ---------------------------------
fs = 100  # frequency rate
window_size_sec = 8  # slide windowed
window_samples = fs * window_size_sec
v = [0.1, 0.25, 0.33, 0.5, 0.66, 0.75]  # PPG % levels

# ---------------------------------
# FILTER FUNCTION
# ---------------------------------
def bandpass_filter(signal, fs=30):
    cutoff = 5  # Hz
    b, a = butter(2, cutoff / (fs / 2), btype='low')
    ppg_filtered = filtfilt(b, a, signal)
    return ppg_filtered

# ---------------------------------
# FEATURE EXTRACTION
# ---------------------------------
def extract_features(ppg, bp):
    ppg = np.array(ppg)
    bp = np.array(bp)

    # PPG second derivative
    ppg_2nd = np.gradient(np.gradient(ppg))

    # PPG peaks
    loc, property = find_peaks(ppg, prominence=0.0001)
    pk = ppg[loc]
    PPG1 = np.max(ppg) - ppg
    loc1, property = find_peaks(PPG1, prominence=0.0001)
    pk1 = PPG1[loc1]

    if len(loc) == 0 or len(loc1) < 2:
        return [0]*19

    # Systolic e diastolic time
    sys_time = np.mean([(loc[i] - loc1[i]) / fs for i in range(min(10, len(loc), len(loc1)))])
    dias_time = np.mean([(loc1[i + 1] - loc[i]) / fs for i in range(min(10, len(loc) - 1, len(loc1) - 1))])

    # Up/down time per livelli
    ppg_21_st = []
    ppg_21_dt = []
    for j in range(6):
        a_idx = next((i for i in range(loc1[0], loc[0] + 1) if ppg[i] >= v[j] * pk[0] + pk1[0]), loc1[0])
        b_idx = next((i for i in range(loc[0], loc1[1] + 1) if ppg[i] <= v[j] * pk[0] + pk1[0]), loc[0])
        ppg_21_st.append((loc[0] - a_idx) / fs)
        ppg_21_dt.append((b_idx - loc[0]) / fs)

    # Main features
    ih = np.mean([ppg[i] for i in loc]) if len(loc) > 0 else 0
    il = np.mean([ppg[i] for i in loc1]) if len(loc1) > 0 else 0
    PIR = ih / il if il != 0 else 0

    # BP peaks/valleys
    bp_peaks, _ = find_peaks(bp)
    bp_valleys, _ = find_peaks(np.max(bp) - bp)
    bpmax = np.mean(bp[bp_peaks]) if len(bp_peaks) > 0 else 0
    bpmin = np.mean(bp[bp_valleys]) if len(bp_valleys) > 0 else 0

    rr_intervals = np.diff(bp_peaks) / fs if len(bp_peaks) > 1 else [1]
    hrfinal = 60 / np.mean(rr_intervals)

    # FFT PPG
    Yy = np.fft.fft(ppg)
    Yy[0] = 0
    S = np.real(np.fft.ifft(Yy))
    pk4, loc4 = find_peaks(S)
    meu = np.mean([S[i] for i in loc4]) if len(loc4) > 0 else 0

    # alpha
    alpha = il * np.sqrt(1060 * hrfinal / meu) if meu != 0 else 0

    # j-s variables
    j = ppg_21_dt[0]
    k = ppg_21_st[0] + ppg_21_dt[0]
    l = ppg_21_dt[0] / ppg_21_st[0] if ppg_21_st[0] != 0 else 0
    m = ppg_21_dt[1]
    n = ppg_21_st[1] + ppg_21_dt[1]
    o = ppg_21_dt[1] / ppg_21_st[1] if ppg_21_st[1] != 0 else 0
    p = ppg_21_dt[2]
    q = ppg_21_st[2] + ppg_21_dt[2]
    r = ppg_21_dt[2] / ppg_21_st[2] if ppg_21_st[2] != 0 else 0
    s = sys_time

    features = [alpha, PIR, bpmax, bpmin, hrfinal, ih, il, meu,
                j, k, l, m, n, o, p, q, r, s, dias_time]
    return features

# ---------------------------------
# EVALUATION METRICS
# ---------------------------------
def evaluate(y_true, y_pred):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

    errors = y_pred - y_true
    me = np.mean(errors)
    sde = np.std(errors, ddof=1)
    abs_errors = np.abs(errors)

    aami_mae = abs(me) < 5
    aami_sd = sde < 8
    aami_compliance = aami_mae and aami_sd

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

# ---------------------------------
# TRAIN MULTI-OUTPUT RANDOM FOREST
# ---------------------------------
def train_rf_evaluate(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0
    )
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    base_model = RandomForestRegressor(n_estimators=500, random_state=2)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    metrics_dict = {}
    for i, target_name in enumerate(["SBP", "DBP", "MAP"]):
        metrics_dict[target_name] = evaluate(Y_test[:, i], Y_pred[:, i])

    joblib.dump(model, f"rf_model.pkl")
    joblib.dump(sc_X, f"rf_scaler.pkl")
    return metrics_dict, model

# ---------------------------------
# EXECUTE FUNCTION
# ---------------------------------
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

    features_list = []
    end = min(len(x), len(y))
    for i in range(0,end):
        ppg_signal = x[i]
        bp_signal = y[i]
        feat = extract_features(ppg_signal, bp_signal)
        features_list.append(feat)

    columns = ['alpha', 'PIR', 'bpmax', 'bpmin', 'hrfinal', 'ih', 'il', 'meu',
               'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'dias_time']

    features_df = pd.DataFrame(features_list, columns=columns)
    features_df.replace([np.inf, -np.inf], 0, inplace=True)
    features_df.fillna(0, inplace=True)

    sbp = features_df['bpmax'].values
    dbp = features_df['bpmin'].values
    map_bp = (2*dbp + sbp)/3
    Y_multi = np.vstack([sbp, dbp, map_bp]).T

    X_features = features_df[['alpha', 'PIR', 'hrfinal', 'ih', 'il', 'meu',
                              'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's']]

    metrics_dict, model = train_rf_evaluate(X_features, Y_multi)

    results_df = pd.DataFrame([{**{"Target": t}, **m} for t, m in metrics_dict.items()])
    results_df.to_csv("rf_results.csv", index=False)
    print(results_df)

# ---------------------------------
# MAIN
# ---------------------------------
if __name__ == "__main__":
    data_path = "D:/iPPGtoBP/dataset_extracted/data_POS2.h5"
    execute(data_path)
