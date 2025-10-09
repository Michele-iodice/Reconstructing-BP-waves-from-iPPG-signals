import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import Configuration
from model.utils import plotSignal, calculate_matrix
from extraction.signal_to_cwt import plotCWT, inverse_cwt
import joblib
from model.unet_ippg_cwt import UNet, ModelAdapter
import torch
from no_deep_methods.rf.rf import extract_features as rf_extract_features
from no_deep_methods.svr.svr import extract_features as svr_extract_features
from no_deep_methods.svr.svr import segment_beats

# --- CONFIGURAZIONE ---
videoFilename = "D:/datasetBP4D+/F001/T1/vid.avi"
bp_filename = "D:/datasetBP4D+/F001/T1/BP_mmHg.txt"
model_path = "D:/iPPGtoBP/models/"
config = Configuration(
    'C:/Users/Utente/Documents/GitHub/Reconstructing-BP-waves-from-iPPG-signals/scripts/python/config.cfg')

# --- ESTRAZIONE FEATURE ---
#feature = extract_feature_on_video(videoFilename, bp_filename, config)
feature_file = "features_BP4D.h5"

feature = []
with h5py.File(feature_file, 'r') as f:
    for key in f.keys():
        grp = f[key]
        row = {
            'sig_IPPG': np.array(grp['sig_IPPG']),
            'sig_BP': np.array(grp['sig_BP']),
            'CWT_IPPG': np.array(grp['CWT_IPPG']),
            'CWT_BP': np.array(grp['CWT_BP'])
        }
        feature.append(row)
feature = pd.DataFrame(feature)

# --- MODELLI ---
# iPPG2BP
checkpoint_path = model_path + "weights.pth"
backbone_name = config.uNetdict['backbone_name']
pretrained = config.get_boolean('UnetParameter', 'pretrained')
freeze_backbone = config.get_boolean('UnetParameter', 'freeze_backbone')
output_channels = config.get_array('output_channels')
in_channels = 2

base_model = UNet(True,
                  out_channel=in_channels,
                  output_channels=output_channels,
                  backbone_name=backbone_name,
                  pretrained=pretrained,
                  freeze_backbone=freeze_backbone)
ippg2bp_model = ModelAdapter(base_model, in_channels)
ippg2bp_model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model_state_dict'], strict=False)
ippg2bp_model.eval()

# SVR
svr_model = joblib.load(model_path + "svr_model.pkl")

# RF
rf_model = joblib.load(model_path + "rf_model.pkl")
rf_scaler = joblib.load(model_path + "rf_scaler.pkl")



# --- FUNZIONE DI VALUTAZIONE ---
def evaluate_all(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# Numero di finestre da plottare
N_plot = 3

# --- ITERAZIONE SU TUTTE LE FINESTRE ---
all_sbp_true, all_dbp_true, all_map_true = [], [], []
all_pred_ippg2bp, all_pred_svr, all_pred_rf= [], [], []

for idx, row in feature.iterrows():
    sig_ippg = row['sig_IPPG']
    sig_bp = row['sig_BP']
    cwt_ippg = row['CWT_IPPG']
    cwt_bp = row['CWT_BP']

    # GROUND TRUTH
    sbp_true, dbp_true, map_true = calculate_matrix(sig_bp)
    all_sbp_true.append(sbp_true)
    all_dbp_true.append(dbp_true)
    all_map_true.append(map_true)

    # --- iPPG2BP MODEL ---
    cwt_tensor = torch.tensor(cwt_ippg).float().unsqueeze(0)
    cwt_pred = ippg2bp_model(cwt_tensor)[0].detach().cpu().numpy()
    bp_pred = inverse_cwt(cwt_pred, f_min=4.5, f_max=0.6)
    sbp_pred_ippg2bp, dbp_pred_ippg2bp, map_pred_ippg2bp = calculate_matrix(bp_pred)
    all_pred_ippg2bp.append((sbp_pred_ippg2bp, dbp_pred_ippg2bp, map_pred_ippg2bp))

    # --- Plot dei segnali e CWT solo per le prime N_plot finestre ---
    if idx < N_plot:
        plotSignal(sig_ippg, f'iPPG signal - Window {idx}')
        plotSignal(sig_bp, f'BP signal - Window {idx}')
        plotCWT(cwt_ippg, [4.5, 0.6], np.int32(config.uNetdict['frameRate']), f'CWT iPPG - Window {idx}')
        plotCWT(cwt_bp, [4.5, 0.6], np.int32(config.uNetdict['frameRate']), f'CWT BP predetto - Window {idx}')

    # --- SVR MODEL ---
    segments = segment_beats(sig_ippg, sig_bp)
    X_test = []
    for seg_ppg, seg_bp in segments:
        feat = svr_extract_features(seg_ppg, seg_bp)
        X_test.append([v for k, v in feat.items() if k not in ['SP','DP']])
    if len(X_test) > 0:
        X_test = np.array(X_test)
        svr_pred_window = svr_model.predict(X_test)
        sbp_pred_svr, dbp_pred_svr = svr_pred_window[0]
        map_pred_svr = (2*dbp_pred_svr + sbp_pred_svr)/3
    else:
        sbp_pred_svr, dbp_pred_svr, map_pred_svr = np.nan, np.nan, np.nan
    all_pred_svr.append((sbp_pred_svr, dbp_pred_svr, map_pred_svr))

    # --- RF MODEL ---
    rf_feature = rf_extract_features(sig_ippg, sig_bp)
    X = rf_feature[:2] + rf_feature[4:18]
    X = np.array(X).reshape(1, -1)
    X_scaled = rf_scaler.transform(X)
    rf_pred_window = rf_model.predict(X_scaled)
    sbp_pred_rf, dbp_pred_rf, map_pred_rf = rf_pred_window[0]
    all_pred_rf.append((sbp_pred_rf, dbp_pred_rf, map_pred_rf))


# --- CONVERSIONE IN ARRAY ---
all_sbp_true = np.array(all_sbp_true)
all_dbp_true = np.array(all_dbp_true)
all_map_true = np.array(all_map_true)

predictions_all = {
    'iPPG2BP': all_pred_ippg2bp,
    'SVR': all_pred_svr,
    'RF': all_pred_rf
}

# --- VALUTAZIONE MODELLI ---
results = []
for model_name, preds in predictions_all.items():
    sbp_pred = np.array([p[0] for p in preds])
    dbp_pred = np.array([p[1] for p in preds])
    map_pred = np.array([p[2] for p in preds])

    sbp_mae, sbp_rmse = evaluate_all(all_sbp_true, sbp_pred)
    dbp_mae, dbp_rmse = evaluate_all(all_dbp_true, dbp_pred)
    map_mae, map_rmse = evaluate_all(all_map_true, map_pred)

    results.append({
        'Model': model_name,
        'SBP_MAE': sbp_mae, 'SBP_RMSE': sbp_rmse,
        'DBP_MAE': dbp_mae, 'DBP_RMSE': dbp_rmse,
        'MAP_MAE': map_mae, 'MAP_RMSE': map_rmse
    })

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('result.csv', index=False)
