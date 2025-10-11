from collections import defaultdict

import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model.unet_ippg_cwt import UNet, ModelAdapter
from model.utils import split_data, test_model
from config import Configuration
import numpy as np
import h5py
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model.utils import calculate_matrix, evaluate
from extraction.signal_to_cwt import inverse_cwt
from no_deep_methods.rf.rf import extract_features as rf_extract_features
from no_deep_methods.svr.svr import extract_features as svr_extract_features
from no_deep_methods.svr.svr import segment_beats

def predict_dataset(config, data_path):

    #PARAMETERS
    BATCH_SIZE = int(np.int32(config.uNetdict['BATCH_SIZE']))
    output_channels = config.get_array('output_channels')
    backbone_name = config.uNetdict['backbone_name']
    pretrained = config.get_boolean('UnetParameter', 'pretrained')
    freeze_backbone = config.get_boolean('UnetParameter', 'freeze_backbone')

    x_train, x_test, x_val, y_train, y_test, y_val = split_data(data_path)

    x_test_torch = torch.tensor(x_test).float()

    y_test_torch = torch.tensor(y_test).float()

    test_dataset = TensorDataset(x_test_torch, y_test_torch)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    in_channels = x_test_torch.shape[1]
    base_model = UNet(True,
                      out_channel=in_channels,
                      output_channels=output_channels,
                      backbone_name=backbone_name,
                      pretrained=pretrained,
                      freeze_backbone=freeze_backbone)


    model = ModelAdapter(base_model, in_channels)

    # loss and optimisation function definition
    criterion = nn.MSELoss()
    checkpoint_path = config.uNetdict['checkpoint_path']
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model_state_dict'], strict=False)
    print("\n start testing...\n")
    test_model(model, criterion, test_loader)

def predict_dataset_compare_model(config, dataset_path, model_path):
    def split_data(data_path):
        """
        Divided the data in input follow this steps:
        step1: group the data by subjects
        step2: Divide the IDs of subjects into train test and val (70% train, 15% test, 15% validation)
        step3: split data using subject's ID
        Step 4: data extraction of CWT and BP for each set (train, validation, test)
        :return: data divided into x,y of test, train and validation
        """
        print(f"start data splitting...")
        with h5py.File(data_path, "r") as f:

            subject_to_groups = defaultdict(list)
            for group_id in f:
                subject_id = f[group_id].attrs["subject_id"]
                subject_to_groups[subject_id].append(group_id)

            subjects = list(subject_to_groups.keys())

            train_subjects, test_subjects = train_test_split(subjects, test_size=0.30, random_state=42)
            val_subjects, test_subjects = train_test_split(test_subjects, test_size=0.50, random_state=42)

            def get_group_ids(subject_list):
                return [gid for subj in subject_list for gid in subject_to_groups[subj]]

            test_ids = get_group_ids(subjects)

            def load_data(group_ids):
                sig_ippgs, sig_bps, cwt_ippgs, cwt_bps, idss= [],[],[],[],[]
                for gid in group_ids:
                    ippg_cwt = f[gid]["ippg_cwt"][:]
                    bp_cwt = f[gid]["bp_cwt"][:]
                    ippg = f[gid]["ippg"][:]
                    bp = f[gid]["bp"][:]
                    id = gid
                    cwt_ippgs.append(ippg_cwt)
                    cwt_bps.append(bp_cwt)
                    sig_ippgs.append(ippg)
                    sig_bps.append(bp)
                    idss.append(id)


                return np.array(sig_ippgs), np.array(sig_bps),np.array(cwt_ippgs),np.array(cwt_bps), np.array(idss)

            sig_ippg_all, sig_bp_all, cwt_ippg_all, cwt_bp_all, ids = load_data(test_ids)


        return sig_ippg_all, sig_bp_all, cwt_ippg_all, cwt_bp_all, ids


    # === CARICA MODELLI ===
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

    # SVR e RF
    svr_model = joblib.load(model_path + "svr_model.pkl")
    rf_model = joblib.load(model_path + "rf_model.pkl")
    rf_scaler = joblib.load(model_path + "rf_scaler.pkl")


    # === RISULTATI ===
    records = []
    # === LETTURA DEL FILE H5 ===
    sig_ippg_all, sig_bp_all, cwt_ippg_all, cwt_bp_all, ids = split_data(dataset_path) # es: "Subject01_T1_1"

    if ids.dtype.kind == 'S':  # se sono bytes
        ids = [x.decode('utf-8') for x in ids]

    N = len(sig_ippg_all)
    print(f"Totale finestre nel dataset: {N}")

    for i in range(N):
        sig_ippg = sig_ippg_all[i]
        sig_bp = sig_bp_all[i]
        cwt_ippg = cwt_ippg_all[i]
        cwt_bp = cwt_bp_all[i]
        window_id = ids[i]

        # === Calcolo valori reali ===
        sbp_true, dbp_true, map_true = calculate_matrix(sig_bp)

        # === iPPG2BP ===
        cwt_tensor = torch.tensor(cwt_ippg).float().unsqueeze(0)
        with torch.no_grad():
            cwt_pred = ippg2bp_model(cwt_tensor)[0].cpu().numpy()
        bp_pred = inverse_cwt(cwt_pred, f_min=4.5, f_max=0.6)
        sbp_pred_ippg2bp, dbp_pred_ippg2bp, map_pred_ippg2bp = calculate_matrix(bp_pred)

        # === SVR ===
        try:
            segments = segment_beats(sig_ippg, sig_bp)
            X_test = []
            for seg_ppg, seg_bp in segments:
                feat = svr_extract_features(seg_ppg, seg_bp)
                X_test.append([v for k, v in feat.items() if k not in ['SP', 'DP']])
            if len(X_test) > 0:
                X_test = np.array(X_test)
                svr_pred_window = svr_model.predict(X_test)
                sbp_pred_svr, dbp_pred_svr = svr_pred_window[0]
                map_pred_svr = (2 * dbp_pred_svr + sbp_pred_svr) / 3
            else:
                sbp_pred_svr, dbp_pred_svr, map_pred_svr = np.nan, np.nan, np.nan
        except Exception as e:
            print(f"[SVR] Errore su {window_id}: {e}")
            sbp_pred_svr, dbp_pred_svr, map_pred_svr = np.nan, np.nan, np.nan

        # === RF ===
        try:
            rf_feature = rf_extract_features(sig_ippg, sig_bp)
            X = rf_feature[:2] + rf_feature[4:18]
            X = np.array(X).reshape(1, -1)
            X_scaled = rf_scaler.transform(X)
            rf_pred_window = rf_model.predict(X_scaled)
            sbp_pred_rf, dbp_pred_rf, map_pred_rf = rf_pred_window[0]
        except Exception as e:
            print(f"[RF] Errore su {window_id}: {e}")
            sbp_pred_rf, dbp_pred_rf, map_pred_rf = np.nan, np.nan, np.nan

        # === SALVATAGGIO ===
        records.append({
            'window_id': window_id,
            'SBP_true': sbp_true,
            'DBP_true': dbp_true,
            'MAP_true': map_true,
            'SBP_pred_iPPG2BP': sbp_pred_ippg2bp,
            'DBP_pred_iPPG2BP': dbp_pred_ippg2bp,
            'MAP_pred_iPPG2BP': map_pred_ippg2bp,
            'SBP_pred_SVR': sbp_pred_svr,
            'DBP_pred_SVR': dbp_pred_svr,
            'MAP_pred_SVR': map_pred_svr,
            'SBP_pred_RF': sbp_pred_rf,
            'DBP_pred_RF': dbp_pred_rf,
            'MAP_pred_RF': map_pred_rf
        })



    # === CREAZIONE DATAFRAME ===
    df_results = pd.DataFrame(records)

    # === METRICHE GLOBALI ===
    results = []
    for model_name, cols in {
        'iPPG2BP': ('SBP_pred_iPPG2BP', 'DBP_pred_iPPG2BP', 'MAP_pred_iPPG2BP'),
        'SVR': ('SBP_pred_SVR', 'DBP_pred_SVR', 'MAP_pred_SVR'),
        'RF': ('SBP_pred_RF', 'DBP_pred_RF', 'MAP_pred_RF')
    }.items():
        sbp_mae, sbp_rmse = evaluate(df_results['SBP_true'], df_results[cols[0]], model_name)
        dbp_mae, dbp_rmse = evaluate(df_results['DBP_true'], df_results[cols[1]], model_name)
        map_mae, map_rmse = evaluate(df_results['MAP_true'], df_results[cols[2]], model_name)

        results.append({
            'Model': model_name,
            'SBP_MAE': sbp_mae, 'SBP_RMSE': sbp_rmse,
            'DBP_MAE': dbp_mae, 'DBP_RMSE': dbp_rmse,
            'MAP_MAE': map_mae, 'MAP_RMSE': map_rmse
        })

    results_df = pd.DataFrame(results)
    print("\n=== RISULTATI GLOBALI ===")
    print(results_df)
    results_df.to_csv("results_over_dataset2.csv", index=False)


if __name__ == "__main__":
    '''config = Configuration(
        'C:/Users/Utente/Documents/GitHub/Reconstructing-BP-waves-from-iPPG-signals/scripts/python/config.cfg')
    dataset_path = config.uNetdict['dataset_path']
    predict_dataset(config, dataset_path)'''
    # === CONFIGURAZIONE ===
    dataset_path = "D:/iPPGtoBP/dataset_extracted/data_POS2.h5"
    model_path = "D:/iPPGtoBP/models/"
    config = Configuration(
        'C:/Users/Utente/Documents/GitHub/Reconstructing-BP-waves-from-iPPG-signals/scripts/python/config.cfg'
    )
    predict_dataset_compare_model(config, dataset_path, model_path)