import csv
import os
from collections import defaultdict
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
from extraction.signal_to_cwt import inverse_cwt

def save_checkpoint(model, optimizer, epoch, loss, file_path):
    """
    Function to save model checkpoint
    :param model: model to use
    :param optimizer: optimizer choice
    :param epoch: number of epoch
    :param loss: loss function choice
    :param file_path: destination path of the file
    :return: ok if it is process
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, file_path)


def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs, checkpoint_path, VERBOSE=True):
    """
    Function to train the model.
    :param model: model to train
    :param criterion: criterion method to use for loss
    :param optimizer: optimizer method to use
    :param train_loader: train data
    :param valid_loader: validation data
    :param epochs: number of epoch
    :param checkpoint_path: destination path to save the model
    :param VERBOSE: if it is true show the progress line
    :return: loss and mae chronology
    """
    history = {'train_loss': [],
               'val_loss': [],
               'train_mae': [],
               'val_mae': []}

    for epoch in tqdm(range(epochs), desc="Training Progress", leave=True):
        model.train()
        running_loss = 0.0
        running_mae = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
        for inputs, targets in train_bar:
            inputs = inputs.to(next(model.parameters()).device)
            targets = targets.to(next(model.parameters()).device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss, mae = compute_batch_metrics(outputs, targets, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_mae += mae * inputs.size(0)
            train_bar.set_postfix(loss=loss.item(), mae=mae)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_mae = running_mae / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['train_mae'].append(epoch_mae)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_bar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} [Valid]", leave=False)
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs = inputs.to(next(model.parameters()).device)
                targets = targets.to(next(model.parameters()).device)

                outputs = model(inputs)
                loss, mae = compute_batch_metrics(outputs, targets, criterion)

                val_loss += loss.item() * inputs.size(0)
                val_mae += mae * inputs.size(0)
                val_bar.set_postfix(val_loss=loss.item(), val_mae=mae)

        val_loss /= len(valid_loader.dataset)
        val_mae /= len(valid_loader.dataset)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        # show progress
        if VERBOSE:
            tqdm.write(f'\nEpoch {epoch + 1}/{epochs}, '
                       f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, '
                       f'Train MAE: {epoch_mae:.4f}, Val MAE: {val_mae:.4f},')

        # save model if it is the best
        if epoch == 0:
            tqdm.write(f"\nValidation loss -> {val_loss:.4f}). Saving model...")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        elif val_loss < min(history['val_loss'][:-1]):
            tqdm.write(f"\nValidation loss decreased ({min(history['val_loss'][:-1]):.4f} -> {val_loss:.4f}). Saving model...")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)


    # Save loss chronology in a CSV file
    with open('result/history.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train MAE', 'Val Loss', 'Val MAE'])
        for i in range(epochs):
            writer.writerow([i + 1,
                             history['train_loss'][i], history['train_mae'][i],
                             history['val_loss'][i], history['val_mae'][i]])

    return history


def test_model(model, criterion, test_loader):
    model.eval()

    test_loss = 0.0
    test_mae = 0.0
    nan_count = 0
    all_dbp_true, all_dbp_pred = [], []
    all_map_true, all_map_pred = [], []
    all_sbp_true, all_sbp_pred = [], []
    all_target = []
    all_prediction = []

    test_bar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for inputs, targets in test_bar:
            inputs = inputs.to(next(model.parameters()).device)
            targets = targets.to(next(model.parameters()).device)

            outputs = model(inputs)
            if torch.isnan(outputs).any():
                nan_count += 1
                continue

            all_target.append(targets)
            all_prediction.append(outputs)

            loss, mae = compute_batch_metrics(outputs, targets, criterion)

            batch_test_loss = loss.item() * inputs.size(0)
            batch_test_mae = mae * inputs.size(0)
            test_loss += batch_test_loss
            test_mae += batch_test_mae

            test_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mae": f"{mae:.4f}"
            })

    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)

    print("\n start metrics test...\n")
    metrics_bar = tqdm(zip(all_target, all_prediction), total=len(all_prediction), desc="Metrics Test")
    nan_metrics = 0
    for target, prediction in metrics_bar:
        outputs_np = prediction.cpu().numpy()
        targets_np = target.cpu().numpy()

        for i in range(outputs_np.shape[0]):
            cwt_pred = outputs_np[i]
            cwt_true = targets_np[i]

            bp_pred = inverse_cwt(cwt_pred, f_min=0.6, f_max=4.5)
            bp_true = inverse_cwt(cwt_true, f_min=0.6, f_max=4.5, recover=True)

            sbp_pred, dbp_pred, map_pred = calculate_matrix(bp_pred)
            sbp_true, dbp_true, map_true = calculate_matrix(bp_true)

            all_sbp_pred.append(sbp_pred)
            all_dbp_pred.append(dbp_pred)
            all_map_pred.append(map_pred)

            all_sbp_true.append(sbp_true)
            all_dbp_true.append(dbp_true)
            all_map_true.append(map_true)

    results = test_metrics_with_bland_altman(
        DBP_true=np.array(all_dbp_true), DBP_pred=np.array(all_dbp_pred),
        MAP_true=np.array(all_map_true), MAP_pred=np.array(all_map_pred),
        SBP_true=np.array(all_sbp_true), SBP_pred=np.array(all_sbp_pred)
    )

    test_results = {
        'Test Loss': [test_loss],
        'Test MAE': [test_mae],
        'NaN batches': [nan_count],
        'Nan metrics': [nan_metrics]
    }

    df_results = pd.DataFrame(test_results)
    df_results.to_csv('result/all_test_results.csv', mode='a', header=not os.path.exists('result/all_test_results.csv'), index=False)
    tqdm.write(f'\nTest Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f} NaN batches: {nan_count}, NaN metrics: {nan_metrics}')
    save_test(results,'result/test_results.csv')


def test_metrics_with_bland_altman(DBP_true, DBP_pred, MAP_true, MAP_pred, SBP_true, SBP_pred):
    results = {}

    # DBP metrics
    dbp_mae, dbp_rmse = mae_rmse_metrix(DBP_true, DBP_pred)
    dbp_bhs_perc, dbp_bhs_grade = bhs_classification(DBP_true, DBP_pred)
    results['DBP'] = {
        'MAE': dbp_mae,
        'RMSE': dbp_rmse,
        'BHS_percentages': dbp_bhs_perc,
        'BHS_grade': dbp_bhs_grade
    }
    bland_altman_plot(DBP_true, DBP_pred, title='DBP', range_limits=[-12.3, 14.3])

    # MAP metrics
    map_mae, map_rmse = mae_rmse_metrix(MAP_true, MAP_pred)
    map_bhs_perc, map_bhs_grade = bhs_classification(MAP_true, MAP_pred)
    results['MAP'] = {
        'MAE': map_mae,
        'RMSE': map_rmse,
        'BHS_percentages': map_bhs_perc,
        'BHS_grade': map_bhs_grade
    }
    bland_altman_plot(MAP_true, MAP_pred, title='MAP', range_limits=[-12.0, 11.6])

    # SBP metrics
    sbp_mae, sbp_rmse = mae_rmse_metrix(SBP_true, SBP_pred)
    sbp_bhs_perc, sbp_bhs_grade = bhs_classification(SBP_true, SBP_pred)
    results['SBP'] = {
        'MAE': sbp_mae,
        'RMSE': sbp_rmse,
        'BHS_percentages': sbp_bhs_perc,
        'BHS_grade': sbp_bhs_grade
    }
    bland_altman_plot(SBP_true, SBP_pred, title='SBP', range_limits=[-19.6, 16.6])

    tqdm.write("\n=== Risultati Test ===")
    for key in ['DBP', 'MAP', 'SBP']:
        tqdm.write(
            f"{key}: MAE={results[key]['MAE']:.3f}, RMSE={results[key]['RMSE']:.3f}, BHS grade={results[key]['BHS_grade']}")
        tqdm.write(f"BHS percentages: {results[key]['BHS_percentages']}")

    return results


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

        train_ids = get_group_ids(train_subjects)
        val_ids = get_group_ids(val_subjects)
        test_ids = get_group_ids(test_subjects)

        def load_data(group_ids):
            X, Y = [], []
            for gid in group_ids:
                ippg_cwt = f[gid]["ippg_cwt"][:]
                bp_cwt = f[gid]["bp_cwt"][:]
                X.append(ippg_cwt)
                Y.append(bp_cwt)
            return np.array(X), np.array(Y)

        x_train, y_train = load_data(train_ids)
        x_val, y_val = load_data(val_ids)
        x_test, y_test = load_data(test_ids)

    return x_train, x_test, x_val, y_train, y_test, y_val


def compute_batch_metrics(outputs, targets, criterion):

    # sqrt(Re² + Im²)
    outputs_mod = torch.sqrt(outputs[:, 0] ** 2 + outputs[:, 1] ** 2)  # [B, H, W]
    targets_mod = torch.sqrt(targets[:, 0] ** 2 + targets[:, 1] ** 2)  # [B, H, W]

    loss = criterion(outputs_mod, targets_mod)

    outputs_flat = outputs_mod.detach().cpu().numpy().flatten()
    targets_flat = targets_mod.detach().cpu().numpy().flatten()

    mae = mean_absolute_error(targets_flat, outputs_flat)

    return loss, mae

def calculate_matrix(signal):
    """
    This method extracts SBP, DBP and MAP from a BP signal

      SBP = average of the maximum peak values (max peaks) of the signal

      DBP = average of the minimum peak values (min peaks) of the signal

      MAP = average of all the samples of the signal
    :param signal: BP signal
    :return: SBP, DBP and MAP
    """
    systolic_peaks_idx, _ = find_peaks(signal,distance=20, prominence=5)
    systolic_peaks = signal[systolic_peaks_idx]

    diastolic_peaks_idx, _ = find_peaks(-signal,distance=20, prominence=3)
    diastolic_peaks = signal[diastolic_peaks_idx]

    sbp = np.mean(systolic_peaks) if len(systolic_peaks) > 0 else np.max(signal)
    dbp = np.mean(diastolic_peaks) if len(diastolic_peaks) > 0 else np.min(signal)
    map = np.mean(signal)

    return  sbp, dbp, map

def mae_rmse_metrix(true_values, pred_values):
    """
        Calcola MAE e RMSE tra true_values e pred_values.
        Entrambi devono essere numpy array di stessa forma.
        """
    mae = mean_absolute_error(true_values, pred_values)
    rmse = root_mean_squared_error(true_values, pred_values)

    return mae, rmse


def bhs_classification(true_values, pred_values):
    """
    Calcola le percentuali di errori entro 5, 10 e 15 mmHg e assegna la classe BHS.

    Ritorna:
    - dict con percentuali
    - classe BHS ('A', 'B' o 'C')
    """
    errors = np.abs(pred_values - true_values)
    p5 = np.mean(errors <= 5) * 100
    p10 = np.mean(errors <= 10) * 100
    p15 = np.mean(errors <= 15) * 100

    if (p5 >= 60) and (p10 >= 85) and (p15 >= 95):
        grade = 'A'
    elif (p5 >= 50) and (p10 >= 75) and (p15 >= 90):
        grade = 'B'
    elif (p5 >= 40) and (p10 >= 65) and (p15 >= 85):
        grade = 'C'
    else:
        grade = 'D'

    return {'<=5mmHg': p5, '<=10mmHg': p10, '<=15mmHg': p15}, grade


def save_test(results, csv_path):
    rows = []
    for key in ['DBP', 'MAP', 'SBP']:
        row = {
            'Type': key,
            'MAE': results[key]['MAE'],
            'RMSE': results[key]['RMSE'],
            'BHS Grade': results[key]['BHS_grade'],
            'BHS <=5 mmHg (%)': results[key]['BHS_percentages']['<=5mmHg'],
            'BHS <=10 mmHg (%)': results[key]['BHS_percentages']['<=10mmHg'],
            'BHS <=15 mmHg (%)': results[key]['BHS_percentages']['<=15mmHg']
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

def plot_train(history):
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # MAE plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Train vs Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()


def plotComparison(GT_signal, predicted_signal):
    time_len= max(len(GT_signal[0][0][0]), len(predicted_signal[0][0][0]))
    time = np.linspace(0, 2.5, time_len)
    plt.figure(figsize=(12, 6))
    plt.plot(time, predicted_signal, label='Pred Signal', color='blue')
    plt.plot(time, GT_signal, label='GT Signal', linestyle='--', color='red')
    plt.ylabel('Blood Pressure (mmHg)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()

def plotSignal(signal):
    time_len= len(signal[0][0][0])
    time = np.linspace(0, 2.5, time_len)
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal, label='Signal', linestyle='-', color='black')
    plt.ylabel('Blood Pressure (mmHg)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()

def plotTest(iPPG_signal, GT_BP, reconstructed_BP):
    time_len = max(len(iPPG_signal[0][0][0]), len(GT_BP[0][0][0]))
    time_len = max(time_len, len(reconstructed_BP[0][0][0]))
    time = np.linspace(0, 2.5, time_len)
    plt.figure(figsize=(12, 6))
    plt.plot(time, iPPG_signal, label='iPPG Signal', color='black')
    plt.plot(time, reconstructed_BP, label='Pred Signal', color='blue')
    plt.plot(time, GT_BP, label='GT Signal', linestyle='--', color='red')
    plt.legend()
    plt.show()


def plot_metrics_histogram(dbp, map_, sbp, title):
    bins = np.linspace(40, 180, 50)

    plt.hist(dbp, bins=bins, alpha=0.5, label='DBP', color='blue')
    plt.hist(map_, bins=bins, alpha=0.5, label='MAP', color='green')
    plt.hist(sbp, bins=bins, alpha=0.5, label='SBP', color='red')

    plt.title(title)
    plt.xlabel('Blood Pressure (mmHg)')
    plt.ylabel('Number of samples')
    plt.legend()
    plt.grid(True)


def plot_sets_metrics(train, validation, test):
    sbp_train=train[0]
    dbp_train=train[1]
    map_train=train[2]
    sbp_val=validation[0]
    dbp_val=validation[1]
    map_val=validation[2]
    sbp_test=test[0]
    dbp_test=test[1]
    map_test=test[2]

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plot_metrics_histogram(dbp_train, map_train, sbp_train, 'Training Set')
    plt.subplot(1, 3, 2)
    plot_metrics_histogram(dbp_val, map_val, sbp_val, 'Validation Set')
    plt.subplot(1, 3, 3)
    plot_metrics_histogram(dbp_test, map_test, sbp_test, 'Test Set')
    plt.tight_layout()
    plt.show()


def bland_altman_plot(ground_truth, predictions, title, range_limits=None):
    avg = (ground_truth + predictions) / 2
    diff = predictions - ground_truth
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    lower_limit = mean_diff - 1.96 * std_diff
    upper_limit = mean_diff + 1.96 * std_diff

    plt.figure(figsize=(7, 5))
    plt.scatter(avg, diff, alpha=0.5)
    plt.axhline(mean_diff, color='gray', linestyle='-.', label=f'Mean = {mean_diff:.2f}')
    plt.axhline(lower_limit, color='red', linestyle='--', label=f'-1.96 SD = {lower_limit:.2f}')
    plt.axhline(upper_limit, color='red', linestyle='--', label=f'+1.96 SD = {upper_limit:.2f}')
    if range_limits:
        plt.ylim(range_limits)
    plt.xlabel('Avg of GT and prediction [mmHg]')
    plt.ylabel('Prediction error [mmHg]')
    plt.title(f'Bland–Altman Plot - {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()