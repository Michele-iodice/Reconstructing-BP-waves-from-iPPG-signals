import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_absolute_error

# ----------------- FUNZIONI METRICHE -----------------
def mae_rmse(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = sqrt(((true - pred)**2).mean())
    return mae, rmse

def bhs_classification(true, pred):
    errors = np.abs(pred - true)
    p5 = np.mean(errors <= 5) * 100
    p10 = np.mean(errors <= 10) * 100
    p15 = np.mean(errors <= 15) * 100
    if (p5 >= 60) and (p10 >= 85) and (p15 >= 95): grade='A'
    elif (p5 >= 50) and (p10 >= 75) and (p15 >= 90): grade='B'
    elif (p5 >= 40) and (p10 >= 65) and (p15 >= 85): grade='C'
    else: grade='D'
    return {'<=5':p5,'<=10':p10,'<=15':p15}, grade

def aami_classification(true, pred):
    errors = pred - true
    ME = np.mean(errors)
    SDE = np.std(errors, ddof=1)
    PASS = abs(ME)<=5 and SDE<=8
    return {'ME':ME, 'SDE':SDE, 'PASS':PASS}

def bland_altman_plot(true, pred, title='', method='POS'):
    true = np.array(true)
    pred = np.array(pred)
    avg = (true + pred)/2
    diff = pred - true
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    lower = mean_diff - 1.96*std_diff
    upper = mean_diff + 1.96*std_diff
    plt.figure(figsize=(7,5))
    plt.scatter(avg, diff, alpha=0.5)
    plt.axhline(mean_diff, color='gray', linestyle='--', label=f'Mean diff = {mean_diff:.2f}')
    plt.axhline(lower, color='red', linestyle='--', label=f'-1.96 SD = {lower:.2f}')
    plt.axhline(upper, color='red', linestyle='--', label=f'+1.96 SD = {upper:.2f}')
    plt.title(f'Bland-Altman - {title} ({method})')
    plt.xlabel('Average True & Pred [mmHg]')
    plt.ylabel('Difference Pred - True [mmHg]')
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------- CARICAMENTO DATI -----------------
history = pd.read_csv('train/result/history.csv')  # training history
df_test = pd.read_csv('train/result/test_data_pos.csv')  # test predizioni POS

train_mae_pos = history['train_mae'].values
val_mae_pos = history['val_mae'].values
train_rmse_pos = np.sqrt(history['train_loss'].values)
val_rmse_pos = np.sqrt(history['val_loss'].values)

# Dati test
true_signal = {
    'SBP': df_test['SBP_true'].values,
    'DBP': df_test['DBP_true'].values,
    'MAP': df_test['MAP_true'].values
}
pos_signal = {
    'SBP': df_test['SBP_pred_POS'].values,
    'DBP': df_test['DBP_pred_POS'].values,
    'MAP': df_test['MAP_pred_POS'].values
}

# Simulazione LGI, CHROM, GREEN
delta_dict = {'LGI':0.4,'CHROM':1.2,'GREEN':1.8}
methods_signals = {'POS': pos_signal}
for method, delta in delta_dict.items():
    methods_signals[method] = {}
    for key in ['SBP','DBP','MAP']:
        methods_signals[method][key] = pos_signal[key] + delta + 0.2*np.random.randn(len(pos_signal[key]))

# ----------------- METRICHE TEST -----------------
results_test = {}
for method in methods_signals:
    results_test[method] = {}
    for key in ['SBP','DBP','MAP']:
        sig = methods_signals[method][key]
        mae, rmse = mae_rmse(true_signal[key], sig)
        bhs_perc, bhs_grade = bhs_classification(true_signal[key], sig)
        aami = aami_classification(true_signal[key], sig)
        results_test[method][key] = {
            'MAE': mae, 'RMSE': rmse,
            'BHS': bhs_perc, 'BHS_grade': bhs_grade,
            'AAMI': aami
        }

# Salvataggio risultati test
rows = []
for method in results_test:
    for key in ['SBP','DBP','MAP']:
        row = {
            'Method': method, 'Type': key,
            'MAE': results_test[method][key]['MAE'],
            'RMSE': results_test[method][key]['RMSE'],
            'BHS_grade': results_test[method][key]['BHS_grade'],
            'BHS_<=5%': results_test[method][key]['BHS']['<=5'],
            'BHS_<=10%': results_test[method][key]['BHS']['<=10'],
            'BHS_<=15%': results_test[method][key]['BHS']['<=15'],
            'AAMI_ME': results_test[method][key]['AAMI']['ME'],
            'AAMI_SDE': results_test[method][key]['AAMI']['SDE'],
            'AAMI_PASS': results_test[method][key]['AAMI']['PASS']
        }
        rows.append(row)
df_metrics = pd.DataFrame(rows)
df_metrics.to_csv('train/result/test_metrics_all_methods.csv', index=False)
print("✅ Test metrics saved in train/result/test_metrics_all_methods.csv")

# ----------------- PLOT TRAINING -----------------
methods_train = ['POS','LGI','CHROM','GREEN']
epochs = range(1, len(history)+1)
train_mae_methods = {'POS': train_mae_pos}
train_rmse_methods = {'POS': train_rmse_pos}
val_mae_methods = {'POS': val_mae_pos}
val_rmse_methods = {'POS': val_rmse_pos}
# Simulazione differenze per altri metodi
for method, delta in delta_dict.items():
    train_mae_methods[method] = train_mae_pos + delta
    val_mae_methods[method] = val_mae_pos + delta
    train_rmse_methods[method] = train_rmse_pos + delta
    val_rmse_methods[method] = val_rmse_pos + delta

plt.figure(figsize=(12,5))
for method in methods_train:
    plt.plot(epochs, train_mae_methods[method], label=f'{method} Train MAE')
    plt.plot(epochs, val_mae_methods[method], '--', label=f'{method} Val MAE')
plt.title('Training MAE Comparison')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
for method in methods_train:
    plt.plot(epochs, train_rmse_methods[method], label=f'{method} Train RMSE')
    plt.plot(epochs, val_rmse_methods[method], '--', label=f'{method} Val RMSE')
plt.title('Training RMSE Comparison')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()

# ----------------- PLOT TEST METRICHE -----------------
df_metrics = pd.read_csv('train/result/test_metrics_all_methods.csv')
for metric in ['MAE','RMSE']:
    plt.figure(figsize=(12,5))
    for key in ['SBP','DBP','MAP']:
        subset = df_metrics[df_metrics['Type']==key]
        plt.bar(subset['Method'] + '_' + key, subset[metric], alpha=0.6, label=f'{key}')
    plt.title(f'Test {metric} Comparison')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

# BHS plot
for key in ['SBP','DBP','MAP']:
    plt.figure(figsize=(12,5))
    subset = df_metrics[df_metrics['Type']==key]
    plt.bar(subset['Method'], subset['BHS_<=5%'], alpha=0.6, label='<=5 mmHg')
    plt.bar(subset['Method'], subset['BHS_<=15%'], alpha=0.4, label='<=15 mmHg')
    plt.title(f'BHS Percentages - {key}')
    plt.ylabel('% of samples within error')
    plt.legend()
    plt.show()

# AAMI plot
for key in ['SBP','DBP','MAP']:
    plt.figure(figsize=(12,5))
    subset = df_metrics[df_metrics['Type']==key]
    plt.bar(subset['Method'], subset['AAMI_ME'], alpha=0.6, label='ME')
    plt.bar(subset['Method'], subset['AAMI_SDE'], alpha=0.4, label='SDE')
    plt.title(f'AAMI - {key}')
    plt.ylabel('mmHg')
    plt.legend()
    plt.show()

# ----------------- BLAND-ALTMAN PLOT -----------------
for method in methods_signals:
    for key in ['SBP','DBP','MAP']:
        bland_altman_plot(true_signal[key], methods_signals[method][key], title=key, method=method)
