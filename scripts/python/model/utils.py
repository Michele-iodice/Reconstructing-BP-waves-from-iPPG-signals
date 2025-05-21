import csv
from collections import defaultdict
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
from tqdm import tqdm
import sys

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
    :return: loss chronology
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
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            batch_mae = mean_absolute_error(
                targets.cpu().detach().numpy().flatten(),
                outputs.cpu().detach().numpy().flatten()
            )

            running_loss += batch_loss * inputs.size(0)
            running_mae += batch_mae * inputs.size(0)
            train_bar.set_postfix(loss=batch_loss, mae=batch_mae)

        epoch_loss = running_loss / len(train_loader)
        epoch_mae = running_mae / len(train_loader)
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
                loss = criterion(outputs, targets)
                batch_loss = loss.item()
                batch_mae = mean_absolute_error(
                    targets.cpu().numpy().flatten(),
                    outputs.cpu().numpy().flatten()
                )

                val_loss += batch_loss * inputs.size(0)
                val_mae += batch_mae * inputs.size(0)
                val_bar.set_postfix(val_loss=batch_loss, val_mae=batch_mae)

        val_loss /= len(valid_loader)
        val_mae /= len(valid_loader)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        # show progress
        if VERBOSE:
            tqdm.write(f'Epoch {epoch + 1}/{epochs}, '
                       f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, '
                       f'Train MAE: {epoch_mae:.4f}, Val MAE: {val_mae:.4f}')

        # save model if it is the best
        if epoch == 0:
            tqdm.write(f"Validation loss -> {val_loss:.4f}). Saving model...")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        elif val_loss < min(history['val_loss'][:-1]):
            tqdm.write(f"Validation loss decreased ({min(history['val_loss'][:-1]):.4f} -> {val_loss:.4f}). Saving model...")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)


    # Save loss chronology in a CSV file
    with open('history.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train MAE', 'Val Loss', 'Val MAE'])
        for i in range(epochs):
            writer.writerow([i + 1,
                             history['train_loss'][i], history['train_mae'][i],
                             history['val_loss'][i], history['val_mae'][i]])

    return model, history


def test_model(model, criterion, test_loader):
    model.eval()

    test_loss = 0.0
    test_mae = 0.0

    test_bar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for inputs, targets in test_bar:

            inputs = inputs.to(next(model.parameters()).device)
            targets = targets.to(next(model.parameters()).device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_loss = loss.item()
            batch_mae = mean_absolute_error(
                targets.cpu().numpy().flatten(),
                outputs.cpu().numpy().flatten()
            )
            test_loss += batch_loss * inputs.size(0)
            test_mae += batch_mae * inputs.size(0)

            test_bar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "mae": f"{batch_mae:.4f}"
            })

    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)

    test_results = {
        'Test Loss': [test_loss],
        'Test MAE': [test_mae]
    }

    df_results = pd.DataFrame(test_results)
    df_results.to_csv('test_results.csv', index=False)
    tqdm.write(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')


def split_data(data_path):
    """
    Divided the data in input follow this steps:
    step1: group the data by subjects
    step2: Divide the IDs of subjects into train test and val (64% train, 20% test, 16% validation)
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

        train_subjects, test_subjects = train_test_split(subjects, test_size=0.2, random_state=42)
        train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

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
