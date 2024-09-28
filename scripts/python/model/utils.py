import csv
import torch
import numpy as np
from sklearn.model_selection import train_test_split


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

    return "File salved."


def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs, checkpoint_path, VERBOSE=True):
    """
    Function to train the model
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
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(valid_loader)
        history['val_loss'].append(val_loss)

        # show progress
        if VERBOSE:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # save model if it is the best
        if epoch == 0 or val_loss < min(history['val_loss'][:-1]):
            print(f"Validation loss decreased ({min(history['val_loss'][:-1]):.4f} -> {val_loss:.4f}). Saving model...")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

    # Save loss chronology in a CSV file
    with open('history.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
        for i in range(epochs):
            writer.writerow([i + 1, history['train_loss'][i], history['val_loss'][i]])

    return history


def split_data(data):
    """
    Divided the data in input follow this steps:
    step1: group the data by subjects
    step2: Divide the IDs of subjects into train test and val (64% train, 20% test, 16% validation)
    step3: split data using subject's ID
    Step 4: data extraction of CWT and BP for each set (train, validation, test)
    :param data: data to split
    :return: data divided into x,y of test, train and validation
    """

    subjects = data['subject_id'].unique()

    train_subjects, test_subjects = train_test_split(subjects, test_size=0.2, random_state=42)
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.25, random_state=42)

    train_data = data[data['subject_id'].isin(train_subjects)]
    val_data = data[data['subject_id'].isin(val_subjects)]
    test_data = data[data['subject_id'].isin(test_subjects)]

    x_train = np.array(list(train_data['CWT']))
    y_train = np.array(train_data['BP'])
    x_val = np.array(list(val_data['CWT']))
    y_val = np.array(val_data['BP'])
    x_test = np.array(list(test_data['CWT']))
    y_test = np.array(test_data['BP'])

    return x_train, x_test, x_val, y_train, y_test, y_val
