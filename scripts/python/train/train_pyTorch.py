import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from segmentation_models_pytorch import Unet
import scipy.io

# GLOBAL VARIABLES
BACKBONE = 'resnext101_32x4d'
FREEZE_ENCODER = False
VERBOSE = True
EPOCHS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    x_data = np.zeros((data['CWT_ppg_training'].shape[1], data['CWT_ppg_training'][0, 0]['cfs'][0, 0].shape[0],
                       data['CWT_ppg_training'][0, 0]['cfs'][0, 0].shape[1], 2))
    y_data = np.zeros((data['CWT_bp_training'].shape[1], data['CWT_bp_training'][0, 0]['cfs'][0, 0].shape[0],
                       data['CWT_bp_training'][0, 0]['cfs'][0, 0].shape[1], 2))

    for i in range(data['CWT_ppg_training'].shape[1]):
        x_data[i, :, :, 0] = np.real(data['CWT_ppg_training'][0, i]['cfs'][0, 0])
        x_data[i, :, :, 1] = np.imag(data['CWT_ppg_training'][0, i]['cfs'][0, 0])
        y_data[i, :, :, 0] = np.real(data['CWT_bp_training'][0, i]['cfs'][0, 0])
        y_data[i, :, :, 1] = np.imag(data['CWT_bp_training'][0, i]['cfs'][0, 0])

    return x_data, y_data


def create_dataloaders(xtrain, ytrain, xvalid, yvalid):
    xtrain_tensor = torch.tensor(xtrain, dtype=torch.float32)
    ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32)
    xvalid_tensor = torch.tensor(xvalid, dtype=torch.float32)
    yvalid_tensor = torch.tensor(yvalid, dtype=torch.float32)

    train_dataset = TensorDataset(xtrain_tensor, ytrain_tensor)
    valid_dataset = TensorDataset(xvalid_tensor, yvalid_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader


class ChannelAdaptation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAdaptation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def train_model(model, train_loader, valid_loader, criterion, optimizer, device):
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                valid_loss += loss.item()

        if VERBOSE:
            print(
                f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {valid_loss / len(valid_loader):.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), 'weights.pth')


def save_model_architecture(model, filepath):
    with open(filepath, 'w') as f:
        f.write(str(model))


# Main script
xtrain, ytrain = load_data('data_training.mat')
xvalid, yvalid = load_data('data_validation.mat')

train_loader, valid_loader = create_dataloaders(xtrain, ytrain, xvalid, yvalid)

model = Unet(BACKBONE, classes=xtrain.shape[3], encoder_weights='imagenet', encoder_freeze=FREEZE_ENCODER)

channel_adapter = ChannelAdaptation(xtrain.shape[3], 3)
model = nn.Sequential(channel_adapter, model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_model(model, train_loader, valid_loader, criterion, optimizer, device)

save_model_architecture(model, 'model_architecture.json')
