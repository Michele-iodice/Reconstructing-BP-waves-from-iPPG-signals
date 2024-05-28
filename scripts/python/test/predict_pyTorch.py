import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import json
import matplotlib.pyplot as plt

# GLOBAL VARIABLES
VERBOSE = True
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


class ChannelAdaptation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAdaptation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def load_model_from_json(json_path, weights_path, input_channels=2):
    with open(json_path, 'r') as json_file:
        model_architecture = json.load(json_file)

    model = nn.Sequential()
    for layer in model_architecture['layers']:
        layer_type = layer['type']
        if layer_type == 'Conv2d':
            model.add_module(layer['name'], nn.Conv2d(**layer['params']))
        elif layer_type == 'ReLU':
            model.add_module(layer['name'], nn.ReLU(**layer['params']))
        # Add other layer types as needed

    # Add channel adaptation if input channels != model input channels
    channel_adapter = ChannelAdaptation(input_channels, model_architecture['input_channels'])
    model = nn.Sequential(channel_adapter, model)

    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    return model


def load_data(file_path, variable_names):
    data = scipy.io.loadmat(file_path)
    data_dict = {}
    for var in variable_names:
        var_data = data[var]
        shaped_data = np.zeros(
            (var_data.shape[1], var_data[0, 0]['cfs'][0, 0].shape[0], var_data[0, 0]['cfs'][0, 0].shape[1], 2))
        for i in range(var_data.shape[1]):
            shaped_data[i, :, :, 0] = np.real(var_data[0, i]['cfs'][0, 0])
            shaped_data[i, :, :, 1] = np.imag(var_data[0, i]['cfs'][0, 0])
        data_dict[var] = shaped_data
    return data_dict


def create_dataloader(x, y):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader


def predict_and_save_results(model, test_loader, device, output_file):
    model.eval()
    results = []

    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            for pred in preds:
                results.append({'prediction': pred.cpu().numpy()})

    scipy.io.savemat(output_file, {'results': results})


    # Prepara i dati per il plot
    predictions = [result['prediction'] for result in results]

    # Assumendo che le previsioni siano array monodimensionali, altrimenti adatta il codice
    if isinstance(predictions[0], (list, np.ndarray)):
        predictions = [pred[0] for pred in predictions]

    # Genera il plot
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, 'bo-', label='Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction')
    plt.title('Model Predictions')
    plt.legend()
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = load_model_from_json('model.json', 'weights.pth')
model = model.to(device)

# Load test data
data = load_data('data_test.mat', ['CWT_ppg_test', 'CWT_bp_test'])
xtest, ytest = data['CWT_ppg_test'], data['CWT_bp_test']

# Create dataloader
test_loader = create_dataloader(xtest, ytest)

# Predict and save results
predict_and_save_results(model, test_loader, device, 'results.mat')



