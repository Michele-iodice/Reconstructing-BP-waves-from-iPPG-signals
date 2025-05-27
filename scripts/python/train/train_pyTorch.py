import torch.nn as nn
import torch
import torch.optim as optim
import json
from torch.utils.data import DataLoader, TensorDataset
from model.unet_ippg_cwt import UNet, ModelAdapter
from extraction.feature_extraction import extract_feature_on_dataset
from model.utils import split_data, train_model, plot_train, test_model
from config import Configuration
import numpy as np
from torchsummary import summary


def train_models(config, extract_data=False,):
    # Parameter
    BATCH_SIZE = int(np.int32(config.uNetdict['BATCH_SIZE']))
    EPOCHS = np.int32(config.uNetdict['EPOCHS'])
    VERBOSE = config.get_boolean('UnetParameter', 'VERBOSE')
    data_path = config.uNetdict['data_path']
    cardinality = np.int32(config.uNetdict['cardinality'])
    n_blocks1 = np.int32(config.uNetdict['n_blocks1'])
    n_blocks2 = np.int32(config.uNetdict['n_blocks2'])
    n_blocks3 = np.int32(config.uNetdict['n_blocks3'])
    n_blocks4 = np.int32(config.uNetdict['n_blocks4'])
    output_channels = config.get_array('output_channels')
    backbone_name = config.uNetdict['backbone_name']
    pretrained = config.get_boolean('UnetParameter', 'pretrained')
    freeze_backbone = config.get_boolean('UnetParameter', 'freeze_backbone')
    checkpoint_path = config.uNetdict['checkpoint_path']
    model_path = config.uNetdict['model_path']

    if extract_data is True:
        print(f"start new training")
        print(f"Data Extraction...")
        extract_feature_on_dataset(config,data_path)
    else:
        print(f"start new training from pre-extracted data")

    x_train, x_test, x_val, y_train, y_test, y_val = split_data(data_path)

    x_train = torch.tensor(x_train).float()  # [N, 1, 256, 256]
    x_val = torch.tensor(x_val).float()
    x_test = torch.tensor(x_test).float()

    y_train = torch.tensor(y_train).float()
    y_val = torch.tensor(y_val).float()
    y_test = torch.tensor(y_test).float()
    in_channels = x_train.shape[1]
    base_model = UNet(True,
                      in_channel=in_channels,
                      output_channels=output_channels,
                      backbone_name=backbone_name,
                      pretrained=pretrained,
                      freeze_backbone=freeze_backbone)

    # if you want use without backbone pretrained
    # base_model = UNet(False, cardinality=cardinality, n_blocks1=n_blocks1, n_blocks2=n_blocks2,
    #                  n_blocks3=n_blocks3, n_blocks4=n_blocks4,
    #                  output_channels=output_channels)

    model = ModelAdapter(base_model, in_channels)

    # loss and optimisation function definition
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Save model architecture
    model_structure = {
        "model_name": model.__class__.__name__,
        "model_parameters": str(model)
    }

    with open(model_path, 'w') as json_file:
        json.dump(model_structure, json_file, indent=4)


    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model summary
    summary(model, input_size=(2, 256, 256))

    # training
    print("start training...\n")
    history = train_model(model, criterion, optimizer, train_loader, valid_loader, EPOCHS, checkpoint_path,
                          VERBOSE=VERBOSE)
    plot_train(history)

    # test
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'], strict=False)
    print("\n start testing...\n")
    test_model(model, criterion, test_loader)


if __name__ == "__main__":
    config = Configuration(
        'C:/Users/Utente/Documents/GitHub/Reconstructing-BP-waves-from-iPPG-signals/scripts/python/config.cfg')

    train_models(config, extract_data=False)