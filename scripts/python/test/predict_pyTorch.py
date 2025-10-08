import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from model.unet_ippg_cwt import UNet, ModelAdapter
from model.utils import split_data, test_model
from config import Configuration
import numpy as np

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



if __name__ == "__main__":
    config = Configuration(
        'C:/Users/Utente/Documents/GitHub/Reconstructing-BP-waves-from-iPPG-signals/scripts/python/config.cfg')
    dataset_path = config.uNetdict['dataset_path']
    predict_dataset(config, dataset_path)