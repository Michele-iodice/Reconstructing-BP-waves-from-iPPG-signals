import torch
from model.unet_ippg_cwt import UNet, ModelAdapter
from config import Configuration
import numpy as np
import pandas as pd
from extraction.features_extraction import extract_feature_on_video, extract_feature_on_dataset
from model.utils import plotComparison, plotTest, plotSignal, split_data
from extraction.signal_to_cwt import plotCWT
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from extraction.signal_to_cwt import inverse_cwt


def predict_signals(config):
    # Parameter
    BATCH_SIZE = int(np.int32(config.uNetdict['BATCH_SIZE']))
    data_path = config.uNetdict['data_path']
    output_channels = config.get_array('output_channels')
    backbone_name = config.uNetdict['backbone_name']
    pretrained = config.get_boolean('UnetParameter', 'pretrained')
    freeze_backbone = config.get_boolean('UnetParameter', 'freeze_backbone')
    checkpoint_path = config.uNetdict['checkpoint_path']

    x_train, x_test, x_val, y_train, y_test, y_val = split_data(data_path)

    x_train_torch = torch.tensor(x_train).float()
    x_test_torch = torch.tensor(x_test).float()
    y_test_torch = torch.tensor(y_test).float()

    in_channels = x_train_torch.shape[1]
    base_model = UNet(True,
                      in_channel=in_channels,
                      output_channels=output_channels,
                      backbone_name=backbone_name,
                      pretrained=pretrained,
                      freeze_backbone=freeze_backbone)

    model = ModelAdapter(base_model, in_channels)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'], strict=False)

    test_dataset = TensorDataset(x_test_torch, y_test_torch)
    test_loader = DataLoader(test_dataset, shuffle=False)
    test_bar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for inputs, targets in test_bar:
            inputs = inputs.to(next(model.parameters()).device)
            targets = targets.to(next(model.parameters()).device)
            outputs = model(inputs)
            if torch.isnan(outputs).any():
                continue
            inputs = inputs.cpu().numpy()
            targets = targets.cpu().numpy()
            outputs = outputs.cpu().numpy()

            ippg = inverse_cwt(inputs[0], f_min=0.6, f_max=4.5)
            bp = inverse_cwt(targets[0], f_min=0.6, f_max=4.5)
            r_bp = inverse_cwt(outputs[0], f_min=0.6, f_max=4.5)

            plotSignal(ippg)
            plotComparison(bp, r_bp)
            plotCWT(inputs, 100, "iPPG")
            plotCWT(targets, 100, "BP")
            plotCWT(outputs, 100, "BP reconstructed")



def predict_video(config, videoFileName, bp ):
    # Parameter
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

    data = extract_feature_on_video(videoFileName, bp, config)
    x_prediction = np.array(data['CWT'])
    y_prediction = np.array(data['CWT_BP'])
    original_signal = np.array(data['original'])
    bp_gt = np.array(data['BP'])
    in_channels = x_prediction.shape[-1]
    base_model = UNet(cardinality=cardinality, n_blocks1=n_blocks1, n_blocks2=n_blocks2,
                      n_blocks3=n_blocks3, n_blocks4=n_blocks4,
                      output_channels=output_channels, backbone_name=backbone_name,
                      pretrained=pretrained, freeze_backbone=freeze_backbone)
    model = ModelAdapter(base_model, in_channels)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    x_prediction_tensor = torch.tensor(x_prediction, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(x_prediction_tensor)

    predictions_np = predictions.cpu().numpy()
    plotCWT(predictions_np)
    plotCWT(y_prediction)

    plotComparison(bp_gt, predictions_np)
    plotTest(original_signal, bp_gt, predictions_np)
    print("Predictions:", predictions)


def predict_dataset(dataset, save_results):
    # Parameter
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
    base_model = UNet(cardinality=cardinality, n_blocks1=n_blocks1, n_blocks2=n_blocks2,
                      n_blocks3=n_blocks3, n_blocks4=n_blocks4,
                      output_channels=output_channels, backbone_name=backbone_name,
                      pretrained=pretrained, freeze_backbone=freeze_backbone)
    checkpoint = torch.load(checkpoint_path)
    df = pd.DataFrame(columns=['prediction', 'BP', 'original', 'subject_id'])
    for idx in range(0, len(dataset)):
        data = extract_feature_on_video(dataset[idx], bp, config)
        data.to_csv("../dataset/prediction_data.csv", index=False)
        x_prediction = np.array(data['CWT'])
        original_signal = np.array(data['original'])
        bp_gt = np.array(data['BP'])
        in_channels = x_prediction.shape[-1]
        model = ModelAdapter(base_model, in_channels)
        model.load_state_dict(checkpoint['model_state_dict'])
        x_prediction_tensor = torch.tensor(x_prediction, dtype=torch.float32)

        with torch.no_grad():
            predictions = model(x_prediction_tensor)

        predictions_np = predictions.cpu().numpy()
        newLine = pd.DataFrame({'prediction': predictions_np, 'BP': bp_gt,
                                'original': original_signal, 'subject_id': data['subject_id']}, index=[0])
        df = pd.concat([df, newLine], ignore_index=True)

    df.to_csv(save_results, index=False)



if __name__ == "__main__":
    config = Configuration(
        'C:/Users/Utente/Documents/GitHub/Reconstructing-BP-waves-from-iPPG-signals/scripts/python/config.cfg')
    videoFileName = "D:/datasetBP4D+/F001/T7/vid.avi"
    bp = "D:/datasetBP4D+/F001/T7/BP_mmHg.txt"
    save_results = "../dataset/results_prediction_data.csv"
    #predict_video(config, videoFileName, bp)
    predict_signals(config)