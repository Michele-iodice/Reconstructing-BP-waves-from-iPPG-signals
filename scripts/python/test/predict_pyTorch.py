import torch
from model.unet_ippg_cwt import UNet, ModelAdapter
from config import Configuration
import numpy as np
import pandas as pd
from extraction.feature_extraction import extract_feature_on_video
from model.utils import plotComparison, plotTest
from extraction.signal_to_cwt import plotCWT


def predict_video(config, videoFileName, bp ):
    # Parameter
    cardinality = np.int32(config.uNetdict['cardinality'])
    n_blocks1 = np.int32(config.uNetdict['n_blocks1'])
    n_blocks2 = np.int32(config.uNetdict['n_blocks2'])
    n_blocks3 = np.int32(config.uNetdict['n_blocks3'])
    n_blocks4 = np.int32(config.uNetdict['n_blocks4'])
    output_channels = config.get_array('output_channels')
    backbone_name = config.uNetdict['backbone_name']
    pretrained = config.get_boolean('uNetdict', 'pretrained')
    freeze_backbone = config.get_boolean('uNetdict', 'freeze_backbone')
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

    print("Output shape:", model.shape)  # Should be [1, 2, 256, 256]

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
    pretrained = config.get_boolean('uNetdict', 'pretrained')
    freeze_backbone = config.get_boolean('uNetdict', 'freeze_backbone')
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
        'C:/Users/39392/Documents/GitHub/Reconstructing-BP-waves-from-iPPG-signals/scripts/python/config.cfg')
    videoFileName = "C:/Users/39392/Desktop/University/MAGISTRALE/NaturalInteraction/progetto/datasetBP4D+/F001/T7/vid.avi"
    bp = "C:/Users/39392/Desktop/University/MAGISTRALE/NaturalInteraction/progetto/datasetBP4D+/F001/T7/BP_mmHg.txt"
    save_results = "../dataset/results_prediction_data.csv"
    predict_video(config, videoFileName, bp)