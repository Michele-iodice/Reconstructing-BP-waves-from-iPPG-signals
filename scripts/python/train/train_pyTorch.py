import torch
import numpy as np
import torch.nn as nn
from model.unet_ippg_cwt import UNet

model=UNet(cardinality=32, n_blocks1=3, n_blocks2=4, n_blocks3=23, n_blocks4=3,
           output_channels=[256, 128, 64, 32, 16], backbone_name='resnext101_32x8d',
           pretrained=True, freeze_backbone=True)
data = np.expand_dims(data, axis=0)
inp = torch.tensor(data, dtype=torch.float32)
l1 = nn.Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
out = model(l1)
model = model(inp, out, name="UNet")

print("Output shape:", model.shape)  # Should be [1, 2, 256, 256]