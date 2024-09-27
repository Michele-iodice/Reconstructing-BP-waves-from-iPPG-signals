import torch
import torch.nn as nn
from resnetxt_blocks import create_resnext_network
from decoder_blocks import create_decoder_network

# Creazione del modello
model = create_resnext_network(cardinality=32, n_blocks1=3, n_blocks2=4, n_blocks3=23, n_blocks4=3)

# Input
x = torch.randn(1, 64, 64, 64)

# Esegui la forward pass
out1, out2, out3, out4 = model(x)


# utilizzo decoder
# Assuming encoder outputs are given as follows
encoder_outputs = [torch.rand(1, 1024, 16, 16),  # Output from encoder 1
                       torch.rand(1, 512, 32, 32),  # Output from encoder 2
                       torch.rand(1, 256, 64, 64),  # Output from encoder 3
                       torch.rand(1, 128, 128, 128)]  # Output from encoder 4

input_tensor = torch.rand(1, 2048, 8, 8)  # Input to the first decoder block
output_channels = [256, 128, 64, 32, 16]  # Desired output channels

# Create the network
decoder_network = create_decoder_network(encoder_outputs, input_channels=2048, output_channels_list=output_channels)

# Forward pass
output = decoder_network(input_tensor, encoder_outputs)
print("Output shape:", output.shape)  # Should be [1, 16, H, W] depending on the input size