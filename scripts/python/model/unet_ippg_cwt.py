import torch.nn as nn
from resnetxt_blocks import create_resnext_network
from decoder_blocks import create_decoder_network
from backbones import Backbones


class UNet(nn.Module):
    def __init__(self, cardinality, n_blocks1, n_blocks2, n_blocks3, n_blocks4,
                 output_channels, backbone_name, pretrained=True, freeze_backbone=True):
        super(UNet, self).__init__()

        self.backbone = Backbones(backbone_name=backbone_name, pretrained=pretrained, freeze_backbone=freeze_backbone)

        in_conv = self.backbone.get_output_features()
        self.conv1 = nn.Conv2d(in_conv, 64, kernel_size=(3, 3), padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Create ResNeXt blocks
        self.resnet_blocks = create_resnext_network(
            input_channels=64,  # Fixed input channels from conv1
            cardinality=cardinality,
            n_blocks1=n_blocks1,
            n_blocks2=n_blocks2,
            n_blocks3=n_blocks3,
            n_blocks4=n_blocks4
        )

        # Create Decoder network
        self.decoder_blocks = create_decoder_network(
            encoder_outputs=None,
            input_channels=None,
            output_channels_list=output_channels
        )

        # Final convolution layer
        self.final_conv = nn.Conv2d(2, 2, kernel_size=(3, 3), padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)

        x = self.conv1(x)

        x = self.max_pool(x)

        # Pass through ResNeXt blocks
        encoder_outputs = self.resnet_blocks(x)

        # Set decoder inputs
        decoder_input_channels = encoder_outputs[3].shape[1]
        self.decoder_blocks.set_encoder_outputs(encoder_outputs)
        self.decoder_blocks.set_input_channels(decoder_input_channels)

        # Pass through Decoder blocks
        decoder_output = self.decoder_blocks(encoder_outputs[3])

        x = self.final_conv(decoder_output)
        return self.sigmoid(x)


class ModelAdapter(nn.Module):
    def __init__(self, base_model, in_channels, out_channels=3):
        super(ModelAdapter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.base_model = base_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.base_model(x)
        return x


# Example of how to instantiate and use the DecoderNetwork
if __name__ == "__main__":
    output= UNet(cardinality=32, n_blocks1=3, n_blocks2=4, n_blocks3=23, n_blocks4=3,
                 output_channels=[256, 128, 64, 32, 16], backbone_name='resnext101_32x8d',
                 pretrained=True, freeze_backbone=True)

    print("Output shape:", output.shape) # Should be [1, 2, 256, 256]