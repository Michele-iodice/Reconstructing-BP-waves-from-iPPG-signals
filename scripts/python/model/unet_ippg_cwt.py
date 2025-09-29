import torch.nn as nn
from model.resnetxt_blocks import create_resnext_network, x
from model.decoder_blocks import create_decoder_network
from model.backbones import Backbones
import torch

class UNet(nn.Module):
    def __init__(self,backbone: bool, out_channel, cardinality=None, n_blocks1=None, n_blocks2=None, n_blocks3=None, n_blocks4=None,
                 output_channels=None, backbone_name=None, pretrained=True, freeze_backbone=False):
        super(UNet, self).__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        if backbone:
            self.backbone = Backbones(backbone_name=backbone_name, pretrained=pretrained, freeze_backbone=freeze_backbone)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)

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


        self.output_channels = output_channels
        # Set Decoder network
        dummy_input = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            _ = self.backbone(dummy_input)
            encoder_outputs = self.backbone.get_encoder_outputs()

        self.decoder_blocks = create_decoder_network(
            encoder_channel_list=[e.shape[1] for e in encoder_outputs],
            output_channels_list=self.output_channels
        )

        # Final convolution layer
        self.final_conv = nn.Conv2d(self.output_channels[4], out_channel, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        if self.backbone:
            x = self.backbone(x)
            encoder_outputs = self.backbone.get_encoder_outputs()
        else:
            x = self.conv1(x)
            self.resnet_blocks.set_out_conv(x)
            x = self.max_pool(x)
            # Pass through ResNeXt blocks
            x = self.resnet_blocks(x)
            encoder_outputs = self.resnet_blocks.get_skips()

        # Pass through Decoder blocks
        decoder_output = self.decoder_blocks(x,encoder_outputs)

        x = self.final_conv(decoder_output)
        return x


class ModelAdapter(nn.Module):
    def __init__(self, base_model, in_channels, out_channels=3):
        super(ModelAdapter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.base_model = base_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.base_model(x)
        return x


# Example of how to instantiate and use the UNetNetwork
if __name__ == "__main__":
    output= UNet(cardinality=32, n_blocks1=3, n_blocks2=4, n_blocks3=23, n_blocks4=3,
                 output_channels=[256, 128, 64, 32, 16], backbone_name='resnext101_32x8d',
                 pretrained=True, freeze_backbone=True)
    print("model UNet correctly build")
