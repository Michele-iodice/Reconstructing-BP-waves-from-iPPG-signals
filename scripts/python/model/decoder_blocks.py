import torch
import torch.nn as nn

# -----------------------------------------------------------------
#                        Decoder Block
# -----------------------------------------------------------------


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, encoder_channels=None, use_concat=True):
        """
        This class is used to compose a Decoder Block
        :param input_channels: number of channels in input
        :param output_channels: number of channels in output
        :param encoder_channels: number of channels of the encoder if it is attached
        :param use_concat: if it is true, the encoder output is concatenate to decoder block
        """
        super(DecoderBlock, self).__init__()

        self.use_concat = use_concat

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.use_concat and encoder_channels is not None:
            conv_input_channels = input_channels + encoder_channels
        else:
            conv_input_channels = input_channels

        self.conv1 = nn.Conv2d(conv_input_channels, output_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x, encoder_output=None):
        """
        :param x: decoder input
        :param encoder_output: optional encoder output to concatenate
        :return: decoder output
        """
        x = self.upsample(x)

        if self.use_concat and encoder_output is not None:
            x = torch.cat([x, encoder_output], dim=1)  # Concatena lungo la dimensione dei canali

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x


# ---------------------------------------------------------------------------
#                              Decoder Network
# ---------------------------------------------------------------------------


class DecoderNetwork(nn.Module):
    def __init__(self, encoder_channels_list, output_channels_list):
        """
        Create the Decoder network of five :class:'DecoderBlock'
        :param output_channels_list: an array with the output channels of each decoder block
        """
        super(DecoderNetwork, self).__init__()

        # Validate the input
        assert len(output_channels_list) == 5, "There must be 5 output channels specified."
        assert len(encoder_channels_list) == 5, "There must be 5 encoder outputs specified."

        self.input_channels = None
        self.output_channels_list = output_channels_list

        # Create the decoder blocks

        self.input_channels = encoder_channels_list[4]
        output_channels_list = self.output_channels_list

        self.decoder1 = DecoderBlock(self.input_channels, output_channels_list[0],
                                     encoder_channels=encoder_channels_list[3])
        self.decoder2 = DecoderBlock(output_channels_list[0], output_channels_list[1],
                                     encoder_channels=encoder_channels_list[2])
        self.decoder3 = DecoderBlock(output_channels_list[1], output_channels_list[2],
                                     encoder_channels=encoder_channels_list[1])
        self.decoder4 = DecoderBlock(output_channels_list[2], output_channels_list[3],
                                     encoder_channels=encoder_channels_list[0])

        self.decoder5 = DecoderBlock(self.output_channels_list[3], self.output_channels_list[4], use_concat=False)

    def forward(self, x, encoder_outputs):
        """
        :param x: (torch.Tensor) decoder Input (es: 8x8x2048).
        :param encoder_outputs: (list of torch.Tensor) List of outputs from encoder blocks to concatenate.
        :return: (torch.Tensor) decoder network final output
        """

        x = self.decoder1(x, encoder_outputs[3])
        x = self.decoder2(x, encoder_outputs[2])
        x = self.decoder3(x, encoder_outputs[1])
        x = self.decoder4(x, encoder_outputs[0])
        x = self.decoder5(x)

        return x

    def set_decoder_input(self,input_channels, encoders_outputs):
        """
         Sets the decoder input with the encoder output
         :param input_channels: input channels of the first decoder block
         :param encoders_outputs: (List of torch.Tensor) the encoders output to concatenate with decoders block
        """
        assert len(encoders_outputs) == 5, "There must be 5 encoder outputs specified."

        self.input_channels = input_channels
        output_channels_list = self.output_channels_list

        self.decoder1 = DecoderBlock(input_channels, output_channels_list[0],
                                     encoder_channels=encoders_outputs[3].shape[1])
        self.decoder2 = DecoderBlock(output_channels_list[0], output_channels_list[1],
                                     encoder_channels=encoders_outputs[2].shape[1])
        self.decoder3 = DecoderBlock(output_channels_list[1], output_channels_list[2],
                                     encoder_channels=encoders_outputs[1].shape[1])
        self.decoder4 = DecoderBlock(output_channels_list[2], output_channels_list[3],
                                     encoder_channels=encoders_outputs[0].shape[1])


def create_decoder_network(encoder_channel_list, output_channels_list):
    """
    Creation of a decoder network
    :param encoder_channel_list: an array with the input channels of each decoder block
    :param output_channels_list: an array with the output channels of each decoder block
    :return: (torch.Tensor) decoder network final output
    """
    return DecoderNetwork(encoder_channels_list=encoder_channel_list,
                          output_channels_list=output_channels_list)


# Example of how to instantiate and use the DecoderNetwork
if __name__ == "__main__":
    # Assuming encoder outputs are given as follows
    encoder_outputs = [torch.rand(1, 1024, 16, 16),  # Output from encoder 1
                       torch.rand(1, 512, 32, 32),  # Output from encoder 2
                       torch.rand(1, 256, 64, 64),  # Output from encoder 3
                       torch.rand(1, 128, 128, 128)]  # Output from encoder 4

    input_tensor = torch.rand(1, 2048, 8, 8)  # Input to the first decoder block
    output_channels = [256, 128, 64, 32, 16]  # Desired output channels

    # Create the network
    decoder_network = create_decoder_network(output_channels_list=output_channels)

    # Forward pass
    decoder_network.set_decoder_input(input_tensor, encoder_outputs)
    output = decoder_network(input_tensor, encoder_outputs)
    print("Output shape:", output.shape)  # Should be [1, 16, H, W] depending on the input size
