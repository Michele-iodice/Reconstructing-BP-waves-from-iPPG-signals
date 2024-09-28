import torch
import torch.nn as nn


class ResNeXtBlock(nn.Module):
    """
    ResNeXt block definition with residual sum.
    :return: (torch.Tensor) the block as output tensor
    """

    def __init__(self, in_channels, cardinality, output_channels):
        super(ResNeXtBlock, self).__init__()

        self.cardinality = cardinality
        self.group_width = in_channels // cardinality

        self.convs1 = nn.ModuleList([
            nn.Conv2d(in_channels, self.group_width, kernel_size=(1, 1), bias=False) for _ in range(cardinality)
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv2d(self.group_width, 4, kernel_size=(3, 3), padding=1, bias=False) for _ in range(cardinality)
        ])

        self.final_conv = nn.Conv2d(cardinality * 4, output_channels, kernel_size=(1, 1), bias=False)

        self.input_conv = nn.Conv2d(in_channels, output_channels, kernel_size=(1, 1), bias=False)

    def forward(self, x):

        identity = x

        path_outputs = []

        for conv1, conv2 in zip(self.convs1, self.convs2):
            out = conv1(x)  # Convolution 1x1
            out = nn.ReLU()(out)
            out = conv2(out)  # Convolution 3x3
            out = nn.ReLU()(out)
            path_outputs.append(out)

        out = torch.cat(path_outputs, dim=1)

        out = self.final_conv(out)

        identity = self.input_conv(identity)

        out = out + identity

        return out


class ResNeXtGroup(nn.Module):
    """
    ResNeXtBlock group definition.
    Manage the block number inside a group and the downsampling operation.
    :return: (torch.Tensor) the ResNeXtGroup as an output tensor
    """

    def __init__(self, in_channels, cardinality, output_channels, num_blocks, downsample=False):
        super(ResNeXtGroup, self).__init__()

        self.blocks = nn.Sequential(
            *[ResNeXtBlock(in_channels if i == 0 else output_channels, cardinality, output_channels) for i in
              range(num_blocks)]
        )

        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.blocks(x)

        if self.downsample:
            out = self.pool(out)

        return out


class CustomResNeXtNetwork(nn.Module):
    """
    Custom ResNeXt network definition with more ResNeXtGroup of different size.
    :return: (torch.Tensor) Return the final output of each group
    """

    def __init__(self, in_channels, cardinality=32, n_blocks1=3, n_blocks2=4, n_blocks3=23, n_blocks4=3):
        """
        :param input_channels: channels of the input tensor
        :param cardinality: number of path of the ResnetBlock
        :param n_blocks1: number of blocks of the first group
        :param n_blocks2: number of blocks of the second group
        :param n_blocks3: number of blocks of the third group
        :param n_blocks4: number of blocks of the fourth group
        """
        super(CustomResNeXtNetwork, self).__init__()

        self.group1 = ResNeXtGroup(in_channels=in_channels, cardinality=cardinality, output_channels=256, num_blocks=n_blocks1)

        self.group2 = ResNeXtGroup(in_channels=256, cardinality=cardinality, output_channels=512, num_blocks=n_blocks2,
                                   downsample=True)

        self.group3 = ResNeXtGroup(in_channels=512, cardinality=cardinality, output_channels=1024, num_blocks=n_blocks3,
                                   downsample=True)

        self.group4 = ResNeXtGroup(in_channels=1024, cardinality=cardinality, output_channels=2048, num_blocks=n_blocks4,
                                   downsample=True)

    def forward(self, x):
        out1 = self.group1(x)

        out2 = self.group2(out1)

        out3 = self.group3(out2)

        out4 = self.group4(out3)

        return out1, out2, out3, out4


def create_resnext_network(input_channels, cardinality=32, n_blocks1=3, n_blocks2=4, n_blocks3=23, n_blocks4=3):
    """
    Creation of a ResNeXt Network.
    :param input_channels: channels of the input tensor
    :param cardinality: number of path of the ResnetBlock
    :param n_blocks1: number of blocks of the first group
    :param n_blocks2: number of blocks of the second group
    :param n_blocks3: number of blocks of the third group
    :param n_blocks4: number of blocks of the fourth group
    :return: (torch.Tensor) Return the final output of each group
    """
    return CustomResNeXtNetwork(in_channels=input_channels, cardinality=cardinality, n_blocks1=n_blocks1, n_blocks2=n_blocks2,
                                n_blocks3=n_blocks3, n_blocks4=n_blocks4)


# Example of using library
if __name__ == "__main__":
    x = torch.randn(1, 64, 64, 64)
    in_channels = x.shape[1]

    model = create_resnext_network(input_channels=in_channels, cardinality=32)

    out1, out2, out3, out4 = model(x)

    print(f'Output gruppo 1: {out1.shape}')
    print(f'Output gruppo 2: {out2.shape}')
    print(f'Output gruppo 3: {out3.shape}')
    print(f'Output gruppo 4: {out4.shape}')


