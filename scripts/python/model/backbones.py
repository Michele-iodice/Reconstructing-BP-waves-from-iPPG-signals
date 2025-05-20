import torch
import torch.nn as nn
import torchvision.models as models
import warnings


class BackboneFactory(nn.Module):
    """
    The class represents a ‘factory’ that creates several pre-trained backbones
    and removes the last classification layer to use it as a feature extractor.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True):
        """
        Initialisation of class
        :param backbone_name: name of the backbone to use.
        :param pretrained: if true, then the pre-trained network will use imagenet weights
        """
        super(BackboneFactory, self).__init__()
        self.backbone_name = backbone_name

        self.backbone = self._load_backbone(backbone_name, pretrained)

        self.out_features = self._remove_classifier(backbone_name)
        self.skips = []


    def setSkips(self, skips):

        self.skips.append(skips[0])
        self.skips.append(skips[1])
        self.skips.append(skips[2])
        self.skips.append(skips[3])
        self.skips.append(skips[4])

    def getSkips(self):
        return self.skips

    def _load_backbone(self, backbone_name, pretrained):

        if not hasattr(models, backbone_name):
            raise ValueError(f"Backbone '{backbone_name}' not supported.")

        # backbone_name Maps -> correct name enum weights (torchvision >=0.13)
        weights_map = {
            # ResNet family
            'resnet18': 'ResNet18_Weights',
            'resnet34': 'ResNet34_Weights',
            'resnet50': 'ResNet50_Weights',
            'resnet101': 'ResNet101_Weights',
            'resnet152': 'ResNet152_Weights',

            # ResNeXt family
            'resnext50_32x4d': 'ResNeXt50_32X4D_Weights',
            'resnext101_32x8d': 'ResNeXt101_32X8D_Weights',

            # Wide ResNet
            'wide_resnet50_2': 'Wide_ResNet50_2_Weights',
            'wide_resnet101_2': 'Wide_ResNet101_2_Weights',

            # MobileNet
            'mobilenet_v2': 'MobileNet_V2_Weights',
            'mobilenet_v3_large': 'MobileNet_V3_Large_Weights',
            'mobilenet_v3_small': 'MobileNet_V3_Small_Weights',

            # EfficientNet
            'efficientnet_b0': 'EfficientNet_B0_Weights',
            'efficientnet_b1': 'EfficientNet_B1_Weights',
            'efficientnet_b2': 'EfficientNet_B2_Weights',
            'efficientnet_b3': 'EfficientNet_B3_Weights',
            'efficientnet_b4': 'EfficientNet_B4_Weights',
            'efficientnet_b5': 'EfficientNet_B5_Weights',
            'efficientnet_b6': 'EfficientNet_B6_Weights',
            'efficientnet_b7': 'EfficientNet_B7_Weights',

            # DenseNet
            'densenet121': 'DenseNet121_Weights',
            'densenet161': 'DenseNet161_Weights',
            'densenet169': 'DenseNet169_Weights',
            'densenet201': 'DenseNet201_Weights',

            # VGG
            'vgg11': 'VGG11_Weights',
            'vgg11_bn': 'VGG11_BN_Weights',
            'vgg13': 'VGG13_Weights',
            'vgg13_bn': 'VGG13_BN_Weights',
            'vgg16': 'VGG16_Weights',
            'vgg16_bn': 'VGG16_BN_Weights',
            'vgg19': 'VGG19_Weights',
            'vgg19_bn': 'VGG19_BN_Weights',

            # ShuffleNet
            'shufflenet_v2_x0_5': 'ShuffleNet_V2_X0_5_Weights',
            'shufflenet_v2_x1_0': 'ShuffleNet_V2_X1_0_Weights',
            'shufflenet_v2_x1_5': 'ShuffleNet_V2_X1_5_Weights',
            'shufflenet_v2_x2_0': 'ShuffleNet_V2_X2_0_Weights',

            # SqueezeNet
            'squeezenet1_0': 'SqueezeNet1_0_Weights',
            'squeezenet1_1': 'SqueezeNet1_1_Weights',

            # AlexNet
            'alexnet': 'AlexNet_Weights',

            # RegNet
            'regnet_y_400mf': 'RegNet_Y_400MF_Weights',
            'regnet_y_800mf': 'RegNet_Y_800MF_Weights',
            'regnet_y_1_6gf': 'RegNet_Y_1_6GF_Weights',
            'regnet_y_3_2gf': 'RegNet_Y_3_2GF_Weights',
            'regnet_y_8gf': 'RegNet_Y_8GF_Weights',
            'regnet_y_16gf': 'RegNet_Y_16GF_Weights',
            'regnet_y_32gf': 'RegNet_Y_32GF_Weights',
            'regnet_x_400mf': 'RegNet_X_400MF_Weights',
            'regnet_x_800mf': 'RegNet_X_800MF_Weights',
            'regnet_x_1_6gf': 'RegNet_X_1_6GF_Weights',
            'regnet_x_3_2gf': 'RegNet_X_3_2GF_Weights',
            'regnet_x_8gf': 'RegNet_X_8GF_Weights',
            'regnet_x_16gf': 'RegNet_X_16GF_Weights',

            # Vision Transformer (ViT)
            'vit_b_16': 'ViT_B_16_Weights',
            'vit_b_32': 'ViT_B_32_Weights',
            'vit_l_16': 'ViT_L_16_Weights',
            'vit_l_32': 'ViT_L_32_Weights',
            # Aggiungi altri se usi modelli nuovi...

        }

        weight_class_name = weights_map.get(backbone_name)

        if weight_class_name is None:
            weights = None
            if pretrained:
                warnings.warn(
                    f"No weights enum defined fo '{backbone_name}', upload without pre-trained weights.")
        else:
            try:
                weights_enum = getattr(models, weight_class_name)
                weights = weights_enum.DEFAULT if pretrained else None
            except AttributeError:
                weights = None
                if pretrained:
                    warnings.warn(f"Weights Enum '{weight_class_name}' not found in torchvision.models.")

        try:
            backbone = getattr(models, backbone_name)(weights=weights)
        except TypeError:
            warnings.warn("'weights' parameter not supported, use fallback with 'pretrained'.")
            backbone = getattr(models, backbone_name)(pretrained=pretrained)

        return backbone


    def _remove_classifier(self, backbone_name):
        if backbone_name.startswith('resnet') or backbone_name.startswith('resnext') or backbone_name.startswith('efficientnet'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Rimuove il fully connected
        elif backbone_name.startswith('vgg'):
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()  # Rimuove il classifier VGG
        elif backbone_name.startswith('densenet'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()  # Rimuove il classifier DenseNet

        else:
            raise ValueError(f"Non è stata implementata la rimozione del classificatore per '{backbone_name}'")

        return in_features

    def forward(self, x):
        if self.backbone_name.startswith('resnet') or self.backbone_name.startswith('resnext') or self.backbone_name.startswith(
                'efficientnet'):
            x = self.backbone.conv1(x)
            print("conv1 shape: ", x.shape)
            out_conv1 = x
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            print("maxpool shape: ", x.shape)

            out1 = self.backbone.layer1(x)
            print("encoder 1 shape: ", out1.shape)
            out2 = self.backbone.layer2(out1)
            print("encoder 2 shape: ", out2.shape)
            out3 = self.backbone.layer3(out2)
            print("encoder 3 shape: ", out3.shape)
            out4 = self.backbone.layer4(out3)
            print("encoder 4 shape: ", out4.shape)
            self.setSkips([out_conv1, out1, out2, out3, out4])

            return out4

        else:
            x = self.backbone(x)

        return x

    def get_output_features(self):
        return self.out_features


class Backbones(nn.Module):
    """
    This class utilises the backbone created by :class:'BackboneFactory' and adds a new classifier,
    enabling Transfer Learning. It is possible to choose whether or not to freeze the backbone,
    making it flexible for use either as a static feature extractor or for full fine-tuning.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True, freeze_backbone=False):
        """
        Initialisation of class
        :param backbone_name: name of the backbone to use.
        :param pretrained: if true, then the pre-trained network will use imagenet weights
        :param freeze_backbone: set it to true if you wish to freeze weights
        """
        super(Backbones, self).__init__()

        self.backbone = BackboneFactory(backbone_name=backbone_name, pretrained=pretrained)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):

        features = self.backbone(x)
        return features

    def get_output_features(self):
        return self.backbone.get_output_features()

    def get_encoder_outputs(self):
        return self.backbone.getSkips()


# Example of using library
if __name__ == '__main__':

    num_classes = 100
    model = Backbones(backbone_name='resnext101_32x8d', pretrained=True, freeze_backbone=True)

    output_feature = model.get_output_features()

    classifier = nn.Sequential(
        nn.Linear(output_feature, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    input_tensor = torch.randn(1, 3, 224, 224)  # Batch of image with dimension 224x224 with 3 channel

    features = model(input_tensor)

    output = classifier(features)
    print(output.shape)