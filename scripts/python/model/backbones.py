import torch
import torch.nn as nn
import torchvision.models as models


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

        self.backbone = self._load_backbone(backbone_name, pretrained)

        self.out_features = self._remove_classifier(backbone_name)

    def _load_backbone(self, backbone_name, pretrained):
        if hasattr(models, backbone_name):
            backbone = getattr(models, backbone_name)(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone '{backbone_name}' non supportato.")
        return backbone

    def _remove_classifier(self, backbone_name):
        if backbone_name.startswith('resnet') or backbone_name.startswith('efficientnet'):
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
        features = self.backbone(x)
        return features

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


# Example of using library
if __name__ == '__main__':

    num_classes = 100
    model = Backbones(backbone_name='resnext101_32x8d', pretrained=True, freeze_backbone=True)

    output_features = model.get_output_features()

    classifier = nn.Sequential(
        nn.Linear(output_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    input_tensor = torch.randn(1, 3, 224, 224)  # Batch of image with dimension 224x224 with 3 channel

    features = model(input_tensor)

    output = classifier(features)
    print(output.shape)