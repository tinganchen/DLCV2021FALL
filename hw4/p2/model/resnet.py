import torchvision
import torch.nn as nn

def resnet_50(num_classes = 65, pretrained = False, **kwargs):
    model = torchvision.models.resnet50(pretrained=pretrained)
    model.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.Linear(1000, num_classes)
        )
    return model

