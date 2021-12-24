import torchvision
import torch.nn as nn
from collections import OrderedDict

org_model = torchvision.models.resnet50(pretrained=False)

class resnet(nn.Module):
    def __init__(self, num_classes = 65, pretrained = False):
        super(resnet, self).__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        self.conv1 = org_model.conv1
        self.bn1 = org_model.bn1
        self.relu = org_model.relu
        self.maxpool = org_model.maxpool
        self.layer1 = org_model.layer1
        self.layer2 = org_model.layer2
        self.layer3 = org_model.layer3
        self.layer4 = org_model.layer4
        self.avgpool = org_model.avgpool
        self.fc0 = nn.Linear(2048, 1000)
        self.fc1 = nn.Linear(1000, self.num_classes)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x) 
        
        self.feature = x
        
        x = x.reshape(x.shape[0], -1)
        
        x = self.fc0(x)
        x = self.fc1(x)
        
        return x

    
    
def resnet_50(num_classes = 65, pretrained = False):
    model = resnet()
    return model

