import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models

def vgg(num_classes = 50, pretrained = True):
    model = models.vgg16(pretrained = pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def vgg2(num_classes = 50, pretrained = True):
    model = models.vgg19_bn(pretrained = pretrained)
    model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 1000),
            nn.Linear(1000, num_classes)
        )
    return model

def conv3x3(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)#, bias = False

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),
                    "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_layers, num_classes = 50, has_mask = None):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6
        
        if has_mask is None : has_mask = [1]*3*n

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace = True)

        self.layer1 = self._make_layer(block, 16, blocks=n, stride=1, has_mask=has_mask[0:n])
        self.layer2 = self._make_layer(block, 32, blocks=n, stride=2, has_mask=has_mask[n:2*n])
        self.layer3 = self._make_layer(block, 64, blocks=n, stride=2, has_mask=has_mask[2*n:3*n])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, has_mask):
        layers = []
        if has_mask[0] == 0 and (stride != 1 or self.inplanes != planes): 
            layers.append(LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0)))
        if not has_mask[0] == 0:
            layers.append(block(self.inplanes, planes, stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if not has_mask[i] == 0:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.avgpool(x)
        self.emb = x.view(x.size(0), -1)
        x = self.fc(self.emb)

        return x
    
def resnet_110(**kwargs):
    return ResNet(ResBasicBlock, 110, **kwargs)
