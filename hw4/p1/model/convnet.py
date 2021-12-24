import torch.nn as nn

from utils.options import args

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600
        
        if args.distance_metric not in ['euclidean', 'cosine_similarity']:
            self.fc = nn.Linear(1600, 300)

    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        if args.distance_metric not in ['euclidean', 'cosine_similarity']:
            x = self.fc(x)
        return x

