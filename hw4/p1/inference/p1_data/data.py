from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
from torchvision.io import read_image
import os
from p1_utils.options import args
from PIL import Image
from p1_data.mini_imagenet import *
from p1_data.samplers import *

class DataLoading:
    def __init__(self, args):
        
        self.args = args
        
    def load(self):
        if self.args.stage == 'train':
            trainset = MiniImageNet(self.args.train_dataset, self.args.train_csv)
            train_sampler = CategoriesSampler(trainset.label, 100,
                                              self.args.train_way, self.args.shot + self.args.query)
            self.train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                           num_workers=8, pin_memory=True)
            
        if self.args.stage != 'test':
            valset = MiniImageNet(self.args.val_dataset, self.args.val_csv)
        else:
            valset = MiniImageNet(self.args.test_dataset, self.args.test_csv)
        val_sampler = CategoriesSampler(valset.label, 400,
                                        self.args.test_way, self.args.shot + self.args.query)
        self.val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                     num_workers=8, pin_memory=True)
        
        
    
