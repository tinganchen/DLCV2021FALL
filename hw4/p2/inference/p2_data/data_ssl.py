from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch.nn as nn
import random
import os
from PIL import Image
import json

filenameToPILImage = lambda x: Image.open(x)

with open('p2_data/class_mapping.json', 'r') as fp:
    label_proj = json.load(fp)


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)
    
    
class DataPreparation(Dataset):
    def __init__(self, data_dir, csv_path, stage):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.stage = stage

        self.transform1 = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
            ])
        
        self.transform2 = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            RandomApply(
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            RandomApply(
                transforms.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            transforms.RandomResizedCrop((128, 128)),
            transforms.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image1 = self.transform1(os.path.join(self.data_dir, path))
        image2 = self.transform2(os.path.join(self.data_dir, path))
        
        if self.stage == 'test':
            return image1, image2, index, path
        else:
            return image1, image2, label_proj[label]

    def __len__(self):
        return len(self.data_df)



class DataLoading:
    def __init__(self, args):
        
        self.args = args

    def load(self):
        if self.args.stage == 'train':
            trainset = DataPreparation(self.args.train_dataset, self.args.train_csv, self.args.stage)
            self.train_loader = DataLoader(dataset=trainset, 
                                           batch_size=self.args.train_batch_size, 
                                           shuffle = True,
                                           num_workers=2, pin_memory=True)
            
        if self.args.stage != 'test':
            valset = DataPreparation(self.args.val_dataset, self.args.val_csv, self.args.stage)
        else:
            valset = DataPreparation(self.args.test_dataset, self.args.test_csv, self.args.stage)
        self.val_loader = DataLoader(dataset=valset, 
                                     batch_size=self.args.eval_batch_size,
                                     shuffle = False,
                                     num_workers=2, pin_memory=True)
        
        
    
