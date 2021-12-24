from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
from torchvision.io import read_image
import os
from PIL import Image
import json

filenameToPILImage = lambda x: Image.open(x)

with open('data/class_mapping.json', 'r') as fp:
    label_proj = json.load(fp)

class DataPreparation(Dataset):
    def __init__(self, data_dir, csv_path):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label_proj[label]

    def __len__(self):
        return len(self.data_df)

class DataLoading:
    def __init__(self, args):
        
        self.args = args

    def load(self):
        if self.args.stage == 'train':
            trainset = DataPreparation(self.args.train_dataset, self.args.train_csv)
            self.train_loader = DataLoader(dataset=trainset, 
                                           batch_size=self.args.train_batch_size, 
                                           shuffle = True,
                                           num_workers=8, pin_memory=True)
            
        if self.args.stage != 'test':
            valset = DataPreparation(self.args.val_dataset, self.args.val_csv)
        else:
            valset = DataPreparation(self.args.test_dataset, self.args.test_csv)
        self.val_loader = DataLoader(dataset=valset, 
                                     batch_size=self.args.eval_batch_size,
                                     shuffle = False,
                                     num_workers=8, pin_memory=True)
        
        
    