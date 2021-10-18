from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
from torchvision.io import read_image
import os
from utils.options import args
from PIL import Image


class DataPreparation(Dataset):
    def __init__(self, root=args, dataset = None,
                 transform=None, target_transform=None):
        
        self.root = root
        self.test_only = self.root.test_only
        
        self.data_path = dataset
        self.data_files = os.listdir(self.data_path)
        self.data_files.sort()
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        img_path = os.path.join(self.data_path, data_file)
        image = Image.open(img_path) # plt.imread(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        if self.root.test_only == 'True':
            return image, int(data_file.split('_')[0]) , data_file
        
        label = int(data_file.split('_')[0]) 
   
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label, data_file
        


class DataLoading:
    def __init__(self, args):
        
        self.args = args
        
    def load(self):
        if args.test_only == 'False':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    
            trainset = DataPreparation(root=args,  
                                       dataset=self.args.train_dataset,
                                       transform=transform_train)
            
            self.loader_train = DataLoader(
                        trainset, batch_size=args.train_batch_size, shuffle=True, 
                        num_workers=2
                        )
            

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = DataPreparation(root=args, 
                                  dataset=args.test_dataset,
                                  transform=transform_test)
        
        self.loader_test = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=2
            )


    