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
import glob


dataset = '/home/ta/Documents/110-1/dlcv/hw1_data/p2_data/train'

class DataPreparation(Dataset):
    def __init__(self, root = args, dataset = None,
                 transform = None, target_transform = None):
        
        self.root = root
        self.test_only = self.root.test_only
        
        self.data_path = dataset
        self.data_files = glob.glob(os.path.join(self.data_path, '*sat*'))
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
        
        if args.test_only:
            return image, data_file.split('/')[-1]
        
        label_path = img_path.replace('jpg', 'png').replace('sat', 'mask')
        label_image = Image.open(label_path)
        
        if self.target_transform:
            mask = self.target_transform(label_image)

        masks = torch.zeros([512, 512])
        maski = 4 * mask[0, :, :] + 2 * mask[1, :, :] + mask[2, :, :]
        masks[maski == 3] = 0  # (Cyan: 011) Urban land 
        masks[maski == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[maski == 5] = 2  # (Purple: 101) Rangeland 
        masks[maski == 2] = 3  # (Green: 010) Forest land 
        masks[maski == 1] = 4  # (Blue: 001) Water 
        masks[maski == 7] = 5  # (White: 111) Barren land 
        masks[maski == 0] = 6  # (Black: 000) Unknown 
        
        labels = torch.zeros([self.root.num_classes, 512, 512])
        
        for i in range(7):
            labels[i, masks == i] = 1.

        return image, labels.type(torch.LongTensor), data_file.split('/')[-1] # image: [3, 512, 512] (0, 1), masks: [512, 512] (0, 1, 2, ..., 6)
        


class DataLoading:
    def __init__(self, args):
        
        self.args = args
        
    def load(self):
        
        target_transform = transforms.Compose([
            #transforms.RandomCrop(512),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_input = transforms.Compose([
            #transforms.RandomCrop(512),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
            
        if args.test_only == 'False':
            trainset = DataPreparation(root=args,  
                                       dataset=self.args.train_dataset,
                                       transform=transform_input,
                                       target_transform=target_transform)
            
            self.loader_train = DataLoader(
                        trainset, batch_size=args.train_batch_size, shuffle=False, 
                        num_workers=2
                        )
   
        testset = DataPreparation(root=args, 
                                  dataset=args.test_dataset,
                                  transform=transform_input,
                                  target_transform=target_transform)
        
        self.loader_test = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=2
            )


    