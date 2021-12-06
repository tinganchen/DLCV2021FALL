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
    def __init__(self, root=args, data_path=None, label_path=None,
                 transform=None, target_transform=None):
        
        self.root = root
        self.data_path = data_path # data_path = '/home/ta/Documents/110-1/dlcv/hw2_TA/hw2_data/face/train'  

        self.transform = transform
        self.target_transform = target_transform
        
        ## preprocess files
        self.preprocess(self.data_path)
        

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        img_path = os.path.join(self.data_path, data_file)
        image = Image.open(img_path) # plt.imread(img_path)
 
        if self.transform:
            image = self.transform(image)
        
        return image, data_file
    
    def preprocess(self, data_path):
        self.data_files = os.listdir(data_path)
        self.data_files.sort()
  
    
class NoiseDataPreparation(Dataset):
    def __init__(self, root=args, data_path=None, label_path=None,
                 transform=None, target_transform=None):
        
        self.root = root
        
    def __len__(self):
        return self.root.num_noise_vectors

    def __getitem__(self, idx):
        SEED = idx
        np.random.seed(SEED)
        noise_input = torch.randn(self.root.input_dim, 1, 1)
        
        return noise_input
    


class Data:
    def __init__(self, args, data_path):
        

        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        
        train_dataset = DataPreparation(root=args,  
                                        data_path=os.path.join(data_path, 'train'),
                                        transform=transform)
        
        self.loader_train = DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, 
            num_workers=2
            )
        
        
        test_dataset = NoiseDataPreparation(root=args)
        
        self.loader_test = DataLoader(
            test_dataset, batch_size=args.train_batch_size, shuffle=False, 
            num_workers=2
            )