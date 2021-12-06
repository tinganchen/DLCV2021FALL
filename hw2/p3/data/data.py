from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader


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
    def __init__(self, root=args, data_path=None, label_path=None,
                 transform=None, target_transform=None):
        
        self.root = root
        self.data_path = data_path # data_path = '/home/ta/Documents/110-1/dlcv/hw2_TA/hw2_data/digits/usps/train'  
        self.label_path = label_path # label_path = '/home/ta/Documents/110-1/dlcv/hw2_TA/hw2_data/digits/usps/train.csv'
        
        self.transform = transform
        self.target_transform = target_transform
        
        ## preprocess files
        self.preprocess(self.data_path, self.label_path)
        

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        img_path = os.path.join(self.data_path, data_file)
        image = Image.open(img_path) # plt.imread(img_path)
 
        if self.transform:
            image = self.transform(image)
        
        if self.label_path is None:
            return image, -1, data_file
        
        label = self.file_labels['label'][self.file_labels['image_name'] == data_file].iloc[0]
            
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, data_file
    
    def preprocess(self, data_path, label_path):
        self.data_files = os.listdir(data_path)
        self.data_files.sort()
  
        if label_path is not None:
            self.file_labels = pd.read_csv(label_path)
        

class Data:
    def __init__(self, args, data_path, label_path):
        

        transform = transforms.Compose([
            transforms.Resize((28, 28)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])
        
        train_dataset = DataPreparation(root=args,  
                                        data_path=data_path,
                                        label_path=label_path,
                                        transform=transform)
        
        self.loader_train = DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, 
            num_workers=2
            )
        
        
        test_data_path = data_path.replace('train', 'test')
        test_label_path = label_path
        
        if label_path is not None:
            test_label_path = label_path.replace('train', 'test')
        
        test_dataset = DataPreparation(root=args,  
                                       data_path=test_data_path,
                                       label_path=test_label_path,
                                       transform=transform)
        
        self.loader_test = DataLoader(
            test_dataset, batch_size=args.train_batch_size, shuffle=False, 
            num_workers=2
            )
       
