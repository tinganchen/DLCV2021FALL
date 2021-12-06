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
from p2_utils.options import args
from PIL import Image

class DataPreparation(Dataset):
    def __init__(self, root=args, data_path=None, label_path=None,
                 transform=None, target_transform=None):
        
        self.root = root
        self.data_path = data_path # data_path = '/home/ta/Documents/110-1/dlcv/hw2_TA/hw2_data/digits/mnistm/train'  
        self.label_path = label_path # label_path = '/home/ta/Documents/110-1/dlcv/hw2_TA/hw2_data/digits/mnistm/train.csv'
        
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
        
        ## generate noise vectors
        SEED = idx
        
        np.random.seed(SEED)
        noise_input = torch.Tensor(np.random.normal(0, 1, self.root.latent_dim))
        gen_label = torch.LongTensor(np.random.randint(0, args.n_classes, 1))

        candidate_imgs = self.file_labels['image_name'][self.file_labels['label'] == gen_label.item()]
        n_candidate = len(candidate_imgs)
        np.random.seed(SEED)
        chosen_id = np.random.choice(range(n_candidate), 1)
        chosen_file = np.array(candidate_imgs)[chosen_id][0]
        
        img_path = os.path.join(self.data_path, chosen_file)
        sample_image = Image.open(img_path) # plt.imread(img_path)
 
        if self.transform:
            sample_image = self.transform(sample_image)

        return image, label, data_file, noise_input, gen_label, sample_image
    
    def preprocess(self, data_path, label_path):
        self.data_files = os.listdir(data_path)
        self.data_files.sort()
  
        if label_path is not None:
            self.file_labels = pd.read_csv(label_path)

    


class NoiseDataPreparation(Dataset):
    def __init__(self, root=args, data_path=None, label_path=None,
                 transform=None, target_transform=None):
        
        self.root = root
        
    
        ## preprocess labels
        self.preprocess(self.root.num_noise_vectors, self.root.n_classes)
        
    def __len__(self):
        return self.root.num_noise_vectors

    def __getitem__(self, idx):
        SEED = idx
        
        np.random.seed(SEED)
        noise_input = torch.Tensor(np.random.normal(0, 1, self.root.latent_dim))
        gen_label = self.gen_labels[idx]
        data_file = self.file_list[idx]
      
        
        return noise_input, gen_label, data_file
    
    def preprocess(self, num_noise_vectors, n_classes):
        num_vectors_per_class = int(num_noise_vectors / n_classes)
        
        self.file_list = []
        self.gen_labels = []
        for i in range(n_classes):
            for j in range(num_vectors_per_class):
                file_name = f'{i}_{j+1:03d}.png'
                self.file_list.append(file_name)
                self.gen_labels.append(i)
        

        
        
class Data:
    def __init__(self, args, data_path, label_path):
        

        transform = transforms.Compose([
            transforms.Resize(args.img_size), 
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])
        
        if data_path is not None:
            train_dataset = DataPreparation(root=args,  
                                            data_path=data_path,
                                            label_path=label_path,
                                            transform=transform)
            
            self.loader_train = DataLoader(
                train_dataset, batch_size=args.train_batch_size, shuffle=True, 
                num_workers=2
                )
        
        
            
        test_dataset = NoiseDataPreparation(root=args)
        
        self.loader_test = DataLoader(
            test_dataset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=2
            )
       
