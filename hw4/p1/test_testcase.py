import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

import utils.common as utils
from utils.options import args

from importlib import import_module

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

device = torch.device(f"cuda:{args.gpus[0]}")

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def test(args, loader_test, model):
    
    # switch to eval mode
    model.eval()
 
    preds = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):

            inputs = inputs.to('cuda:0')
        
            p = args.shot * args.test_way
            data_shot, data_query = inputs[:p], inputs[p:]
            
            
            ## validate
            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)
    
            label = torch.arange(args.test_way).repeat(args.query) # re-label class from 0 to n_cls
            label = label.type(torch.cuda.LongTensor)
    
            logits = utils.distance_metric(model(data_query), proto, args.distance_metric)
            _, pred = logits.topk(1, 1, True, True)
            
            preds.append(list(pred.reshape(-1).cpu().detach().numpy()))
    
    
    return preds
        


if __name__=='__main__':

    test_dataset = MiniDataset(args.test_csv, args.test_dataset)

    test_loader = DataLoader(
        test_dataset, batch_size=args.test_way * (args.query + args.shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    model = import_module(f'model.{args.model}').__dict__['Convnet']().to(device)
    
    
    if args.pretrained == 'True':
        # Load pretrained weights
        ckpt = torch.load(args.pretrain_dir + args.pretrain_file, map_location = device)
        state_dict = ckpt[list(ckpt.keys())[0]]

        model.load_state_dict(state_dict)
        
        model = model.to(device)

    prediction_results = test(args, test_loader, model)

    # output your prediction to csv
    output = pd.read_csv('sample.csv')
    
    for i, pred in enumerate(prediction_results):
        output.loc[i][1:] = pred 
    
    output.to_csv(args.output_csv, index = False)
