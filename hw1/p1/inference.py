import os

import utils.common as utils
from utils.options import args

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

from data import data

import pandas as pd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import time

import warnings
warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))


def main():

    start_epoch = 0
    best_acc = 0.0
    
    # Data loading
    print('=> Loading preprocessed data..')
    
    loader = data.DataLoading(args)
    loader.load()
    
    loader_test = loader.loader_test
    
    
    # Create model
    print('=> Building model...')
    model = import_module('model.models').__dict__[args.model](num_classes = args.num_classes).to(device)

    # Load pretrained weights
    print('=> Loading trained weights...')
    ckpt = torch.load(args.pretrain_dir + args.pretrain_file, map_location = device)
    state_dict = ckpt[list(ckpt.keys())[0]]
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    print('=> Start predicting...')
    s = time.time()
    
    if args.ground_truth == 'True':
        acc = test(args, loader_test, model, 0)
        print(f'Best @acc: {acc:.3f}\n')
    else:
         test(args, loader_test, model, 0)
    e = time.time()
    
    print(f'Finished in {e-s:.3f} seconds\n')
    print(f'Please check the prediction results in:\n"{args.output_file}"')



def test(args, loader_test, model, epoch):
    
    num_iterations = len(loader_test)
    
    preds = []
    data_files = []
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets, data_file) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)

            logits = model(inputs).to(device)

            _, pred = logits.topk(1, 1, True, True)
            
            preds.extend(list(pred.reshape(-1).cpu().detach().numpy()))
            data_files.extend(list(data_file))
    

    output = dict()
    output['image_id'] = data_files
    output['label'] = preds
    
    output = pd.DataFrame.from_dict(output)
    output.to_csv(args.output_file, index = False)
    
    
    if args.ground_truth == 'True':
        acc = 0.
        for i in range(len(data_files)):
            label = int(data_files[i].split('_')[0])
            pred = int(preds[i])
            if label == pred:
                acc += 1
        
        acc /= len(data_files) 
        acc *= 100.
        return acc
    
    return


if __name__ == '__main__':
    main()

