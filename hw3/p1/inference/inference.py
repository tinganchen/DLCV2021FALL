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

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))


def main():

    # Data loading
    print('=> Loading preprocessed data..')
    
    loader = data.DataLoading(args)
    loader.load()

    loader_test = loader.loader_test
    
    # Create model
    
    print('=> Building model...')

    model = torch.load(args.pretrain_arch)
    
   
    if args.pretrained == 'True':
        # Load pretrained weights
        ckpt = torch.load(args.pretrain_dir + args.pretrain_file, map_location = device)
        state_dict = ckpt[list(ckpt.keys())[0]]

        new_state_dict = dict()
            
        for k, v in model.state_dict().items():
            new_state_dict[k] = state_dict[k]

        model.load_state_dict(new_state_dict)
        
        model = model.to(device)
   

   
    inference(args, loader_test, model)
    
    print('Finish inference.')


           
            
def inference(args, loader_test, model):
    
    
    # switch to eval mode
    preds = []
    data_files = []
    #gt = []
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets, data_file) in enumerate(loader_test, 1):
            
            inputs = inputs.to(device)

            logits = model(inputs).to(device)

            _, pred = logits.topk(1, 1, True, True)
            
            preds.extend(list(pred.reshape(-1).cpu().detach().numpy()))
            data_files.extend(list(data_file))
            #gt.extend(list(targets.reshape(-1).cpu().detach().numpy()))

    output = dict()
    output['filename'] = data_files
    output['label'] = preds
    
    output = pd.DataFrame.from_dict(output)
    output.to_csv(args.output_file, index = False)
    
    
    #output['label'] = gt
    
    #output = pd.DataFrame.from_dict(output)
    #output.to_csv('gt.csv', index = False)       

    return 
        


if __name__ == '__main__':
    main()

