import os

import p2_utils.common as utils
from p2_utils.options import args

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

import torch.nn.functional as F

from p2_data import data_ssl

import pandas as pd
import json

import warnings
warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))


with open('p2_data/class_mapping.json', 'r') as fp:
    label_proj = json.load(fp)

labels = list(label_proj.keys())

def main():

    # Data loading
    print('=> Loading preprocessed data..')
    
    loader = data_ssl.DataLoading(args)
    loader.load()
    
    loader_test = loader.val_loader

    
    # Create model
    
    print('=> Building model...')
    model_target = import_module(f'p2_model.{args.model}').__dict__['resnet_50']().to(device)
    
    if args.pretrained == 'True':
        # Load pretrained weights
        state_dict = torch.load(args.pretrain_dir + args.pretrain_file, 
                                map_location = device)['state_dict_target']
        
        model_target.load_state_dict(state_dict)
        
        model_target = model_target.to(device)
   
    print('=> Start inference...')
    test(args, loader_test, model_target)

    print('Finish inference.')    

def test(args, loader_test, model, epoch = 0):

    # switch to eval mode
    model.eval()
    
    num_iterations = len(loader_test)
    
    ids = []
    data_files = []
    preds = []

    with torch.no_grad():
        for i, (inputs, _, index, path) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)

            logits = model(inputs)
            
            _, pred = logits.topk(1, 1, True, True)
            
            ids.extend(list(index.reshape(-1).cpu().detach().numpy()))
            data_files.extend(list(path))
            
            prediction = list(pred.reshape(-1).cpu().detach().numpy())
            
            preds.extend([labels[int(pred)] for pred in prediction])
            

    output = dict()
    output['id'] = ids
    output['filename'] = data_files
    output['label'] = preds
    
    output = pd.DataFrame.from_dict(output)
    output.to_csv(args.output_csv, index = False)
    


if __name__ == '__main__':
    main()

