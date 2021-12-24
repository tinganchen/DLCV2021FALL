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
    
    loader_test = loader.val_loader
    
    '''
    loaded_data = next(iter(loader_test))
    len(loaded_data) # 2
    loaded_data[0].shape # [80, 3, 84, 84], 80 = 5*(1+15) = n_cls*(n_shot+n_query)
    loaded_data[1] # labels
    '''
    
    # Create model
    
    print('=> Building model...')
    model = import_module(f'model.{args.model}').__dict__['Convnet']().to(device)
    
    
    if args.pretrained == 'True':
        # Load pretrained weights
        ckpt = torch.load(args.pretrain_dir + args.pretrain_file, map_location = device)
        state_dict = ckpt[list(ckpt.keys())[0]]

        model.load_state_dict(state_dict)
        
        model = model.to(device)
   
    test_acc = validate(args, loader_test, model)
      
 
    print_logger.info(f"Best @test_acc: {test_acc:.3f}")

    
 
            
def validate(args, loader_test, model):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()
    
    
    # switch to eval mode
    model.eval()
    
    num_iterations = len(loader_test)
    
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):

            inputs = inputs.to(device)
        
            p = args.shot * args.test_way
            data_shot, data_query = inputs[:p], inputs[p:]
            
            
            ## validate
            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)
    
            label = torch.arange(args.test_way).repeat(args.query) # re-label class from 0 to n_cls
            label = label.type(torch.cuda.LongTensor)
    
            logits = utils.euclidean_metric(model(data_query), proto)
            error = cross_entropy(logits, label)
          
            losses.update(error.item(), inputs.size(0))
  
            prec1, prec5 = utils.accuracy(logits, label, topk = (1, 5))

            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
           
    print_logger.info('Test_acc {top1.avg:.3f}\n'
                 '====================================\n'
                 .format(top1 = top1))
    
    
    return top1.avg
        


if __name__ == '__main__':
    main()

