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

    start_epoch = 0
    best_acc = 0.0
    
    # Data loading
    print('=> Loading preprocessed data..')
    
    loader = data.DataLoading(args)
    loader.load()
    
    loader_train = loader.train_loader
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

        new_state_dict = dict()
            
        for k, v in model.state_dict().items():
            if 'fc.' not in k:
                new_state_dict[k] = state_dict[k]
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        
        model = model.to(device)
   

    param = [param for name, param in model.named_parameters()]

    optimizer = optim.Adam(param, lr = args.lr)
    scheduler = StepLR(optimizer, step_size = args.lr_decay_step, gamma = 0.5)
   
    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location=device)
        best_acc = ckpt['best_acc']
        start_epoch = ckpt['epoch']

        model.load_state_dict(ckpt['state_dict'])

        optimizer.load_state_dict(ckpt['optimizer'])

        scheduler.load_state_dict(ckpt['scheduler'])

        print('=> Continue from epoch {}...'.format(start_epoch))


    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)

        train(args, loader_train, model, optimizer, epoch)
        test_acc = validate(args, loader_test, model)

        is_best = best_acc < test_acc
        best_acc = max(test_acc, best_acc)
              
        state = {
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
     
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
      
 
    print_logger.info(f"Best @test_acc: {best_acc:.3f}")

    
       
def train(args, loader_train, model, optimizer, epoch):
    losses = utils.AverageMeter()
    
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    cross_entropy = nn.CrossEntropyLoss()
    
    # switch to train mode
    model.train()
 
    num_iterations = len(loader_train)
    
    for i, (inputs, targets) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i
        
        inputs = inputs.to(device)
        
        p = args.shot * args.train_way
        data_shot, data_query = inputs[:p], inputs[p:]
    
        
        ## train weights
        optimizer.zero_grad()

        proto = model(data_shot)
        proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

        label = torch.arange(args.train_way).repeat(args.query) # re-label class from 0 to n_cls
        label = label.type(torch.cuda.LongTensor)

        logits = utils.distance_metric(model(data_query), proto, args.distance_metric)
        error = cross_entropy(logits, label)
      
        error.backward() # retain_graph = True
        losses.update(error.item(), inputs.size(0))
        
        ## step forward
        optimizer.step()
     

        ## evaluate
        prec1, prec5 = utils.accuracy(logits, label, topk = (1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        
 
        if i % args.print_freq == 0:
            print_logger.info(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                'Train_acc {top1.val:.3f} ({top1.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses, 
                top1 = top1))
        
            
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
    
            logits = utils.distance_metric(model(data_query), proto, args.distance_metric)
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

