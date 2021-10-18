import os
import numpy as np
import utils.common as utils
from utils.options import args

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from data import cifar100

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
    
    loader = cifar100.Data(args)
    
    loader_train = loader.loader_train
    loader_test = loader.loader_test
    
    # Create model
    print('=> Building model...')
    
    
    model = import_module('model.models').__dict__[args.model](num_classes = args.num_classes).to(device)
    
    param = [param for name, param in model.named_parameters()]

    optimizer = optim.SGD(param, lr = args.lr, momentum = args.momentum, 
                          weight_decay = args.weight_decay)
    
    scheduler = StepLR(optimizer, step_size = args.lr_decay_step, gamma = 0.1)
   
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
        test_acc = test(args, loader_test, model, epoch)

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
        targets = targets.to(device)
        
        optimizer.zero_grad()
    
        ## train weights
        output = model(inputs).to(device)
        
        error = cross_entropy(output, targets)

        error.backward() # retain_graph = True
        
        losses.update(error.item(), inputs.size(0))
        

        ## step forward
        optimizer.step()
     

        ## evaluate
        prec1, prec5 = utils.accuracy(output, targets, topk = (1, 5))
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
        
            
def test(args, loader_test, model, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model.eval()
    
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs).to(device)
            loss = cross_entropy(logits, targets)
            
            prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            

    print_logger.info('Test_acc {top1.avg:.3f}\n'
                      '====================================\n'
                      .format(top1 = top1))

    return top1.avg
    

if __name__ == '__main__':
    main()

