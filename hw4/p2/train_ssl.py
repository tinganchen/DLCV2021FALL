import os

import utils.common as utils
from utils.options import args

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

import torch.nn.functional as F

from data import data_ssl

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
    
    loader = data_ssl.DataLoading(args)
    loader.load()
    
    loader_train = loader.train_loader
    loader_test = loader.val_loader

    
    # Create model
    
    print('=> Building model...')
    model_online = import_module(f'model.{args.model}').__dict__['resnet_50']().to(device)
    model_target = import_module(f'model.{args.model}').__dict__['resnet_50']().to(device)
    
    if args.pretrained == 'True':
        # Load pretrained weights
        state_dict = torch.load(args.pretrain_dir + args.pretrain_file, map_location = device)
        
        new_state_dict = model_online.state_dict()
        
        state_dict_keys = list(state_dict.keys())
        
        for i, (k, v) in enumerate(list(model_online.state_dict().items())[:-2]):
            if 'fc' not in k:
                new_state_dict[k] = state_dict[state_dict_keys[i]]
            else:
                new_state_dict[k] = state_dict[state_dict_keys[i]]

        model_online.load_state_dict(new_state_dict)
        model_target.load_state_dict(new_state_dict)
        
        model_online = model_online.to(device)
        model_target = model_target.to(device)
   

    param_online = [param for name, param in model_online.named_parameters()]
    param_target = [param for name, param in model_target.named_parameters()]

    optimizer_online = optim.Adam(param_online, lr = args.lr)
    scheduler_online = StepLR(optimizer_online, step_size = args.lr_decay_step, gamma = 0.5)
   
    optimizer_target = optim.Adam(param_target, lr = args.lr)
    scheduler_target = StepLR(optimizer_target, step_size = args.lr_decay_step, gamma = 0.5)
    
    models = [model_online, model_target]
    optimizers = [optimizer_online, optimizer_target]

    for epoch in range(start_epoch, args.num_epochs):
        scheduler_online.step(epoch)
        scheduler_target.step(epoch)

        train(args, loader_train, models, optimizers, epoch)
        test_acc = validate(args, loader_test, model_target)

        is_best = best_acc < test_acc
        best_acc = max(test_acc, best_acc)
              
        state = {
            'state_dict_online': model_online.state_dict(),
            'state_dict_target': model_target.state_dict(),
            'best_acc': best_acc,
            
            'optimizer_online': optimizer_online.state_dict(),
            'scheduler_online': scheduler_online.state_dict(),
            
            'optimizer_target': optimizer_target.state_dict(),
            'scheduler_target': scheduler_target.state_dict(),
     
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
      
 
    print_logger.info(f"Best @test_acc: {best_acc:.3f}")

    
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for (name, current_params), (_, ma_params) in zip(current_model.named_parameters(), ma_model.named_parameters()):
        if 'fc' not in name:
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = ema_updater.update_average(old_weight, up_weight)
        
def train(args, loader_train, models, optimizers, epoch):
    losses = utils.AverageMeter()
    byol_losses = utils.AverageMeter()
    
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    model_online = models[0]
    model_target = models[1]
    
    optimizer_online = optimizers[0]
    optimizer_target = optimizers[1]
    
    cross_entropy = nn.CrossEntropyLoss()
    
    
    for name, param in model_online.named_parameters():
        if 'fc' not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
   
    target_ema_updater = EMA(0.99)
    
    # switch to train mode
    model_online.train()
    model_target.train()
 
    num_iterations = len(loader_train)
    
    for i, (inputs1, inputs2, targets) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i

        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        targets = targets.to(device)
        
        optimizer_online.zero_grad()
        optimizer_target.zero_grad()
    
        ## train weights
        output1_online = model_online(inputs1)
        feature1_online = model_online.feature
        
        output2_online = model_online(inputs2)
        feature2_online = model_online.feature
        
        
        with torch.no_grad():
            output1_target = model_target(inputs1)
            feature1_target = model_target.feature
            
            _ = model_target(inputs2)
            feature2_target = model_target.feature
         
        
        output1_target = model_target(inputs1)
        feature1_target = model_target.feature
        
        output2_target = model_target(inputs2)
        feature2_target = model_target.feature
            
        loss_one = loss_fn(feature1_online, feature2_target)
        loss_two = loss_fn(feature2_online, feature1_target)

        byol_error = (loss_one + loss_two).mean()
        
        #byol_error.backward(retain_graph = True) # retain_graph = True
        
        byol_losses.update(byol_error.item(), inputs1.size(0))
        
        '''
        error = cross_entropy(output1_online, targets)/4 + \
                cross_entropy(output1_target, targets)/4 + \
                cross_entropy(output2_online, targets)/4 + \
                cross_entropy(output2_target, targets)/4 + byol_error
        
        error = cross_entropy(output1_online, targets) + \
                cross_entropy(output1_target, targets) + byol_error
        '''
        error = byol_error

        error.backward() # retain_graph = True
        
        losses.update(error.item(), inputs1.size(0))
        
        ## step forward
        optimizer_online.step()
        optimizer_target.step()
    
        ## evaluate
        prec1, prec5 = utils.accuracy(output1_target, targets, topk = (1, 5))
        top1.update(prec1[0], inputs1.size(0))
        top5.update(prec5[0], inputs1.size(0))
        
        if i % args.print_freq == 0:
            print_logger.info(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                'BYOL loss: {byol_loss.val:.4f} ({byol_loss.avg:.4f})\n'
                'Train_acc {top1.val:.3f} ({top1.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses, 
                byol_loss = byol_losses,
                top1 = top1))  
            
        update_moving_average(target_ema_updater, 
                              model_target, model_online)
        
def validate(args, loader_test, model, epoch = 0):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model.eval()
    
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, _, targets) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            
            prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

    print_logger.info('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
                      '===============================================\n'
                      .format(top1 = top1, top5 = top5))

    return top1.avg
        


if __name__ == '__main__':
    main()

