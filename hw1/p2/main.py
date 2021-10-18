import os
import numpy as np
import utils.common as utils
from utils.options import args
#from tensorboardX import SummaryWriter
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch import Tensor
from data import data

import pandas as pd
import numpy as np

import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from mean_iou_evaluate import *

import time 

import warnings
warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
#writer_train = SummaryWriter(args.job_dir + '/run/train')
#writer_test = SummaryWriter(args.job_dir + '/run/test')

#print(str(args.test_only))
    
def main():

    start_epoch = 0
    best_miou = 0.0
    
    # Data loading
    print('=> Loading preprocessed data..')
    
    loader = data.DataLoading(args)
    loader.load()
    
    loader_train = loader.loader_train
    loader_test = loader.loader_test
    
    # Create model
    print('=> Building model...')
    
    if args.model == 'fcn_resnet50':
        pretrain_model = import_module('model.models').__dict__[args.model]()
    
        model = import_module('model.models').__dict__[args.model](num_classes = args.num_classes, 
                                                                   pretrained = False).to(device)
        state_dict = pretrain_model.state_dict()
        del pretrain_model
        
        new_state_dict = dict()
        
        for k, v in model.state_dict().items():
            if v.size == state_dict[k].size:
                new_state_dict[k] = state_dict[k]
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        
        del state_dict, new_state_dict
    
    else:
        model = import_module('model.models').__dict__[args.model](num_classes = args.num_classes, 
                                                                   pretrained = True).to(device)
    model = model.to(device)
        
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
        test_miou = validate(args, loader_test, model, epoch)

        is_best = best_miou < test_miou
        best_miou = max(test_miou, best_miou)
              
        state = {
            'state_dict': model.state_dict(),
            'best_miou': best_miou,
            
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
     
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
      
 
    print_logger.info(f"Best @test_mIOU: {best_miou:.3f}")




def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]-1):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / (input.shape[1]-1)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def train(args, loader_train, model, optimizer, epoch):
    losses = utils.AverageMeter()
    
    top1 = utils.AverageMeter()
    miou = utils.AverageMeter()

    #for name, param in list(model.named_parameters()):
        #print(name)
        #if 'feature' in name:
            #param.requires_grad = False
        
    criterion = nn.CrossEntropyLoss()
    
    # switch to train mode
    model.train()

    num_iterations = len(loader_train)
    
    for i, (inputs, targets, _) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
    
        ## train weights
        if args.model == 'fcn_resnet50':
            masks_pred = model(inputs)['out'].to(device)
        else:
            masks_pred = model(inputs).to(device)

        true_masks = torch.argmax(targets, 1)
     
        error = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       #F.one_hot(true_masks, args.num_classes).permute(0, 3, 1, 2).float(),
                                       targets.float(),
                                       multiclass=True)

        error.backward() # retain_graph = True
        
        losses.update(error.item(), inputs.size(0))
        
        #writer_train.add_scalar('Performance_loss', error.item(), num_iters)
        
        ## step forward
        optimizer.step()
    
        ## evaluate
        label_preds = torch.argmax(masks_pred, 1).cpu()
        label_trues = torch.argmax(targets, 1).cpu()
        acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(label_preds, 
                                                                    label_trues, 
                                                                    args.num_classes)
        
        miou.update(mean_iu*100., inputs.size(0))
        
        
        if i % args.print_freq == 1:
            print_logger.info(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                'mIOU {miou.val:.3f} ({miou.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses,  
                miou = miou))
        
            
def validate(args, loader_test, model, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    miou = utils.AverageMeter()


    # switch to eval mode
    model.eval()
    
    num_iterations = len(loader_test)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
                
    with torch.no_grad():
        for i, (inputs, targets, datafiles) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            if args.model == 'fcn_resnet50':
                output = model(inputs)['out'].to(device)
            else:
                output = model(inputs).to(device)

            ## evaluate
            label_preds = torch.argmax(output, 1).cpu()
            label_trues = torch.argmax(targets, 1).cpu()
            
            acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(label_preds, label_trues, args.num_classes)
            
            miou.update(mean_iu*100., inputs.size(0))
            
            # output predicted images
            preds = torch.argmax(output, 1).cpu()
 
            for j in range(inputs.size(0)):
                rgb_pred = class2img(preds[j])
                pil_pred  = transforms.ToPILImage()(rgb_pred)
                
                
                datafile = datafiles[j]
                datafile = datafile[:-4] + '.png'
                pil_pred.save(os.path.join(args.output_dir, datafile))

        mean_iou = mean_iou_score(pred = read_masks(args.test_dataset), 
                                  labels = read_masks(args.output_dir))
        time.sleep(0.5)
        print_logger.info(
            '\nAll  classes: mIOU {miou.val:.3f} ({miou.avg:.3f})\n'
            'Known classes: mIOU {mean_iou:.3f} \n'
            '=================================================\n'.format( 
            miou = miou,
            mean_iou = mean_iou*100))
    
    return mean_iou*100

def class2img(pred):
    class2rgb ={0: [0, 1, 1],
                1: [1, 1, 0],
                2: [1, 0, 1], 
                3: [0, 1, 0], 
                4: [0, 0, 1],
                5: [1, 1, 1],
                6: [0, 0, 0]}
    
    rgb_pred = torch.zeros([3, 512, 512])
    
    for i in range(args.num_classes):
        rgb = class2rgb[i]
        
        for j in range(3):
            rgb_pred[j][pred == i] = rgb[j]
        
    return rgb_pred
        


if __name__ == '__main__':
    main()

