import os
import numpy as np
import utils.common as utils
from utils.options import args
from tensorboardX import SummaryWriter
from importlib import import_module

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from model.gan import weights_init, Discriminator



import torchvision.utils as vutils
import warnings

warnings.filterwarnings("ignore")

# torch.backends.cudnn.benchmark = False
device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')


def main():

    start_epoch = 0
    best_acc = 0.0
 
 
    # Data loading
    print('=> Preparing data..')
 
    loader = import_module('data.data_pretrain').__dict__['Data'](args, 
                                                                  data_path=args.data_path,
                                                                  label_path=args.label_path)

    loader_train = loader.loader_train
    loader_test = loader.loader_test


    # Create model
    print('=> Building model...')
    
    ARCH = args.arch

    # models

    model_d = Discriminator()
    model_d.apply(weights_init)

    model_d = model_d.to(device)
    models = [model_d]
    
    
    param_d = [param for name, param in model_d.named_parameters()]
    optimizer_d = optim.Adam(param_d, lr = args.lr, betas=(args.b1, args.b2),
                             weight_decay = args.weight_decay)
    scheduler_d = StepLR(optimizer_d, args.lr_decay_step, gamma = args.lr_gamma)
    

    optimizers = [optimizer_d]
    schedulers = [scheduler_d]

    
    
    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)
        
        train(args, loader_train, models, optimizers, epoch)
        
        test_acc = test(args, loader_test, model_d)
        
        is_best = best_acc <= test_acc
        best_acc = max(test_acc, best_acc)
    

        state = {
            'discriminator': model_d.state_dict(),
            
            'best_acc': best_acc,
            
            'optimizer_d': optimizer_d.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),
            
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        

    print_logger.info(f'Best acc. : {best_acc:.3f}')
   


 
def train(args, loader_train, models, optimizers, epoch):
    losses = utils.AverageMeter()
    
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model = models[0]

    cross_entropy = nn.CrossEntropyLoss()
    
    optimizer = optimizers[0]
    
    # switch to train mode
    model.train()
        
    num_iterations = len(loader_train)
    


    for i, (inputs, targets, _) in enumerate(loader_train, 1):
        
        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
    
        ## train weights
        _, output = model(inputs)
        output = output.to(device)
        
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
        
      
 
def test(args, loader_test, model):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    
    # switch to eval mode
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, targets, data_file) in enumerate(loader_test, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            _, logits = model(inputs)
            logits = logits.to(device)
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

