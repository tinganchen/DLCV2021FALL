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
    
    loader_train = loader.loader_train
    loader_test = loader.loader_test
    
    # Create model
    print('=> Building model...')
    model = import_module('model.models').__dict__[args.model](num_classes = args.num_classes).to(device)

    
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
        test_acc, _ = validate(args, loader_test, model, epoch)

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
    
    for i, (inputs, targets, _) in enumerate(loader_train, 1):
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
        
            
def validate(args, loader_test, model, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()
    
    embs = []
    tgt_label = []
    
    # switch to eval mode
    model.eval()
    
    num_iterations = len(loader_test)
    
    
    with torch.no_grad():
        for i, (inputs, targets, data_file) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs).to(device)
            loss = cross_entropy(logits, targets)
            

            prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
            if args.tsne == 'True':
                tgt_label.extend(list(targets.cpu().numpy()))
                
                batch_embs = model.emb.reshape([inputs.size(0), -1]).cpu().detach().numpy()
                for batch_emb in batch_embs:
                    embs.append(batch_emb)
            
    if args.tsne == 'True':
        # t-SNE
        print('\n=> Preparing t-SNE visualization...')
        NUM_COLORS = args.num_classes
        cm = plt.get_cmap('tab20b')
        clrs = [cm(1.*color_id/NUM_COLORS) for color_id in range(NUM_COLORS)]
        
        tsne_class = TSNE(perplexity = 30, n_components = 2, init='pca', n_iter = 5000)
        class_emb = tsne_class.fit_transform(embs)
        
        for c in range(len(class_emb)):
            plt.scatter(class_emb[c, 0], class_emb[c, 1], 
                        color = clrs[tgt_label[c]], s = 2)
            
        plt.axis('off')
        plt.savefig(f'{os.path.join(args.job_dir, "tsne_result.png")}')
        
        print(f'Please see the t-SNE result:\n{os.path.join(args.job_dir, "tsne_result.png")}\n')
            
    print_logger.info('Test_acc {top1.avg:.3f}\n'
                 '====================================\n'
                 .format(top1 = top1))
    
    
    return top1.avg, embs
        


if __name__ == '__main__':
    main()

