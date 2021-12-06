import os
import numpy as np
import utils.common as utils
from utils.options import args
#from utils.options import args
from tensorboardX import SummaryWriter
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from utils.optimizer import SGD


#from ptflops import get_model_complexity_info # from thop import profile

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
    
    tgt_best_prec1 = 0.0
    tgt_best_prec5 = 0.0
    tgt_best_prec1_domain = 0.0

    src_best_prec1 = 0.0
    src_best_prec5 = 0.0
    src_best_prec1_domain = 0.0
 
    # Data loading
    print('=> Preparing data..')
 
    src_loader = import_module('data.data').__dict__['Data'](args, 
                                                             data_path=args.src_data_path, 
                                                             label_path=args.src_label_path)
    
    tgt_loader = import_module('data.data').__dict__['Data'](args,  
                                                             data_path=args.tgt_data_path, 
                                                             label_path=args.tgt_label_path)

    
    src_data_loader_train = src_loader.loader_train
    tgt_data_loader_train = tgt_loader.loader_train
    
    src_data_loader_test = src_loader.loader_test
    tgt_data_loader_test = tgt_loader.loader_test
      

    # Create model
    print('=> Building model...')
    
    ARCH = args.arch

    # load training model
    model_t = import_module(f'model.{ARCH}').__dict__[args.model]().to(device)
    

    # Load pretrained weights
    if args.pretrained:
            
        ckpt = torch.load(args.source_dir + args.source_file, map_location = device)
        state_dict = ckpt['state_dict_t']
    
        model_dict_t = model_t.state_dict()
        
        for name, param in state_dict.items():
            if name in list(model_dict_t.keys()):
                model_dict_t[name] = param
        
        model_t.load_state_dict(model_dict_t)
        model_t = model_t.to(device)

        del ckpt, state_dict, model_dict_t
        
       
    models = [model_t]

    
    param_t = [param for name, param in model_t.named_parameters()]
    optimizer_t = optim.SGD(param_t, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    scheduler_t = StepLR(optimizer_t, args.lr_decay_step, gamma = args.lr_gamma)


    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location=device)
        best_prec1 = ckpt['best_prec1']
        start_epoch = ckpt['epoch']

        model_t.load_state_dict(ckpt['state_dict_t'])

        optimizer_t.load_state_dict(ckpt['optimizer_t'])

        scheduler_t.load_state_dict(ckpt['scheduler_t'])
        
        print('=> Continue from epoch {}...'.format(start_epoch))

    '''
    if args.test_only:
        test_prec1, test_prec5 = test(args, loader.loader_test, model_s)
        print('=> Test Prec@1: {:.2f}'.format(test_prec1))
        return
    '''

    optimizers = [optimizer_t]
    schedulers = [scheduler_t]

    #optimizers = [optimizer, optimizer_t]
    #schedulers = [scheduler, scheduler_t]

    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)
        
        train(args, src_data_loader_train, tgt_data_loader_train, models, optimizers, epoch)
        
        tgt_test_prec1, tgt_test_prec5, tgt_test_prec1_domain = test(args, tgt_data_loader_test, 
                                                                     model_t, epoch, 'target')
        
        src_test_prec1, src_test_prec5, src_test_prec1_domain = test(args, src_data_loader_test, 
                                                                     model_t, epoch, 'source')
        
        
        is_best = tgt_best_prec1 < tgt_test_prec1
        tgt_best_prec1 = max(tgt_test_prec1, tgt_best_prec1)
        tgt_best_prec5 = max(tgt_test_prec5, tgt_best_prec5)
        
        tgt_best_prec1_domain = max(tgt_test_prec1_domain, tgt_best_prec1_domain)
        
        src_best_prec1 = max(src_test_prec1, src_best_prec1)
        src_best_prec5 = max(src_test_prec5, tgt_best_prec5)
        
        src_best_prec1_domain = max(src_test_prec1_domain, src_best_prec1_domain)


        state = {
            'state_dict_t': model_t.state_dict(),
            
            'tgt_best_prec1': tgt_best_prec1,
            'tgt_best_prec5': tgt_best_prec5,
            'tgt_best_prec1_domain': tgt_best_prec1_domain,
            
            'src_best_prec1': src_best_prec1,
            'src_best_prec5': src_best_prec5,
            'src_best_prec1_domain': src_best_prec1_domain,
            
            'optimizer_t': optimizer_t.state_dict(),
            'scheduler_t': scheduler_t.state_dict(),
            
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        

    print_logger.info(f'Best @prec1: {tgt_best_prec1:.3f} @prec5: {tgt_best_prec5:.3f} [Target class]')
    print_logger.info(f'Best @prec1: {src_best_prec1:.3f} @prec5: {src_best_prec5:.3f} [Source class]')
 


def adjust_learning_rate(optimizer, p):
    
    lr_0 = args.lr
    alpha = args.alpha
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        '''
    # office & mnist-svhn
    lr_0 = 0.001
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups[:2]:
        param_group['lr'] = lr
    for param_group in optimizer.param_groups[2:]:
        param_group['lr'] = 10 * lr'''
    return lr

 
def train(args, src_data_loader, tgt_data_loader, models, optimizers, epoch):
    losses_t = utils.AverageMeter()
    losses_src_class = utils.AverageMeter()
    losses_tgt_class = utils.AverageMeter()
    losses_src_domain = utils.AverageMeter()
    losses_tgt_domain = utils.AverageMeter()


    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model_t = models[0]
    
    param_t = []
    
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if 'alpha' not in name:
            param_t.append((name, param))
            
    criterion = nn.CrossEntropyLoss()
    
    optimizer_t = optimizers[0]
    
    # switch to train mode
    model_t.train()
        
    num_iterations = min(len(src_data_loader), len(tgt_data_loader))
    
    src_iter_data = iter(src_data_loader)
    tgt_iter_data = iter(tgt_data_loader)
    
    num_iters = 0
    insert_iter = -1
    tmp_info = None
    
    #for i, ((images_src, class_src), (images_tgt, _)) in data_zip:
    for i in range(num_iterations): 

        num_iters = num_iterations * epoch + i
        
        
        images_src, class_src, _ = src_iter_data.next()
        images_tgt, class_tgt, _ = tgt_iter_data.next()
        
        if i == insert_iter:
            info = tmp_info
            if info[-1] == 'tgt':
                insert_size = tmp_info[0].shape[0]
                images_tgt[:insert_size] = tmp_info[0]
            else:
                insert_size = tmp_info[0].shape[0]
                images_src[:insert_size] = tmp_info[0]
                class_src[:insert_size] = tmp_info[1]
        
        if images_tgt.shape[0] < images_src.shape[0]:
            tmp_info = [images_tgt, 'tgt']
            n = len(tgt_data_loader) - 1
            insert_iter = i + np.random.choice(range(n))
            
            tgt_iter_data = iter(tgt_data_loader)
            images_tgt, class_tgt, _ = tgt_iter_data.next()
        
        if images_tgt.shape[0] > images_src.shape[0]:
            tmp_info = [images_src, class_src, 'src']
            n = len(src_data_loader) - 1
            insert_iter = i + np.random.choice(range(n))
            
            src_iter_data = iter(src_data_loader)
            images_src, class_src, _ = src_iter_data.next()

        #print(images_src.shape, images_tgt.shape)
        
        p = float(num_iters) / args.num_epochs / num_iterations
        lambd = 2. / (1. + np.exp(-10 * p) + 1e-6) - 1
        #  2. / (1. + np.exp(-10 * p)) - 1
        
 
        
        # prepare domain label
        size_src = len(images_src)
        size_tgt = len(images_tgt)
  
        # make images variable
        class_src = class_src.to(device)
        class_tgt = class_tgt.to(device)
        images_src = images_src.to(device)
        images_tgt = images_tgt.to(device)
        
        
        # train on source domain
        src_class_output, loss_mmd = model_t(images_src, images_tgt, class_src)
        src_loss_class = criterion(src_class_output, class_src)
        
        tgt_class_output, _ = model_t(images_tgt, images_tgt, None)
        tgt_loss_class = criterion(tgt_class_output, class_tgt)
        
        if args.method == 'src_only':
            loss = src_loss_class 
        elif args.method == 'src_tgt':
            loss = src_loss_class + args.param * lambd * loss_mmd
        elif args.method == 'tgt_only':
            loss = tgt_loss_class
        else:
            print(f'{args.method} in argument is incorrectly named.')    
        
        


        # optimize dann
        loss.backward()
        optimizer_t.step()


        ## train weights        
        losses_t.update(loss.item(), size_src)
        
        losses_src_class.update(src_loss_class.item(), size_src)
        losses_tgt_class.update(tgt_loss_class.item(), size_tgt)

    
    
        writer_train.add_scalar('Performance_loss', loss.item(), num_iters)
        writer_train.add_scalar('Source_class_loss', src_loss_class.item(), num_iters)
        
  
        
        ## evaluate
        prec1, prec5 = utils.accuracy(src_class_output, class_src, topk = (1, 5))
        top1.update(prec1[0], size_src)
        top5.update(prec5[0], size_src)
        
        writer_train.add_scalar('Train-top-1', top1.avg, num_iters)
        writer_train.add_scalar('Train-top-5', top5.avg, num_iters)
        
        if i % args.print_freq == 0:
            if args.method == 'src_only':
                print_logger.info(
                    'Epoch[{0}]({1}/{2}): \n'
                    'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                    'Source_class_loss: {src_class_loss.val:.4f} ({src_class_loss.avg:.4f})\n'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, num_iterations, 
                    train_loss = losses_t,
                    src_class_loss = losses_src_class,
                    top1 = top1, 
                    top5 = top5))
            elif args.method == 'src_tgt':
                print_logger.info(
                    'Epoch[{0}]({1}/{2}): \n'
                    'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                    'Source_class_loss: {src_class_loss.val:.4f} ({src_class_loss.avg:.4f})\n'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, num_iterations, 
                    train_loss = losses_t,
                    src_class_loss = losses_src_class,
                    top1 = top1, 
                    top5 = top5))
            elif args.method == 'tgt_only':
                print_logger.info(
                    'Epoch[{0}]({1}/{2}): \n'
                    'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                    'Target_class_loss: {tgt_class_loss.val:.4f} ({tgt_class_loss.avg:.4f})\n'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, num_iterations, 
                    train_loss = losses_t,
                    tgt_class_loss = losses_tgt_class,
                    top1 = top1, 
                    top5 = top5))
      
 
def test(args, loader_test, model_t, epoch, flag):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    top1_domain = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to eval mode
    model_t.eval()
    
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, targets, data_file) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            preds, _ = model_t(inputs, inputs, None)
    
            loss = criterion(preds, targets)
            
            writer_test.add_scalar('Test_loss', loss.item(), num_iters)
            
            # image classification results
            prec1, prec5 = utils.accuracy(preds, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
            writer_test.add_scalar('Test-top-1', top1.avg, num_iters)
            writer_test.add_scalar('Test-top-5', top5.avg, num_iters)
            
 
    print_logger.info(f'{flag}')
    print_logger.info(f'Prec@1 {int(top1.avg*10**3)/10**3} Prec@5 {int(top5.avg*10**3)/10**3} (Image)')
    print_logger.info('============================================================')
    
    return top1.avg, top5.avg, top1_domain.avg
    


if __name__ == '__main__':
    main()

