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

    # Data loading
    print('=> Preparing data..')
 
    
    tgt_loader = import_module('data.data').__dict__['Data'](args,  
                                                             data_path=args.tgt_data_path, 
                                                             label_path=args.tgt_label_path)

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
        
       
    tgt_best_prec1, tgt_best_prec5 = test(args, tgt_data_loader_test, 
                                          model_t, 0, 'target')

   
    print_logger.info(f'Best @prec1: {tgt_best_prec1:.3f} @prec5: {tgt_best_prec5:.3f} [Target class]')


 
def test(args, loader_test, model_t, epoch, flag):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

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
    
    return top1.avg, top5.avg
    


if __name__ == '__main__':
    main()

