import os
import numpy as np
import p2_utils.common as utils
from p2_utils.options import args
from tensorboardX import SummaryWriter
from importlib import import_module

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from p2_model.gan import weights_init, Generator, Discriminator

from digit_classifier import Classifier



import torchvision.utils as vutils
import warnings

warnings.filterwarnings("ignore")

# torch.backends.cudnn.benchmark = False
device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')

# Set random seed for reproducibility.
seed = 369
np.random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)


def main():

    start_epoch = 0
    best_acc = 0.0
 
 
    # Data loading
    print('=> Preparing data..')
 
    loader = import_module('p2_data.data').__dict__['Data'](args, 
                                                            data_path=None,
                                                            label_path=None)

    loader_test = loader.loader_test

    
    # Create model
    print('=> Building model...')
    
    ARCH = args.arch

    # models
    model_g = Generator()
    model_g.apply(weights_init)

   
    model_g = model_g.to(device)
   
    if args.pretrained:
        state_dict = torch.load(args.source_dir + args.source_file)
        model_g.load_state_dict(state_dict['generator'])

   
    # classifier
    classifier = Classifier().to(device)
    
    state_dict = torch.load(args.classifer_model)
    classifier.load_state_dict(state_dict['state_dict'])
    
    classifier.to(device)
    

        
    best_acc = test(args, loader_test, model_g, classifier)
        
       
    print_logger.info(f'Best acc. : {best_acc:.3f}')
   
       
      
 
def test(args, loader_test, model_g, classifier):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    
    # switch to eval mode
    model_g.eval()
    classifier.eval()
    
    
    if not os.path.exists(os.path.join(args.output_test_data_path)):
        os.makedirs(os.path.join(args.output_test_data_path))
    
    COUNT_IMG = 1

    with torch.no_grad():
        for i, (noise_inputs, gen_labels, data_files) in enumerate(loader_test, 1):
            
            noise_inputs = noise_inputs.to(device)
            #print(noise_inputs.shape)
            gen_labels = gen_labels.to(device)

            gen_imgs = model_g(noise_inputs, gen_labels)
            
            # classification
            
            pred = classifier(gen_imgs)
            
            prec1, prec5 = utils.accuracy(pred, gen_labels, topk = (1, 5))
            top1.update(prec1[0], noise_inputs.size(0))
            top5.update(prec5[0], noise_inputs.size(0))

            # image classification results
            gen_imgs = gen_imgs.detach().cpu()
            
            for j in range(len(gen_imgs)):
                gen_img = vutils.make_grid(gen_imgs[j], padding=2, normalize=True)
     
                out_img  = transforms.ToPILImage()(gen_img)
                    
                datafile = data_files[j]
       
                out_img.save(os.path.join(args.output_test_data_path, datafile))

                COUNT_IMG += 1
         

    ## evaluate
    
    print_logger.info('Test_acc {top1.avg:.3f}\n'
                 '====================================\n'
                 .format(top1 = top1))

    return top1.avg
    


if __name__ == '__main__':
    main()

