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

from model.gan import weights_init, Generator, Discriminator

import matplotlib.pyplot as plt
from PIL import Image
import pickle as pkl

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
seed = 1000
np.random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Parameters to define the model.
params = {
    "bsize" : args.train_batch_size,# Batch size during training.
    'imsize' : args.img_size,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nz' : args.input_dim,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 10,# Number of training epochs.
    'lr' : args.lr,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 2}# Save step.



def main():


    # Data loading
    print('=> Preparing data..')
 
    loader = import_module('data.data').__dict__['Data'](args, 
                                                         data_path=args.data_path)

    loader_test = loader.loader_test


    # Create model
    print('=> Building model...')
    
    # models
    model_g = Generator(params).to(device)

    
    
    if args.pretrained:
        state_dict = torch.load(args.source_dir + args.source_file)
        model_g.load_state_dict(state_dict['generator'])

    model_g = model_g.to(device)

    is_score, fid_score = test(args, loader_test, model_g, 0)
    

    print(f'IS: {is_score:.3f}, FID: {fid_score:.3f}')            
      
 
def test(args, loader_test, model_g, epoch):

    # switch to eval mode

    model_g.eval()
    
    num_iterations = len(loader_test)
    
    if not os.path.exists(os.path.join(args.output_test_data_path)):
        os.makedirs(os.path.join(args.output_test_data_path))
    
    COUNT_IMG = 1

    with torch.no_grad():
        for i, (noise_inputs) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            noise_inputs = noise_inputs.to(device)

            gen_imgs = model_g(noise_inputs).detach().cpu()
    

            # image classification results
            for j in range(len(gen_imgs)):
                gen_img = vutils.make_grid(gen_imgs[j], padding=2, normalize=True)
     
                #print(img)
                out_img  = transforms.ToPILImage()(gen_img)
                    
                datafile = f'{COUNT_IMG:04d}.png'
       
                out_img.save(os.path.join(args.output_test_data_path, datafile))
                
                #im = Image.fromarray(img.reshape((args.img_size, -1, 3)))
                #im.save(os.path.join(args.output_test_data_path, datafile))

                COUNT_IMG += 1
         

    ## evaluate
    
    is_score = utils.is_score(args.output_test_data_path)
    fid_score = utils.fid_score(args.output_test_data_path, 
                                os.path.join(args.data_path, 'test'))

            
    writer_test.add_scalar('IS ', is_score, num_iters)
    writer_test.add_scalar('FID', fid_score, num_iters)
    
    print_logger.info(f'IS  {is_score:.3f}')
    print_logger.info(f'FID {fid_score:.3f}\n=================================\n')

    return is_score, fid_score
    


if __name__ == '__main__':
    main()

