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
seed = 369
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

    start_epoch = 0


    best_is = 0.0
    best_fid = 0.0

 
    # Data loading
    print('=> Preparing data..')
 
    loader = import_module('data.data').__dict__['Data'](args, 
                                                         data_path=args.data_path)

    loader_train = loader.loader_train
    loader_test = loader.loader_test


    # Create model
    print('=> Building model...')
    
    ARCH = args.arch

    # models
    model_g = Generator(params).to(device)
    model_g.apply(weights_init)
    
    model_d = Discriminator(params).to(device)
    model_d.apply(weights_init)
    
    model_g = model_g.to(device)
    model_d = model_d.to(device)
    models = [model_g, model_d]
    
    
    if args.pretrained:
        state_dict = torch.load(args.source_dir + args.source_file)
        model_g.load_state_dict(state_dict['generator'])

    # optimizers and schedulers
    param_g = [param for name, param in model_g.named_parameters()]
    optimizer_g = optim.Adam(param_g, lr = args.lr, betas=(params['beta1'], 0.999),
                             weight_decay = args.weight_decay)
    scheduler_g = StepLR(optimizer_g, args.lr_decay_step, gamma = args.lr_gamma)
    
    param_d = [param for name, param in model_d.named_parameters()]
    optimizer_d = optim.Adam(param_d, lr = args.lr, betas=(params['beta1'], 0.999),
                             weight_decay = args.weight_decay)
    scheduler_d = StepLR(optimizer_d, args.lr_decay_step, gamma = args.lr_gamma)
    
    

    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location=device)
        best_prec1 = ckpt['best_prec1']
        start_epoch = ckpt['epoch']

        model_g.load_state_dict(ckpt['state_dict_g'])

        optimizer_g.load_state_dict(ckpt['optimizer_g'])

        scheduler_g.load_state_dict(ckpt['scheduler_g'])
        
        model_d.load_state_dict(ckpt['state_dict_d'])

        optimizer_d.load_state_dict(ckpt['optimizer_d'])

        scheduler_d.load_state_dict(ckpt['scheduler_d'])
        
        print('=> Continue from epoch {}...'.format(start_epoch))

    '''
    if args.test_only:
        test_prec1, test_prec5 = test(args, loader.loader_test, model_s)
        print('=> Test Prec@1: {:.2f}'.format(test_prec1))
        return
    '''

    optimizers = [optimizer_g, optimizer_d]
    schedulers = [scheduler_g, scheduler_d]


    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)
        
        train(args, loader_train, models, optimizers, epoch)
        
        is_score, fid_score = test(args, loader_test, model_g, epoch)
        
        
        is_best = (best_is <= is_score) or (best_fid >= fid_score)
        
        if is_best:
            best_is = is_score
            best_fid = fid_score
        

        state = {
            'generator': model_g.state_dict(),
            
            'params': params,
            
            'best_is': best_is,
            'best_fid': best_fid,
            
            'optimizer_g': optimizer_g.state_dict(),
            'scheduler_g': scheduler_g.state_dict(),
            
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        

    print_logger.info(f'Best IS : {best_is:.3f} [Larger]')
    print_logger.info(f'Best FID: {best_fid:.3f} [Smaller]')
 


 
def train(args, loader_train, models, optimizers, epoch):
    losses_g = utils.AverageMeter()
    losses_d = utils.AverageMeter()

    model_g = models[0]
    model_d = models[1]

    adversarial_loss = nn.BCELoss()
    
    optimizer_g = optimizers[0]
    optimizer_d = optimizers[1]
    
    # switch to train mode
    model_g.train()
    model_d.train()
        
    num_iterations = len(loader_train)
    
    real_label = 0.9 #1
    fake_label = 0.
    
    if not os.path.exists(os.path.join(args.output_train_data_path)):
        os.makedirs(os.path.join(args.output_train_data_path))


    for i, (inputs, datafiles) in enumerate(loader_train, 1):
        
        num_iters = num_iterations * epoch + i

        # inputs
        real_imgs = inputs.to(device)
        b_size = real_imgs.size(0)
        
        # train discriminator
        optimizer_d.zero_grad()
       
        
        label = torch.full((b_size, ), real_label, device=device)
        output = model_d(real_imgs).view(-1)
        
        d_real_loss = adversarial_loss(output, label)
        d_real_loss.backward()
        
        noise_inputs = torch.randn(b_size, params['nz'], 1, 1, device=device)
        # Generate fake data (images).
        gen_imgs = model_g(noise_inputs)
 
        output_fake = model_d(gen_imgs.detach()).view(-1)
        label.fill_(fake_label  )

        d_fake_loss = adversarial_loss(output_fake, label)
        d_fake_loss.backward()
        
        
        d_loss = (d_real_loss + d_fake_loss)/2

        optimizer_d.step()
        
        # train generator
        optimizer_g.zero_grad()
        
        #noise_inputs = torch.from_numpy(np.random.uniform(-1, 1, (inputs.shape[0], args.input_dim))).float().to(device)
        
        gen_imgs = model_g(noise_inputs)
        
        output_gen = model_d(gen_imgs).view(-1)
        label.fill_(real_label)
        
        g_loss = adversarial_loss(output_gen, label)

        g_loss.backward()
        optimizer_g.step()
        
        

        
        ## train weights        
        losses_g.update(g_loss.item(), inputs.size(0))
        losses_d.update(d_loss.item(), inputs.size(0))
    
        writer_train.add_scalar('Generator_loss', g_loss.item(), num_iters)
        writer_train.add_scalar('Discriminator_loss', d_loss.item(), num_iters)
        
        ## output generated images
        '''
        for j in range(len(gen_imgs)):
            gen_img = gen_imgs[j].reshape([3, args.img_size, args.img_size])
        
            out_img  = transforms.ToPILImage()(gen_img)
                
            datafile = datafiles[j]
       
            out_img.save(os.path.join(args.output_train_data_path, datafile))
        '''
        if i % args.print_freq == 0:
            print_logger.info(
                'Epoch[{0}]({1}/{2}): \n'
                'Generation_loss: {g_loss.val:.4f} ({g_loss.avg:.4f})\n'
                'Discrimination_loss: {d_loss.val:.4f} ({d_loss.avg:.4f})\n'.format(
                epoch, i, num_iterations, 
                g_loss = losses_g,
                d_loss = losses_d))
            
    ## evaluate
    #is_score = utils.is_score(args.output_train_data_path)
    #fid_score = utils.fid_score(args.output_train_data_path, 
    #                            os.path.join(args.data_path, 'train'))

    #writer_train.add_scalar('Train-IS', is_score, num_iters)
    #writer_train.add_scalar('Train-FID', fid_score, num_iters)
        

    print_logger.info(
        'Epoch[{0}]({1}/{2}): \n'
        'Generation_loss: {g_loss.val:.4f} ({g_loss.avg:.4f})\n'
        'Discrimination_loss: {d_loss.val:.4f} ({d_loss.avg:.4f})\n'.format(
        epoch, i, num_iterations, 
        g_loss = losses_g,
        d_loss = losses_d))
                
      
 
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

