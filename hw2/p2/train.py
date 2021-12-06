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

from digit_classifier import Classifier

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


def main():

    start_epoch = 0
    best_acc = 0.0
 
 
    # Data loading
    print('=> Preparing data..')
 
    loader = import_module('data.data').__dict__['Data'](args, 
                                                         data_path=args.data_path,
                                                         label_path=args.label_path)

    loader_train = loader.loader_train
    loader_test = loader.loader_test

    
    # Create model
    print('=> Building model...')
    
    ARCH = args.arch

    # models
    model_g = Generator()
    model_g.apply(weights_init)

    
    model_d = Discriminator()
    model_d.apply(weights_init)

    
    model_g = model_g.to(device)
    model_d = model_d.to(device)
    models = [model_g, model_d]
    
    
    
    if args.pretrained:
        state_dict = torch.load(args.source_dir + args.source_file)
        model_d.load_state_dict(state_dict['discriminator'])

    # optimizers and schedulers
    param_g = [param for name, param in model_g.named_parameters()]
    optimizer_g = optim.Adam(param_g, lr = args.g_lr, betas=(args.b1, args.b2),
                             weight_decay = args.weight_decay)
    scheduler_g = StepLR(optimizer_g, args.lr_decay_step, gamma = args.lr_gamma)
    
    param_d = [param for name, param in model_d.named_parameters()]
    optimizer_d = optim.Adam(param_d, lr = args.d_lr, betas=(args.b1, args.b2),
                             weight_decay = args.weight_decay)
    scheduler_d = StepLR(optimizer_d, args.lr_decay_step, gamma = args.lr_gamma)
    
    

    optimizers = [optimizer_g, optimizer_d]
    schedulers = [scheduler_g, scheduler_d]

    # classifier
    classifier = Classifier().to(device)
    
    state_dict = torch.load(args.classifer_model)
    classifier.load_state_dict(state_dict['state_dict'])
    
    classifier.to(device)
    
    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)
        
        train(args, loader_train, models, optimizers, epoch)
        
        test_acc = test(args, loader_test, model_g, classifier, epoch)
        
        is_best = best_acc <= test_acc
        best_acc = max(test_acc, best_acc)
    

        state = {
            'generator': model_g.state_dict(),
            
            'best_acc': best_acc,
            
            'optimizer_g': optimizer_g.state_dict(),
            'scheduler_g': scheduler_g.state_dict(),
            
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        

    print_logger.info(f'Best acc. : {best_acc:.3f}')
   


 
def train(args, loader_train, models, optimizers, epoch):
    losses_g = utils.AverageMeter()
    losses_d = utils.AverageMeter()
    
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model_g = models[0]
    model_d = models[1]

    adversarial_loss = nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    
    optimizer_g = optimizers[0]
    optimizer_d = optimizers[1]
    
    # switch to train mode
    model_g.train()
    model_d.train()
        
    num_iterations = len(loader_train)
    
    real_label = 1.
    fake_label = 0.
    
    if not os.path.exists(os.path.join(args.output_train_data_path)):
        os.makedirs(os.path.join(args.output_train_data_path))


    for i, (inputs, labels, datafiles, 
            noise_inputs, gen_labels, sample_images) in enumerate(loader_train, 1):
        
        num_iters = num_iterations * epoch + i

        # inputs
        real_imgs = inputs.to(device)
        batch_size = real_imgs.size(0)
        
        # labels
        labels = labels.to(device)
        
        real_labels = torch.full((batch_size, 1), real_label, device=device)
        fake_labels = torch.full((batch_size, 1), fake_label, device=device)
        
        
        # train generator
        optimizer_g.zero_grad()
        
        ## noise inputs & labels
        noise_inputs = noise_inputs.to(device)
        gen_labels = gen_labels.reshape(-1).to(device)
        
        #print(gen_labels.shape)
        gen_imgs = model_g(noise_inputs, gen_labels)
        sample_images = sample_images.to(device)
        
        style_similarity = torch.mean(torch.abs(gen_imgs - sample_images)) / torch.max(torch.abs(gen_imgs - sample_images))
        
        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = model_d(gen_imgs)
        g_loss = style_similarity + (adversarial_loss(validity, real_labels) + auxiliary_loss(pred_label, gen_labels)) / 2
        
        
        
        g_loss.backward()
        optimizer_g.step()
        
        
        # train discriminator
        optimizer_d.zero_grad()
        
        ## Loss for real images
        real_pred, real_aux = model_d(real_imgs)

        d_real_loss = (adversarial_loss(real_pred, real_labels) + auxiliary_loss(real_aux, labels)) / 2
        
        ## Loss for fake images
        fake_pred, fake_aux = model_d(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake_labels) + auxiliary_loss(fake_aux, gen_labels)) / 2
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        optimizer_d.step()
        
        
        ## train weights        
        losses_g.update(g_loss.item(), inputs.size(0))
        losses_d.update(d_loss.item(), inputs.size(0))
    
        writer_train.add_scalar('Generator_loss', g_loss.item(), num_iters)
        writer_train.add_scalar('Discriminator_loss', d_loss.item(), num_iters)
        
        ## evaluate
        output = torch.cat([real_aux.data.cpu(), fake_aux.data.cpu()], axis=0)
        targets = torch.cat([labels.data.cpu(), gen_labels.data.cpu()], axis=0)
        
        prec1, prec5 = utils.accuracy(output, targets, topk = (1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        

        if i % args.print_freq == 0:
            print_logger.info(
                'Epoch[{0}]({1}/{2}): \n'
                'Generation_loss: {g_loss.val:.4f} ({g_loss.avg:.4f})\n'
                'Discrimination_loss: {d_loss.val:.4f} ({d_loss.avg:.4f})\n'
                'Train_acc {top1.val:.3f} ({top1.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                g_loss = losses_g,
                d_loss = losses_d, 
                top1 = top1))
            
    #print(noise_inputs.shape)
    print_logger.info(
        'Epoch[{0}]({1}/{2}): \n'
        'Generation_loss: {g_loss.val:.4f} ({g_loss.avg:.4f})\n'
        'Discrimination_loss: {d_loss.val:.4f} ({d_loss.avg:.4f})\n'.format(
        epoch, i, num_iterations, 
        g_loss = losses_g,
        d_loss = losses_d))
                
      
 
def test(args, loader_test, model_g, classifier, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    
    # switch to eval mode
    model_g.eval()
    classifier.eval()
    
    num_iterations = len(loader_test)
    
    if not os.path.exists(os.path.join(args.output_test_data_path)):
        os.makedirs(os.path.join(args.output_test_data_path))
    
    COUNT_IMG = 1

    with torch.no_grad():
        for i, (noise_inputs, gen_labels, data_files) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
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

