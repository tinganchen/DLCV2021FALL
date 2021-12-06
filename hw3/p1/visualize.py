import os

import utils.common as utils
from utils.options import args

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR

from data import data

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2

import warnings
warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))

def postprocess_activations(activations, img_size):
      
      #using the approach in https://arxiv.org/abs/1612.03928
      output = np.abs(activations)/np.abs(activations).max()
      #resize and convert to image 
      output = cv2.resize(output, img_size)
      #activations+=1
      output *= 255#/2
      
      return 255-np.abs(output.astype('uint8'))
  
def apply_heatmap(weights, img):
      #generate heat maps 
      heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
      heatmap = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)
      return heatmap

def main():

    # Create model
    
    print('=> Building model...')

    model = torch.load(args.pretrain_arch)
    
   
    if args.pretrained == 'True':
        # Load pretrained weights
        ckpt = torch.load(args.pretrain_dir + args.pretrain_file, map_location = device)
        state_dict = ckpt[list(ckpt.keys())[0]]

        new_state_dict = dict()
            
        for k, v in model.state_dict().items():
            new_state_dict[k] = state_dict[k]

        model.load_state_dict(new_state_dict)
        
        model = model.to(device)
   
    # visualize position embedding
    print('=> Visualize position embedding...')
    
    pos_embed = model.pos_embed
    print(pos_embed.shape) # [1, 577, 768]
    
    # Visualize position embedding similarities.
    # One cell shows cos similarity between an embedding and all the other embeddings.
    #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    fig = plt.figure(figsize=(8, 8))

    fig.suptitle("Position embedding similarities", fontsize=24)
    for i in range(1, pos_embed.shape[1]):
        sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
        sim = sim.reshape((24, 24)).detach().cpu().numpy()
        ax = fig.add_subplot(24, 24, i)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        
        im = ax.imshow(sim)
        im.set_clim(vmin=-1,vmax=1)
    fig.colorbar(im, ax=fig.get_axes())
     
    
    print('Finish position embedding visualization.')
    
    # visualize attention

    #pos_embed = model.pos_embed
    #cls_token = model.cls_token
    
    data_files = ['26_5064.jpg', '29_4718.jpg', '31_4838.jpg'] # '26_5064.jpg', '29_4718.jpg', '31_4838.jpg'
    
    # data_file = '26_5064.jpg'
    for data_file in data_files:
        img_path = os.path.join(args.test_dataset, data_file)
        image = Image.open(img_path).convert('RGB')
        
        transform_test = transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        
        image = transform_test(image)
        x = image.unsqueeze(0).to(device)
        
        B = x.shape[0]
        x = model.patch_embed(x)
    
        cls_tokens = model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + model.pos_embed
        x = model.pos_drop(x)
        
        blk = model.blocks[11]
        x = blk.norm1(x)
        
        attention = blk.attn
        
        B, N, C = x.shape
        qkv = attention.qkv(x).reshape(B, N, 3, attention.num_heads, C // attention.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * attention.scale # [1, 12, 577, 577]
        #attn = attn.softmax(dim=-1)
        #attn = attention.attn_drop(attn)
    
        #x = (attn @ v).transpose(1, 2).reshape(B, N, C) # [1, 577, 768]
        
        # show org img
        img = cv2.imread(img_path)[:,:,::-1]
        img = cv2.resize(img, (args.img_size, args.img_size))
        ax = plt.imshow(img)
        plt.axis('off')
        
        # Visualize attention
        # median of the query results
        #for i in range(577):
        activations = torch.mean(attn[0, :, 0, 1:], 0).reshape((24, 24)).detach().cpu().numpy()
        weights = postprocess_activations(activations, (img.shape[1], img.shape[0]))
        heatmap = apply_heatmap(weights, img)
    
        plt.axis('off')
        ax = plt.imshow(heatmap)
        plt.show()
        #plt.savefig(f'plt/dog/{i}.png')
       
            



if __name__ == '__main__':
    main()

