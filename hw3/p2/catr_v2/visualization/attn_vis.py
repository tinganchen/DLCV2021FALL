import torch

from transformers import BertTokenizer
from PIL import Image, ImageDraw, ImageFont
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision
import math


parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, default = '../../../hw3_TA/hw3_data/hw3_data/p2_data/images/', help='path to image')
parser.add_argument('--v', type=str, help='version', default='v3')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
parser.add_argument('--fig_path', type=str, help='figure path', default='fig')
args = parser.parse_args()

image_path = args.path
version = args.v
checkpoint_path = args.checkpoint
fig_path = args.fig_path

if not os.path.isdir(fig_path):
    os.mkdir(fig_path)

config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
      raise NotImplementedError('No model to chose from!')
    else:
      if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
      print("Found checkpoint! Loading!")
      model,_ = caption.build_model(config)
      print("Loading Checkpoint...")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      model.load_state_dict(checkpoint['model'])
      


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template



def postprocess_activations(activations, img_size):
 
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
  
@torch.no_grad()
def visualize(tokenizer, start_token, img_path):
    caption, cap_mask = create_caption_and_mask(
        start_token, config.max_position_embeddings)
    
    img = cv2.imread(img_path)[:,:,::-1]

    plt.imsave(f'{fig_path}/0.jpg', img)
    words = ['<start>']
            
    model.eval()    
    output = []
    
    COUNT = 0
    for i in range(config.max_position_embeddings - 1):
        
        image = Image.open(img_path)
        image = coco.val_transform(image)
        image = image.unsqueeze(0)

        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        
        if predicted_id[0] == 102:            
            break
            
        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
        
        word = tokenizer.decode(predicted_id[0].tolist(), skip_special_tokens=True)
        output.append(predicted_id[0])
        
        if '[' not in word and '#' not in word:
            COUNT += 1
            words.append(word.replace(".", "<end>"))
            last_decode_layer = model.transformer.decoder.layers[5]
            attn = last_decode_layer.attn # [1, 128, 247]
            activations = attn[0, i, :].reshape([model.transformer.h, model.transformer.w]).detach().cpu().numpy()
       
            weights = postprocess_activations(activations, (img.shape[1], img.shape[0]))
            heatmap = apply_heatmap(weights, img)
            
            plt.imsave(f'{fig_path}/{COUNT}.jpg', heatmap)
            
    result = tokenizer.decode(output, skip_special_tokens=True)
    print(result.capitalize())
    return caption, words


def coplot(img_path, words, rm_indiv_fig = True):
    figs = [f for f in os.listdir(f'{fig_path}') if '.jpg' in f]
    n_figs = len(figs)
    n_row = 4
    n_col = math.ceil(n_figs / n_row)
    
    path = f'{fig_path}/0.jpg'
    img = Image.open(path)
    
    new_im = Image.new('RGB', (n_row*img.size[0], n_col*img.size[1]))
    
    for i in range(0, n_row):
        for j in range(0, n_col):
        
            k = j*n_row + i
            
            if k >= n_figs:
                break
            
            path = f'{fig_path}/{k}.jpg'
            img = Image.open(path)
            
            draw = ImageDraw.Draw(img)  
            font = ImageFont.truetype("ArnoPro-Bold.otf", size=60)
            text = words[k]
            
            if text == '<start>':
                color = 'yellow'
            else:
                color = 'white'
                
            draw.text((10, 10), text = text, fill = color, font = font)
            
            new_im.paste(img, (i*img.size[0], j*img.size[1]))
      
            
    new_im.save(f'{fig_path}/{img_path.split("/")[-1][:-4]}.png')
    
    if rm_indiv_fig:
        os.system(f'rm -r {fig_path}/*.jpg')

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    
    for path in os.listdir(image_path):
        print(f'\nProcess figure: {path}\nCaption results:')
        img_path = os.path.join(image_path, path)
        caption, words = visualize(tokenizer, start_token, img_path)
        #result = tokenizer.decode(caption[0], skip_special_tokens=True)
        coplot(img_path, words)
        plt.close('all')
        
        