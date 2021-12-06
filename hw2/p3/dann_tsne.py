import os

import utils.common as utils
from utils.options import args
#from utils.options import args
from tensorboardX import SummaryWriter
from importlib import import_module

import torch
import torch.nn as nn

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    
    src_loader = import_module('data.data').__dict__['Data'](args,  
                                                             data_path=args.src_data_path, 
                                                             label_path=args.src_label_path)
    
    tgt_loader = import_module('data.data').__dict__['Data'](args,  
                                                             data_path=args.tgt_data_path, 
                                                             label_path=args.tgt_label_path)

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
        
       
    tsne(args, src_data_loader_test, tgt_data_loader_test, model_t, cm = 'rainbow') # 'tab20b'
    
    print('Finish t-SNE.')



def tsne(args, src_loader_test, tgt_loader_test, model_t, cm):
   
    src_embs, src_labels = get_embeddings_labels(src_loader_test, model_t)
    tgt_embs, tgt_labels = get_embeddings_labels(tgt_loader_test, model_t)
    
    embs = src_embs + tgt_embs
    labels = src_labels + tgt_labels
    domains = [0]*len(src_embs) + [1]*len(tgt_embs)
    
    # t-SNE
    ## Color map
    COLOR_MAP = cm
    print('\n=> Preparing t-SNE visualization...')
    cm = plt.get_cmap(f'{COLOR_MAP}')
    
    ## classification result
    NUM_COLORS = 10
    clrs = [cm(1.*color_id/NUM_COLORS) for color_id in range(NUM_COLORS)]
    
    tsne_class = TSNE(perplexity = 30, n_components = 2, init='pca') # n_iter = 300
    class_emb = tsne_class.fit_transform(embs)
    
    for i in range(len(class_emb)):
        plt.scatter(class_emb[i, 0], class_emb[i, 1], 
                    color = clrs[labels[i]], s = 2)
        
    plt.axis('off')
    plt.savefig(f'{os.path.join(args.job_dir, "tsne_class_result.png")}')
    
    ## domain result
    clrs = ['blue', 'red'] # -> ['src', 'tgt']
    
    for i in range(len(class_emb)):
        plt.scatter(class_emb[i, 0], class_emb[i, 1], 
                    color = clrs[domains[i]], s = 2)
        
    plt.axis('off')
    plt.savefig(f'{os.path.join(args.job_dir, "tsne_domain_result.png")}')
    
    return 
    
def get_embeddings_labels(loader_test, model_t):
    
    # switch to eval mode
    model_t.eval()
    
  
    embs = []
    labels = []

    with torch.no_grad():
        for i, (inputs, targets, data_file) in enumerate(loader_test, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            labels.extend(list(targets.cpu().numpy()))
  
            preds, domain = model_t(inputs, alpha=0)
            
            batch_embs = model_t.emb.reshape([inputs.size(0), -1]).cpu().detach().numpy()
            for batch_emb in batch_embs:
                embs.append(batch_emb)
    
    return embs, labels



if __name__ == '__main__':
    main()

