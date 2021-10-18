import os

import utils.common as utils
from utils.options import args
from importlib import import_module

import torch
import torchvision.transforms as transforms

from data import data

#from mean_iou_evaluate import *

import time
import warnings

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))

def main():

    start_epoch = 0
    best_miou = 0.0
    
    # Data loading
    print('=> Loading preprocessed data..')
    
    loader = data.DataLoading(args)
    loader.load()
    
    loader_test = loader.loader_test
    
    # Create model
    print('=> Building model...')
    
    model = import_module('model.models').__dict__[args.model](num_classes = args.num_classes, 
                                                               pretrained = False).to(device)
    
    # Load trained model
    print('=> Loading trained weights...')
    if args.pretrained == 'True':
        # Load pretrained weights
        ckpt = torch.load(args.pretrain_dir + args.pretrain_file, map_location = device)
        state_dict = ckpt[list(ckpt.keys())[0]]
        
        model.load_state_dict(state_dict)
        model = model.to(device)
    
    print('=> Start predicting...')
    s = time.time()
    if args.ground_truth_dir:
        mean_iou = test(args, loader_test, model, 0)
        time.sleep(0.5)
        print_logger.info(f"Best mIOU: {mean_iou:.3f}\n")
    else:
        test(args, loader_test, model, 0)
    
    e = time.time()
    
    print(f'Finished in {e-s:.3f} seconds\n')
    #print(f'Please check the prediction results under:\n"{args.output_dir}"')

def test(args, loader_test, model, epoch):

    # switch to eval mode
    model.eval()
    
    num_iterations = len(loader_test)
    

    try:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    except:
        print(f'Check path exists: {args.output_dir}\n')

        
    with torch.no_grad():
        for i, (inputs, datafiles) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
           
            if args.model == 'fcn_resnet50':
                output = model(inputs)['out'].to(device)
            else:
                output = model(inputs).to(device)

            # output predicted images
            preds = torch.argmax(output, 1).cpu()
 
            rgb_preds = class2img(preds)
                
            for j in range(inputs.size(0)): 
                pil_pred = transforms.ToPILImage()(rgb_preds[j])
                datafile = datafiles[j]
                datafile = datafile[:-4] + '.png'
                pil_pred.save(os.path.join(args.output_dir, datafile))
        '''
        if args.ground_truth_dir:
            mean_iou = mean_iou_score(pred = read_masks(args.test_dataset), 
                                      labels = read_masks(args.output_dir))
 
            return mean_iou*100
            '''
    return

def class2img(preds):
    class2rgb ={0: [0, 1, 1],
                1: [1, 1, 0],
                2: [1, 0, 1], 
                3: [0, 1, 0], 
                4: [0, 0, 1],
                5: [1, 1, 1],
                6: [0, 0, 0]}
    
    rgb_preds = torch.zeros([preds.size(0), 3, 512, 512])
    
    for cls, rgb in class2rgb.items():
        for i in range(preds.size(0)):
            rgb_preds[i, :, preds[i] == cls] = torch.FloatTensor(rgb).unsqueeze(1)
        
    return rgb_preds
        

if __name__ == '__main__':
    main()

