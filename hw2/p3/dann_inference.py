import os
import pandas as pd
import utils.common as utils
from utils.options import args
#from utils.options import args
from tensorboardX import SummaryWriter
from importlib import import_module

import torch



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
        
       
    if args.tgt_label_path is not None:
        acc = inference(args, tgt_data_loader_test, model_t)
        print(f'Accuracy: {acc:.2f}%')
    else:
        inference(args, tgt_data_loader_test, model_t)
    print(f'\nFinish inference.\nPlease see the output file in {args.output_csv}')
 
 
def inference(args, loader_test, model_t):

    # switch to eval mode
    model_t.eval()

    
    preds = []
    data_files = []

    with torch.no_grad():
        for i, (inputs, _, data_file) in enumerate(loader_test, 1):
            
            inputs = inputs.to(device)

            
            logits, domain = model_t(inputs, alpha=0)
            
            _, pred = logits.topk(1, 1, True, True)
            
            preds.extend(list(pred.reshape(-1).cpu().detach().numpy()))
            data_files.extend(list(data_file))

    # output prediction results
    output = dict()
    output['image_name'] = data_files
    output['label'] = preds
    
    output = pd.DataFrame.from_dict(output)
    output.to_csv(args.output_csv, index = False)
    
    if args.tgt_label_path is not None:
        test_tgt_labels = args.tgt_label_path.replace('train', 'test')
        
        labels = pd.read_csv(test_tgt_labels)['label']
        acc = 0.
        for i in range(len(preds)):
            label = int(labels[i])
            pred = int(preds[i])
            if label == pred:
                acc += 1
        
        acc /= len(preds) 
        acc *= 100.
        return acc
    
    return


if __name__ == '__main__':
    main()

