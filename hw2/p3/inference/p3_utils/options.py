import argparse
import os 
# import pdb

parser = argparse.ArgumentParser(description = 'Quantization')


INDEX = 'inference_0'

'''
tmux, index

'''

METHOD = 'src_tgt' # src_only, tgt_only, src_tgt

SRC = 'svhn' # svhn, mnistm, usps
TGT = 'mnistm' # mnistm, usps, svhn
ARCH = 'dann'

ALPHA = 10 # step fn. slope modification
#LAMBDA2 = 4 # sigmoid scale

PRETRAINED = True


# SEED = 12345 
# 1. training: main.py [cifar-10]
# 2. fine-tune: finetune.py [cifar-10]

  
## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')
parser.add_argument('--src_data_path', type = str, default = f'/home/ta/Documents/110-1/dlcv/hw2_TA/hw2_data/digits/{SRC}/train', help = 'Dataset to train')
parser.add_argument('--src_label_path', type = str, default = f'/home/ta/Documents/110-1/dlcv/hw2_TA/hw2_data/digits/{SRC}/train.csv', help = 'Dataset to train')

parser.add_argument('--tgt_data_path', type = str, default = f'/home/ta/Documents/110-1/dlcv/hw2_TA/hw2_data/digits/{TGT}/train', help = 'Dataset to train')

#parser.add_argument('--tgt_label_path', type = str, default = f'/home/ta/Documents/110-1/dlcv/hw2_TA/hw2_data/digits/{TGT}/train.csv', help = 'Dataset to train')
parser.add_argument('--tgt_label_path', type = str, default = None, help = 'Dataset to train')

parser.add_argument('--method', type = str, default = f'{METHOD}', help = 'The quantization method.')

parser.add_argument('--src_data', type = str, default = f'{SRC}', help = 'The directory where the input data is stored.')
parser.add_argument('--tgt_data', type = str, default = f'{TGT}', help = 'The directory where the input data is stored.')


parser.add_argument('--job_dir', type = str, default = f'experiment/{ARCH}/tsne/{SRC}_{TGT}/{METHOD}/t_{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--output_csv', type = str, default = 'test_pred.csv', help = 'The output csv.') # 'experiments/'
parser.add_argument('--pretrained', action = 'store_true', default = PRETRAINED, help = 'Load pretrained model')


parser.add_argument('--source_dir', type = str, default = 'p3_best_model/', help = 'The directory where the teacher model saved.')
parser.add_argument('--source_file', type = str, default = f'{ARCH}_model_{SRC[0]}{TGT[0]}.pt', help = 'The file the teacher model weights saved as.')

parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')

parser.add_argument('--refine', type = str, default = None, help = 'Path to the model to be fine tuned.') # None
#parser.add_argument('--refine', type = str, default = f'experiment/resnet/t_{T}_mask_{MASK}_sigma_{SIGMA}_lambda_{LAMBDA}_{LAMBDA2}_kd_{KD}_{INDEX}/checkpoint/model_best.pt', help = 'Path to the model to be fine tuned.') # None


## Training

parser.add_argument('--arch', type = str, default = 'resnet', help = 'Architecture of teacher and student')
parser.add_argument('--model', type = str, default = f'resnet50_{ARCH}', help = 'The target model.')

parser.add_argument('--num_epochs', type = int, default = 20, help = 'The num of epochs to train.') # 100
# 1. train: default = 100
# 2. fine_tuned: default = 50

parser.add_argument('--train_batch_size', type = int, default = 64, help = 'Batch size for training.')
parser.add_argument('--eval_batch_size', type = int, default = 100, help = 'Batch size for validation.')
parser.add_argument('--img_size', type = int, default = 28, help = 'Batch size for validation.')

parser.add_argument('--momentum', type = float, default = 0., help = 'Momentum for MomentumOptimizer.')
parser.add_argument('--lr', type = float, default = 0.0001)
# 1. train: default = 0.01
# 2. fine_tuned: default = 5e-2
parser.add_argument('--lr_gamma', type = float, default = 0.1)

parser.add_argument('--lr_decay_step', type = int, default = 30)
parser.add_argument('--lr_decay_steps', type = list, default = [80])
# 1. train: default = 30
# 2. fine_tuned: default = 30

parser.add_argument('--mask_step', type = int, default = 200, help = 'The frequency of mask to update') # 800
parser.add_argument('--weight_decay', type = float, default = 0., help = 'The weight decay of loss.')


parser.add_argument('--random', action = 'store_true', help = 'Random weight initialize for finetune')

parser.add_argument('--keep_grad', action = 'store_true', help = 'Keep gradients of mask for finetune')
parser.add_argument('--alpha', type = float, default = ALPHA, help = 'Modify the approximated slope of step function.')
#parser.add_argument('--lam2', type = float, default = LAMBDA2, help = 'Scale the sigmoid function.')
parser.add_argument('--param', type = float, default = 0.3, help = 'Scale the sigmoid function.')
## Status
parser.add_argument('--print_freq', type = int, default = 500, help = 'The frequency to print loss.')
parser.add_argument('--test_only', action = 'store_true', default = False, help = 'Test only?') 

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

if args.tgt_data == 'mnistm':
    SRC = 'svhn' # svhn, mnistm, usps
    TGT = 'mnistm' # mnistm, usps, svhn
elif args.tgt_data == 'usps':
    SRC = 'mnistm' 
    TGT = 'usps' 
else:
    SRC = 'usps' 
    TGT = 'svhn' 

ARCH = args.arch

args.src_data = f'{SRC}'
args.job_dir = f'experiment/{ARCH}/tsne/{SRC}_{TGT}/{METHOD}/t_{INDEX}/'
args.source_file = f'{ARCH}_model_{SRC[0]}{TGT[0]}.pt'
args.model = f'resnet50_{ARCH}'
