import argparse
import os 
# import pdb

parser = argparse.ArgumentParser(description = 'Quantization')


INDEX = 't_inference'


METHOD = 'gan' # 

ARCH = 'resnet'

PRETRAINED = True

# SEED = 12345 
  
## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')
parser.add_argument('--data_path', type = str, default = '../hw2_TA/hw2_data/face', help = 'Dataset to train')
parser.add_argument('--output_train_data_path', type = str, default = f'experiment/{ARCH}/t_{INDEX}/output/train', help = 'Dataset to train')
parser.add_argument('--output_test_data_path', type = str, default = f'experiment/{ARCH}/t_{INDEX}/output/test', help = 'Dataset to train')

parser.add_argument('--method', type = str, default = f'{METHOD}', help = 'The quantization method.')


parser.add_argument('--job_dir', type = str, default = f'experiment/{ARCH}/t_{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--pretrained', action = 'store_true', default = PRETRAINED, help = 'Load pretrained model')


parser.add_argument('--source_dir', type = str, default = 'p1_best_model/', help = 'The directory where the teacher model saved.')
parser.add_argument('--source_file', type = str, default = 'model.pt', help = 'The file the teacher model weights saved as.')

parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')

parser.add_argument('--refine', type = str, default = None, help = 'Path to the model to be fine tuned.') # None
#parser.add_argument('--refine', type = str, default = f'experiment/resnet/t_{T}_mask_{MASK}_sigma_{SIGMA}_lambda_{LAMBDA}_{LAMBDA2}_kd_{KD}_{INDEX}/checkpoint/model_best.pt', help = 'Path to the model to be fine tuned.') # None


## Training

parser.add_argument('--arch', type = str, default = 'resnet', help = 'Architecture of teacher and student')
parser.add_argument('--model', type = str, default = 'resnet50_discriminator', help = 'The target model.')

parser.add_argument('--input_dim', type = int, default = 100, help = 'The num of epochs to train.') # 100
parser.add_argument('--num_epochs', type = int, default = 52, help = 'The num of epochs to train.') # 100
parser.add_argument('--num_noise_vectors', type = int, default = 1000, help = 'The num of noise vectors to generate images.') # 100
# 1. train: default = 100
# 2. fine_tuned: default = 50

parser.add_argument('--train_batch_size', type = int, default = 128, help = 'Batch size for training.')
parser.add_argument('--eval_batch_size', type = int, default = 64, help = 'Batch size for validation.')
parser.add_argument('--img_size', type = int, default = 64, help = 'Batch size for validation.')

parser.add_argument('--momentum', type = float, default = 0.9, help = 'Momentum for MomentumOptimizer.')
parser.add_argument('--lr', type = float, default = 0.0002)
# 1. train: default = 0.01
# 2. fine_tuned: default = 5e-2
parser.add_argument('--lr_gamma', type = float, default = 0.9)

parser.add_argument('--lr_decay_step', type = int, default = 10)
parser.add_argument('--lr_decay_steps', type = list, default = [80])
# 1. train: default = 30
# 2. fine_tuned: default = 30

parser.add_argument('--mask_step', type = int, default = 200, help = 'The frequency of mask to update') # 800
parser.add_argument('--weight_decay', type = float, default = 0., help = 'The weight decay of loss.')


parser.add_argument('--random', action = 'store_true', help = 'Random weight initialize for finetune')

parser.add_argument('--keep_grad', action = 'store_true', help = 'Keep gradients of mask for finetune')
# Status
parser.add_argument('--print_freq', type = int, default = 100, help = 'The frequency to print loss.')
parser.add_argument('--test_only', action = 'store_true', default = False, help = 'Test only?') 

args = parser.parse_args()


