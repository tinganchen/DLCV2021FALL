import argparse
import os 

parser = argparse.ArgumentParser()

PRETRAINED = 'True'
INDEX = 'test_0'

# SEED = 12345 
# 1. training: main.py [cifar-10]
# 2. fine-tune: finetune.py [cifar-10]


## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')


parser.add_argument('--train_dataset', type = str, default = os.getcwd() + '/../../hw4_TA/hw4_data/mini/train', help = 'Dataset to train')
parser.add_argument('--train_csv', type = str, default = os.getcwd() + '/../../hw4_TA/hw4_data/mini/train.csv', help = 'Dataset to train')
parser.add_argument('--val_dataset', type = str, default = os.getcwd() + '/../../hw4_TA/hw4_data/mini/val', help = 'Dataset to validate or test')
parser.add_argument('--val_csv', type = str, default = os.getcwd() + '/../../hw4_TA/hw4_data/mini/val.csv', help = 'Dataset to train')

parser.add_argument('--test_dataset', type = str, default = os.getcwd() + '/../../hw4_TA/hw4_data/mini/val', help = 'Dataset to validate or test')
parser.add_argument('--test_csv', type = str, default = os.getcwd() + '/../../hw4_TA/hw4_data/mini/val.csv', help = 'Dataset to train')
parser.add_argument('--testcase_csv', type = str, default = os.getcwd() + '/../../hw4_TA/hw4_data/mini/val_testcase.csv', help = 'Dataset to train')

parser.add_argument('--output_csv', type = str, default = 'output.csv', help = 'Dataset to train')

parser.add_argument('--distance_metric', type = str, default = 'euclidean', help = 'euclidean, cosine, parametric')

parser.add_argument('--stage', type = str, default = 'test', help = 'train, val, test') 

parser.add_argument('--model', type = str, default = 'convnet', help = 'The model.')

parser.add_argument('--job_dir', type = str, default = f'experiment/fsl_conv/test_{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'

parser.add_argument('--pretrained', type = str, default = PRETRAINED, help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--pretrain_dir', type = str, default = '', help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--pretrain_file', type = str, default = 'best_model/model_best.pt', help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')

parser.add_argument('--refine', type = str, default = None, help = 'Path to the model to be fine tuned.') # None
#parser.add_argument('--refine', type = str, default = f'experiment/mobile/t_{T}_mask_{MASK}_sigma_{SIGMA}_lambda_{LAMBDA}_{LAMBDA2}_kd_{KD}_{INDEX}/checkpoint/model_best.pt', help = 'Path to the model to be fine tuned.') # None

parser.add_argument('--shot', type=int, default=1) # 1, 5, 10
parser.add_argument('--query', type=int, default=15)
parser.add_argument('--train-way', type=int, default=30)
parser.add_argument('--test-way', type=int, default=5)

## Training
parser.add_argument('--num_epochs', type = int, default = 200, help = 'The num of epochs to train.') # 100


#parser.add_argument('--train_batch_size', type = int, default = 32, help = 'Batch size for training.')
#parser.add_argument('--eval_batch_size', type = int, default = 32, help = 'Batch size for validation.')

parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--lr_decay_step',type = int, default = 20)

## Status
parser.add_argument('--print_freq', type = int, default = 40, help = 'The frequency to print loss.')


args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))
