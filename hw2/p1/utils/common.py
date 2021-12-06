from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path

import os

import torch
import logging
import IS 
import FID
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils.options import args

device = torch.device(f"cuda:{args.gpus[0]}")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = Path(args.job_dir)
        self.ckpt_dir = self.job_dir / 'checkpoint'
        self.run_dir = self.job_dir / 'run'

        if args.reset:
            os.system('rm -rf ' + args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)
        
        config_dir = self.job_dir / 'config.txt'
        with open(config_dir, 'w') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
    
    def save_model(self, state, epoch, is_best):
        save_path = f'{self.ckpt_dir}/model_{epoch}.pt'
        # print('=> Saving model to {}'.format(save_path))
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pt')

def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt = '%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class GAN_Dataset(Dataset):
    def __init__(self, filepath):
        self.figsize = 64
        self.images = []
        self.file_list = os.listdir(filepath)
        self.file_list.sort()

        print("Load file from :" ,filepath)
        for i, file in enumerate(self.file_list):
            print("\r%d/%d" %(i,len(self.file_list)),end = "")
            img = Image.open(os.path.join(filepath, file)).convert('RGB')
            self.images.append(img)
        
        print("")
        print("Loading file completed.")
        
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.num_samples = len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index])

    def __len__(self):
        return self.num_samples

def is_score(out_folder):
    train_dataset = GAN_Dataset(filepath = out_folder)
    score, _ = IS.inception_score(train_dataset, cuda=True, batch_size=32, 
                                  resize=True, splits=10)
    return score

def fid_score(out_folder, gt_folder):
    score = FID.calculate_fid_given_paths([out_folder, gt_folder], 
                                          batch_size = 50,
                                          device = device,
                                          dims = 2048)
    return score