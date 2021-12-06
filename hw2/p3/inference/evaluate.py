import pandas as pd

import argparse

#from ptflops import get_model_complexity_info # from thop import profile



def main(args):

    acc = evaluate(args.pred_csv, args.gt_csv)
    print(f'Accuracy: {acc:.3f}%')
 
def evaluate(pred_csv, gt_csv):

    
    labels = pd.read_csv(gt_csv)['label']
    preds = pd.read_csv(pred_csv)['label']
    
    acc = 0.
    for i in range(len(preds)):
        label = int(labels[i])
        pred = int(preds[i])
        if label == pred:
            acc += 1
    
    acc /= len(preds) 
    acc *= 100.
    
    return acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'DA')

    parser.add_argument('--pred_csv', type = str, default = None, help = 'Dataset to train')
    parser.add_argument('--gt_csv', type = str, default = None, help = 'Dataset to train')

    args = parser.parse_args()

    main(args)

