import pandas as pd
import argparse



def main(args):

    acc = evaluate(args.p, args.g)
    print(f'Accuracy: {acc:.3f}%')

def evaluate(pred_csv_path, gt_csv_path):
    pred_csv = pd.read_csv(pred_csv_path)
    gt_csv = pd.read_csv(gt_csv_path)
    
    assert any(pred_csv.columns == gt_csv.columns)
    
    labels = gt_csv['label']
    preds = pred_csv['label']
    
    acc = 0.
    for i in range(len(preds)):
        label = labels[i]
        pred = preds[i]
        if label == pred:
            acc += 1
    
    acc /= len(preds) 
    acc *= 100.
    
    return acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'SSL')

    parser.add_argument('--p', type = str, default = None, help = 'Prediction')
    parser.add_argument('--g', type = str, default = None, help = 'Ground truth')

    args = parser.parse_args()

    main(args)

