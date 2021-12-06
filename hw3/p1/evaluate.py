import pandas as pd
import argparse

def calculate_accuracy(pred_csv, gt_csv):
    pred = pd.read_csv(pred_csv)
    gt = pd.read_csv(gt_csv)
    
    acc = 0.
    for i in range(gt.shape[0]):
        image_id, label_true = gt.iloc[i]
        image_id, label_pred = pred[pred['filename'] == image_id].iloc[0]
        if label_true == label_pred:
            acc += 1
    acc /= gt.shape[0]
    print(f'\nAccuracy: {acc*100.:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gt_csv', help='ground truth csv', type=str)
    parser.add_argument('-p', '--pred_csv', help='prediction csv', type=str)
    args = parser.parse_args()

    calculate_accuracy(args.pred_csv, args.gt_csv)
