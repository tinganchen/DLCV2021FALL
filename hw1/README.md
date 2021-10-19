# Usage
    git clone https://github.com/tinganchen/DLCV2021FALL.git
    cd hw1/

# Dataset
    bash ./get_dataset.sh

or 
1. Download from the [link](https://drive.google.com/file/d/1LS4V8r1iBjP6OwqpzLLxXUUB7IvSJ-uh/view?usp=sharing) 
2. Unzip hw1_data.zip as hw1_data/
3. Add to <your_data_path> 

# Requirements

    pip3 install -r requirements.txt

Can further *pip3 install imageio==2.9.0* for evaluation in the following implementations

# Implementations

## Problem 1 ― Image Classification
    cd p1/

### Pretraining 

Pretraining ResNet-110 on CIFAR-100

1. Self-pretrain

```shell
python3 pretrain.py --job dir <your_job_path> --num_classes 100 --num_epochs 100 --train_batch_size 128 --eval_batch_size 100 --momentum 0.9 --lr 0.05 --lr_decay_step 30 --weight_decay 0.0002 --print_freq 100
2. Check the best model under <your_job_path>/checkpoint
```
or

1. Download pretrained model [model_best.pt](https://drive.google.com/file/d/1Mtz2hvfDawPHLCtV0xWiTt4zbSlYmJqt/view?usp=sharing) 
2. Add to hw1/pretrain/resnet110_cifar100/ 

### Train & Validation

Training & validating the (pretrained) ResNet-110 on hw1_data/p1_data

1.
```shell
python3 main.py --train_dataset <your_data_path>/hw1_data/p1_data/train_50 --test_dataset <your_data_path>/hw1_data/p1_data/val_50 --tsne False --model resnet_110 --job_dir <your_job_path> --output_file <output_csv> --pretrained True --pretrain_dir pretrain/ --pretrain_file resnet110_cifar100/model_best.pt --num_classes 50 --num_epochs 10 --train_batch_size 128 --lr 0.01 --weight_decay 0.0002
```
2. Check trained model *model_best.pt* under <your_job_path>/checkpoint or Download our trained model [here](https://drive.google.com/file/d/1o16upMUqmz4kbnOwZCEbo5iyjAjjHAVV/view?usp=sharing)

3. Copy *model_best.pt* under hw1/p1/best_model/

### Inference

Doing inference under hw1/

```shell
bash hw1_1.sh <test_dataset_dir> <output_csv>
```

Output csv will be in two columns with column names ["image_id", "label"]

### Evaluation

```shell
python3 evaluate.py -p <output_csv> -g <ground_truth_csv>
```

The ground truth csv should be in the same format as <output_csv>, e.g. hw1_data/p1_data/val_gt.csv

## Problem 2 ― Semantic Segmentation
    cd p2/

### Train & Validation

Training & validating the ResNet-50+FCN (VGG-16+FCN32) on hw1_data/p2_data (pretrained model from *torchvision*)

1.
```shell
python3 main.py --train_dataset <your_data_path>/hw1_data/p2_data/train --test_dataset <your_data_path>/hw1_data/p2_data/validation --model fcn_resnet50 --job_dir <your_job_path> --output_dir result.csv --pretrained True --num_classes 7 --num_epochs 10 --train_batch_size 8 --lr 0.005 --lr_decay_step 5 --weight_decay 0.0002
```
*--model VGG_FCN32* replaced for VGG-16+FCN32

2. Check trained model *model_best.pt* under <your_job_path>/checkpoint or Download our trained model here [(ResNet-50+FCN)](https://drive.google.com/file/d/1Db7VYGiQTcmJ_uP7DWKkMMKUlkcU84ZY/view?usp=sharing) / [(VGG-16+FCN32)](https://drive.google.com/file/d/1w8akHOZvSrMiGntN0pz4D5LN9ajXD4Hp/view?usp=sharing)

3. Copy *model_best.pt* under hw1/p2/best_model/

### Inference

Doing inference under hw1/

```shell
bash hw1_2.sh <test_dataset_dir> <output_segmented_image_dir>
```

The mIOU can be evaluated during inference time only unmarked the comments in Line 12 and Line 104-108.

Meanwhile, *--ground_truth_dir <your_ground_truth_path>* should be claimed. Otherwise, the default is *None*, and no mIOU will be evaluated.

### mIOU Evaluation

```shell
python3 mean_iou_evaluate.py -p <output_segmented_image_dir> -g <ground_truth_image_dir>
```

The evaluation in *mean_iou_evaluate.py* ignores the unknown class (class_id: 6).

The evaluation in hw1/p2/utils/common.py considers all classes.


### Visualization
To visualization the ground truth or predicted semantic segmentation map in an image, you can run the provided visualization script provided in the starter code by using the following command.

    python3 viz_mask.py <--img_path xxxx_sat.jpg> <--seg_path xxxx_mask.png>

# Results

Please check the [report](./hw1_d09921014.pdf)
