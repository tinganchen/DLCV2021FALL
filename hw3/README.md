# Usage
    git clone https://github.com/tinganchen/DLCV2021FALL.git
    cd hw3/

# Dataset
    bash ./get_dataset.sh

or 
1. Download from the [link](https://drive.google.com/file/d/1UgyLoTqrBnpNrCHfNlz3bHsUjq-57qZm/view?usp=sharing) 
2. Unzip hw3_data.zip as hw3_data/
3. Add to <your_data_path> 

# Requirements

    pip3 install -r requirements.txt


# Implementations

## Problem 1 ― Vision Transformer (ViT)
    cd p1/

### Pretraining 

Pretrained model and weights refer to the [github](https://github.com/rwightman/pytorch-image-models/blob/master/timm/)

We save the trained model architecture with the pretrained weights *model.pt* to [here](https://drive.google.com/file/d/1R_t4gmQRNuzRyvF4ysaJPPvzjgozcI6f/view?usp=sharing)

Copy *model.pt* under hw3/inference/best_model/

### Train & Validation

Training & validating the (pretrained) ViT on hw3_data/p1_data

1.
```shell
python3 main.py --train_dataset <your_data_path>/hw3_data/p1_data/train --test_dataset <your_data_path>/hw3_data/p1_data/val --test_only False --model ViT --job_dir <your_job_path> --output_file <output_csv> --pretrained False --num_classes 37 --num_epochs 50 --img_size 384 --lr 0.0005 
```

2. Check trained model *model_best.pt* under <your_job_path>/checkpoint or Download our trained model [here](https://drive.google.com/file/d/1REDkIWx7lR3iAKy3GXilGw7SbycsmK25/view?usp=sharing)

3. Copy *model_best.pt* under hw3/inference/best_model/

### Inference

Doing inference under hw3/inference/

```shell
bash hw3_1.sh <test_dataset_dir> <output_csv>
```

The output csv will be in two columns with column names ["filename", "label"]

### Evaluation

```shell
python3 evaluate.py -p <output_csv> -g <ground_truth_csv>
```

### Visualization
To visualize the correlations of position embeddings & the attention.

```shell
python3 visualize.py --test_dataset <img_dir_path>
```

datafiles in 104 in *visualize.py* can be modified to examine other image attention.

## Problem 2 ― Caption Transformer
    cd p2/

### Pretraining 

Pretrained model and weights refer to the [github](https://github.com/saahiluppal/catr)


### Inference

Doing inference under hw3/

```shell
python3 predict.py --path <image_file_path>
```
<image_file_path>: e.g. <your_data_path>/hw3_data/p2_data/images/bike.jpg



### Visualization
To visualize the attention map and the caption results under ~/hw3_data/p2_data/images/.

```shell
bash hw3_2.sh <your_data_path>/hw3_data/p2_data/images <output_fig_path>
```

# Results

Please refer to the [report](./hw3_d09921014.pdf)
