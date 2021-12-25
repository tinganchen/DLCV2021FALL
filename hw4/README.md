# Usage
    git clone https://github.com/tinganchen/DLCV2021FALL.git
    cd hw4/

# Dataset
    bash ./get_dataset.sh

or 
1. Download from the [link](https://drive.google.com/file/d/1agcf1V8U6inIfw1ZTQI6jWsMEyycPeGM/view?usp=sharing) 
2. Unzip hw4_data.zip as hw4_data/
3. Add to <your_data_path> 

# Requirements

    pip3 install -r requirements.txt


# Implementations

## Problem 1 ― Few-shot Learning (FSL) - Prototypical Net
    cd p1/


### Train & Validation

Training & validating the model on hw4_data/mini/ dataset

1.
```shell
python3 train.py --train_dataset <your_data_path>/hw4_data/mini/train --train_csv <your_data_path>/hw4_data/mini/train.csv --test_dataset <your_data_path>/hw4_data/mini/val --test_csv <your_data_path>/hw4_data/mini/test.csv  --output_file <output_csv> 
```

2. Check trained model *model_best.pt* under <your_job_path>/checkpoint or Download our trained model [here](https://drive.google.com/file/d/1kSal-XAN6zrVRIYRj_Ccmd5MxUhRgkAi/view?usp=sharing)

3. Copy *model_best.pt* under hw4/p1/inference/best_model_p1/

or

```shell
bash hw4_download.sh
```

### Inference

Doing inference under hw4/p1/inference/

```shell
bash hw4_p1.sh <test_csv> <test_dataset_dir> <output_csv>
```

The output csv will be in two columns with same format as hw4/p1/inference/sample.csv

### Evaluation

```shell
python3 eval.py <output_csv> <ground_truth_csv>
```


## Problem 2 ― Self-supervised Learning (SSL) - BYOL [paper](https://arxiv.org/abs/2006.07733)
    cd p2/

### Pretraining 

1. Pretrained model (ResNet-50) on mini-ImageNet can be downloaded [here](https://drive.google.com/file/d/19ME6fOwIrE_994hP2x4pdG9BCPKZid3j/view?usp=sharing)

2. Copy *pretrain_model_SL.pt* under hw4/p1/prtrained/

## Training & Validation

```shell
python3 train_ssl.py 
```

We also compare with supervised learning (SL)...

```shell
python3 train_sl.py 
```


### Inference

Doing inference under hw4/p2/inference

```shell
bash hw4_download.sh
```

```shell
bash hw4_p2.sh <test_csv> <test_dataset_dir> <output_csv>
```




# Results

Please refer to the [report](./hw3_d09921014.pdf)
