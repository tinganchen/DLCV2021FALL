# Usage
    git clone https://github.com/tinganchen/DLCV2021FALL.git
    cd hw2/

# Dataset
    bash ./get_dataset.sh

or 
1. Download from the [link](https://drive.google.com/file/d/1SEhOw-9lN8Vao5E5MCjJnitQBqKBO53S/view?usp=sharing) 
2. Unzip hw2_data.zip as hw2_data/
3. Add to <your_data_path> 

# Requirements

    pip3 install -r requirements.txt


# Implementations

## Problem 1 ― GAN to generate human faces
    cd p1/

### Pretraining 

Pretrained weights can be downloaded [here](https://drive.google.com/file/d/1r9fnO0tloxCfYpYcobFV89CXUUGjfuiq/view?usp=sharing)

Copy *model_final.pth* to hw2/p1/pretrained/

### Train & Validation

Training & validating the (pretrained) GAN on hw2_data/p2_data

1.
```shell
python3 main.py --data_path <your_data_path>/hw2_data/p1_data/face --output_test_data_path <generated_test_data_path> --source_dir pretrained/ --source_file model_final.pth
```

2. Check trained model *model_best.pt* under <your_job_path>/checkpoint or Download our trained model named *model.pt* [here](https://drive.google.com/file/d/114c2ewuxYtxvvXulRlnbn9YnYfBBJUKx/view?usp=sharing)

3. Copy *model.pt* under hw2/p1/best_model/

### Inference

Doing inference under hw2/p1/inference

```shell
bash hw2_p1.sh <generated_test_data_path>
```

### Evaluation & Visualization

```shell
python3 visualization.py --data_path <your_data_path>/hw2_data/p1_data/face --source_dir best_model/ --source_file model.pt --output_test_data_path <generated_test_data_path>
```

## Problem 2 ― ACGAN to generate digit numbers
    cd p2/

### Pretraining

Self-pretain the discriminator

```shell
python3 pretrain.py --data_path <your_data_path>/hw2_data/digits/mnistm/train --label_path <your_data_path>/hw2_data/digits/mnistm/train.csv --output_test_data_path <generated_test_data_path> --pretrained False --lr 0.01
```

Check trained model *model_best.pt* under <your_job_path>/checkpoint or Download our trained model named *model_d.pt* [here](https://drive.google.com/file/d/168upVTErMenJDNMicqegM7ChjpZpTDHU/view?usp=sharing)

Copy *model_d.pt* under hw2/p2/pretrained/


### Train & Validation

Training & validating ACGAN on hw2_data/p2_data/digits/mnistm

Evaluate by classifier *Classifier.pth* downloaded [here](https://drive.google.com/file/d/1BDeP24VQJZuNdoAEtvxpnJnxpAShLxpt/view?usp=sharing)

1.
```shell
python3 main.py --data_path <your_data_path>/hw2_data/digits/mnistm/train --label_path <your_data_path>/hw2_data/digits/mnistm/train.csv --output_test_data_path <generated_test_data_path> --pretrained True --source_dir pretrained/ --source_file model_d.pt  --lr 0.01 --classifer_model Classifier.pth
```

2. Check trained model *model_best.pt* under <your_job_path>/checkpoint or Download our trained model [here](https://drive.google.com/file/d/1zYn4RTR394rR0LRVlv9QDj-6MHopavnu/view?usp=sharing)

3. Copy *model_best.pt* under hw2/p2/best_model/

### Inference

Doing inference under hw2/p2/inference/

```shell
bash hw2_p2.sh <generated_test_data_path>
```

### Visualization
To visualization the generated digit data. 

```shell
python3 visualization.py --data_path <your_data_path>/hw2_data/digits/mnistm/train --output_test_data_path <generated_test_data_path> --pretrained True --source_dir best_model/ --source_file model_best.pt 
```

### TODO
Our ACGAN can generate digit images that the [classifier](./p2/digit_classifier.py) can discriminate the digits.
However, the image style is hardly similar to the original data. 
We consider that it may be the pretrained discriminator has strong discrimination performance.
We will try to use discriminator without being pretrained.


## Problem 3 ― Unsupervised Domain Adaptation (DANN)
    cd p3/


### Train & Validation

Training & validating DANN on hw2_data/p2_data/digits/mnistm, svhn, usps

There are three tasks for <SRC>-><TGT> (using source data to predict target data)

Evaluate by classifier *Classifier.pth* downloaded [here](https://drive.google.com/file/d/1BDeP24VQJZuNdoAEtvxpnJnxpAShLxpt/view?usp=sharing)

1.
```shell
python3 main.py --data_path <your_data_path>/hw2_data/digits/mnistm/train --label_path <your_data_path>/hw2_data/digits/mnistm/train.csv --output_test_data_path <generated_test_data_path> --pretrained True --source_dir pretrained/ --source_file model_d.pt  --lr 0.01 --classifer_model Classifier.pth
```

2. Check trained model *model_best.pt* under <your_job_path>/checkpoint or Download our trained model [here](https://drive.google.com/file/d/1zYn4RTR394rR0LRVlv9QDj-6MHopavnu/view?usp=sharing)

3. Copy *model_best.pt* under hw2/p2/best_model/

### Inference

Doing inference under hw2/p2/inference/

```shell
bash hw2_p2.sh <generated_test_data_path>
```

### Visualization
To visualization the generated digit data. 

```shell
python3 visualization.py --data_path <your_data_path>/hw2_data/digits/mnistm/train --output_test_data_path <generated_test_data_path> --pretrained True --source_dir best_model/ --source_file model_best.pt 

# Results

Please refer to the [report](./hw2_d09921014.pdf)
