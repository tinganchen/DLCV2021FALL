# Usage
    git clone https://github.com/tinganchen/DLCV2021FALL.git
    cd hw1/

# Dataset
    bash ./get_dataset.sh

or 
1. Download from the [link](https://drive.google.com/file/d/1LS4V8r1iBjP6OwqpzLLxXUUB7IvSJ-uh/view?usp=sharing) 
2. Unzip hw1_data.zip as hw1_data/
3. Add to <your_data_path> 

# Training

## Problem 1 ― Image Classification
    cd p1/

### Pretraining 

Pretraining ResNet-110 on CIFAR-100

1. Self-pretrain
    python3 pretrain.py --job dir <your_job_path> --num_classes 100 --num_epochs 100 --train_batch_size 128 --eval_batch_size 100 --momentum 0.9 --lr 0.05 --lr_decay_step 30 --weight_decay 0.0002 --print_freq 100
2. Check the best model under <your_job_path>/checkpoint

or

1. Download from the [model_best.pt](https://drive.google.com/file/d/1Mtz2hvfDawPHLCtV0xWiTt4zbSlYmJqt/view?usp=sharing) 
2. Add to hw1/pretrain/resnet110_cifar100/ 

### Train & Validation

Training & validating the (pretrained) ResNet-110 on hw1_data/p1_data

    python3 main.py --train_dataset `<your_data_path>`/hw1_data/p1_data/train_50 --test_dataset <your_data_path>/hw1_data/p1_data/val_50 --tsne False --model resnet_110 --job_dir experiment/resnet_110/output/ --output_file result.csv --pretrained True --pretrain_dir pretrain/ --pretrain_file resnet110_cifar100/model_best.pt --num_classes 50 --num_epochs 10 --train_batch_size 128 --lr 0.01 --weight_decay 0.0002

## Problem 2 ― Semantic Segmentation

### Evaluation
To evaluate your semantic segmentation model, you can run the provided evaluation script provided in the starter code by using the following command.

    python3 mean_iou_evaluate.py <--pred PredictionDir> <--labels GroundTruthDir>

 - `<PredictionDir>` should be the directory to your predicted semantic segmentation map (e.g. `hw1_data/prediction/`)
 - `<GroundTruthDir>` should be the directory of ground truth (e.g. `hw1_data/validation/`)

Note that your predicted segmentation semantic map file should have the same filename as that of its corresponding ground truth label file (both of extension ``.png``).

### Visualization
To visualization the ground truth or predicted semantic segmentation map in an image, you can run the provided visualization script provided in the starter code by using the following command.

    python3 viz_mask.py <--img_path xxxx_sat.jpg> <--seg_path xxxx_mask.png>

# Submission Rules
### Deadline
2021/10/26 (Tue.) 11:59 PM

### Late Submission Policy
You have a three-day delay quota for the whole semester. Once you have exceeded your quota, the credit of any late submission will be deducted by 30% each day.

Note that while it is possible to continue your work in this repository after the deadline, **we will by default grade your last commit before the deadline** specified above. If you wish to use your quota or submit an earlier version of your repository, please contact the TAs and let them know which commit to grade.

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.


### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the root directory of this repository:
 1.   `hw1_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Submission*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 2.   `hw1_1.sh`  
The shell script file for running your classification model.
 3.   `hw1_2.sh`  
The shell script file for running your semantic segmentation model.

We will run your code in the following manner:

    bash hw1_1.sh $1 $2
where `$1` is the testing images directory (e.g. `test/images/`), and `$2` is the path of folder where you want to output your prediction file (e.g. `test/label_pred/` ). Please do not create the output prediction directory in your bash script or python codes.

    bash hw1_2.sh $1 $2
where `$1` is the testing images directory (e.g. `test/images/`), and `$2` is the output prediction directory for segmentation maps (e.g. `test/label_pred/` ). Please do not create the output prediction directory in your bash script or python codes.

### Packages
This homework should be done using python3.6. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

### Remarks
- If your model is larger than GitHub’s maximum capacity (100MB), you can upload your model to another cloud service (e.g. Dropbox). However, your shell script files should be able to download the model automatically. For a tutorial on how to do this using Dropbox, please click [this link](https://goo.gl/XvCaLR).
- **DO NOT** hard code any path in your file or script, and the execution time of your testing code should not exceed an allowed maximum of **10 minutes**.
- **Please refer to HW1 slides for details about the penalty that may incur if we fail to run your code or reproduce your results.**

# Q&A
If you have any problems related to HW1, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under hw1 FAQ section in FB group
