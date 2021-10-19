CURRENT_PATH=$(pwd)

# change work space
cd ./p2

# create directory
mkdir best_model/
mkdir best_model/fcn_resnet50/

mkdir pretrain/
mkdir pretrain/resnet110_cifar100

# Download model weights
wget https://www.dropbox.com/s/9iqdgdwm0a0qhtk/model_best.pt?dl=1
mv model_best.pt?dl=1 best_model/fcn_resnet50/model_best.pt

wget https://www.dropbox.com/s/sb4jbd99xwj4fny/model_best.pt?dl=1
mv model_best.pt?dl=1 pretrain/resnet110_cifar100/model_best.pt

# inference
python3 inference.py --bash_path $CURRENT_PATH --test_dataset $1 --output_dir $2 --test_only True --pretrained True

cd ..


# evaluate with mIOU
# python3 mean_iou_evaluate.py -g $1 -p $2

