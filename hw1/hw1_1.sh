cd ./p1

mkdir best_model/

wget https://www.dropbox.com/s/x4guurxmcvhtpsp/model_best.pt

mv model_best.pt best_model/model_best.pt

python3 inference.py --test_dataset $1 --output_file $2 --test_only True --tsne False --pretrained True --ground_truth False


# evaluate
# python3 evaluate.py -p $2 -g $3

cd ..

