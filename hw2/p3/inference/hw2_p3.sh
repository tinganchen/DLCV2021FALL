wget --max-redirect=20 -O download.zip https://www.dropbox.com/sh/4f2j99pmw3d9v13/AABZPqmVsnIl3w05LMO4ZRWWa?dl=1

unzip download.zip -d p3_best_model


python3 dann_inference.py --tgt_data_path $1 --tgt_data $2 --output_csv $3 --arch dann 

rm download.zip

rm -r p3_best_model