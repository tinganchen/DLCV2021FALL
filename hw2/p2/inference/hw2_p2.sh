wget --max-redirect=20 -O download.zip https://www.dropbox.com/sh/tkxyvitlep6di5f/AABYfo0aGLGB9L4hg_0XUqUwa?dl=1
unzip download.zip -d p2_best_model


python3 acgan_inference.py --output_test_data_path $1 

rm download.zip

rm -r p2_best_model