wget --max-redirect=20 -O download.zip https://www.dropbox.com/sh/dedh31836mdbkqq/AAB6Osk6NwW5QJyycm3QzR7Qa?dl=1
unzip download.zip -d p1_best_model


python3 gan_inference.py --output_test_data_path $1 

rm download.zip

rm -r p1_best_model