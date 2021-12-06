wget --max-redirect=20 -O download.zip https://www.dropbox.com/sh/jzpdhmsdlijr7l8/AAA7m56JW8CPQaA3NofS_B53a?dl=1

unzip download.zip -d best_model

python3 inference.py --test_dataset $1 --output_file $2

rm download.zip

rm -r best_model