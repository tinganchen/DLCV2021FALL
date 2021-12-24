import pandas as pd
import json


data_df = pd.read_csv('../../hw4_TA/hw4_data/office/train.csv')
        
label_proj = dict()
COUNT = 0

for label in data_df["label"]:
    if label not in label_proj:
        label_proj[label] = COUNT
        COUNT += 1
        

with open('data/class_mapping.json', 'w') as fp:
    json.dump(label_proj, fp)
    