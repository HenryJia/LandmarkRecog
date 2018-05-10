import os

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv('/home/data/LandmarkRetrieval/train_split.csv')
val_data = pd.read_csv('/home/data/LandmarkRetrieval/val_split.csv')

train_ids, train_counts = np.unique(train_data['landmark_id'], return_counts = True)
val_ids, val_counts = np.unique(val_data['landmark_id'], return_counts = True)

train_dict = dict(zip(train_ids.tolist(), train_counts.tolist()))
val_dict = dict(zip(val_ids.tolist(), val_counts.tolist()))

ratios = []
for k in train_dict.keys():
  ratios += [train_dict[k] / (train_dict[k] + val_dict[k])]

print(len(train_ids), len(val_ids))
print(np.mean(ratios))
