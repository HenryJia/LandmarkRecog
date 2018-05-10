import os, sys
import math

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tqdm
from tqdm import tqdm
tqdm.monitor_interval = 0

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

recog = pd.read_csv('./submission.csv')

result_retrieve = pd.DataFrame(index = np.arange(len(recog)), columns = ['id', 'images'])

recog_dict = dict()
for j, row in tqdm(recog.iterrows(), total = len(recog)):
    recog_dict[row['id']] = row['landmarks'].split(' ')[0]

for i, rowi in tqdm(recog.iterrows(), total = len(recog)):
    id_current = rowi['id']
    cat_current = rowi['landmarks'].split(' ')[0]

    result_str = ''
    for k in recog_dict.keys():
        if recog_dict[k] == cat_current:
            result_str += k + ' '

    result_retrieve.loc[i] = {'id' : id_current, 'images' result_str}

print(result_retrieve)
result_retrieve.to_csv('submission_retrieve.csv', index = False)
