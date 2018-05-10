import sys, os

from concurrent.futures import ThreadPoolExecutor, as_completed, wait

import tqdm
from tqdm import tqdm
tqdm.monitor_interval = 0

import numpy as np
import pandas as pd
from scipy.misc import imread
import cv2

if len(sys.argv) != 4:
    print('Syntax: {} <in_file.csv> <out_file.csv> <directory>'.format(sys.argv[0]))
    sys.exit(0)

(in_file, out_file, directory) = sys.argv[1:]

in_df = pd.read_csv(in_file)
out_df = pd.DataFrame(index = np.arange(len(in_df)), columns = list(in_df))
print(in_df)
print(out_df)

def exists(i, row):
    if os.path.exists(directory + row['id'] + '.jpg'):
        try:
            img = cv2.imread(directory + row['id'] + '.jpg')
            return i, row
        except:
            #os.remove(directory + row['id'] + '.jpg')
            return None

pool = ThreadPoolExecutor(5)

futures = [pool.submit(exists, i, row) for i, row in tqdm(in_df.iterrows(), total = len(in_df))]

for row in tqdm(as_completed(futures), total = len(in_df)):
    out = row.result()
    if out:
        i, row = out
        out_df.loc[i] = row


#for i, row in tqdm(in_df.iterrows(), total = len(in_df)):
    #if os.path.exists(directory + row['id'] + '.jpg'):
        #try:
            #img = imread(directory + row['id'] + '.jpg')
            #out_df.loc[i] = row
        #except:
            #os.remove(directory + row['id'] + '.jpg')
            #continue

out_df = out_df.dropna(how = 'any')

print(out_df)
out_df.to_csv(out_file, index = False)
