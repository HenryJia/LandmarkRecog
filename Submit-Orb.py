import os
import shutil
from time import time
import pickle
from collections import OrderedDict

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from cv2 import KeyPoint

from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from tqdm import tqdm

from data_utils import grab, sample, read_img
from cv_utils import save_orb, load_orb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

test_data = pd.read_csv('/home/data/LandmarkRetrieval/test.csv')
result_df = pd.DataFrame(index = test_data['id'].tolist(), columns = ['id', 'images'])
print(result_df.head())
print(result_df.shape)

# Setup all the environmental stuff we need
print("Generating Test Dictionary")
test_dict = {}
for idx, row in tqdm(test_data.iterrows(), total = len(test_data)):
    test_dict[row['id']] = row['url']

print("Initialise ORB and the Brute Force Matcher")
t0 = time()
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
print("Done - ", time() - t0)

print("Initialise ThreadPoolExecutor")
t0 = time()
pool = ThreadPoolExecutor(20)
print("Done - ", time() - t0)


def get_orb(img_id):
    # If we have already computed the ORB of this image, load it
    fn = '/home/data/LandmarkRetrieval/orb/' + img_id
    if os.path.exists(fn + '.npy') and os.path.exists(fn + '.kp'):
        try:
            out = load_orb(fn)
            if out is None:
                raise
            return (img_id, out)
        except:
            pass # if we can't load then compute and overwrite

    # Otherwise, compute the ORB
    img = read_img(img_id)
    if img is None:
        return (img_id, None) # If the image doesn't exist don't save anything and return a None
    out = orb.detectAndCompute(img, None)

    # Save the ORB
    save_orb(fn, out)

    return (img_id, out)

if not os.path.exists('/home/data/LandmarkRetrieval/orb'):
    os.makedirs('/home/data/LandmarkRetrieval/orb')

print("Submitting ORB jobs to pool")
futures = [pool.submit(get_orb, k) for k in tqdm(list(test_dict.keys()))]
features = []
print("Waiting for pool to finish")
for feat in tqdm(as_completed(futures), total = len(test_data)):
    features += [feat.result()]
print("Done")

features = dict(features)

def match(id1):
    result_df = pd.DataFrame(columns = ['id', 'images'])
    result_all = dict()

    sample = test_data.sample(frac = 0.1) # Compare every image with 10% of the dataset and take the 100 best matches
    score = OrderedDict()

    try:
        kp1, ds1 = features[id1] # If we can't get the features of the image we're looking at, return nothing
    except:
        return (id1, '')

    for _, row in sample.iterrows():
        id2 = row['id']

        if id1 == id2:
            continue # Skip if we're comparing the same image

        try: # If we can't get the features of the image we're trying to match with or we can't match with it, skip it
            kp2, ds2 = features[id2]

            matches = bf.match(ds1, ds2)

            score_img = 0
            for k in range(min(len(matches), 50)): # Only use the 50 best matches
                score_img += 1.0 / (matches[k].distance + 1e-7)

            score[id2] = score_img

        except:
            continue

    # Sort the matches and take the 100 best
    score = OrderedDict(sorted(score.items(),key = lambda t:t[1]))
    out_str = ''
    for s in list(score.keys())[:min(100, len(score.keys()))]:
        out_str = out_str + s + ' '
    out_str = out_str[:-1]

    return (id1, out_str)

print("Submitting brute force matching jobs to pool")
t0 = time()
for i in range(1):
    print(match(list(test_dict.keys())[i]))
print(time() - t0)
futures = [pool.submit(match, k) for k in tqdm(list(test_dict.keys()))]
print("Waiting for pool to finish")
for out in tqdm(as_completed(futures), total = len(test_data)):
    print("checkpoint")
    result = out.result()
    result_df.loc[result[0]] = list(result)

result_df.to_csv('submission.csv')
print("Done")
