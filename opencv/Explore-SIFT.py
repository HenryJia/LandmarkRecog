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
from cv_utils import save_feat, load_feat

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

print("Initialise SIFT and the Bag Of Words Clustering")
t0 = time()
descriptor = cv2.xfeatures2d.SIFT_create()
bow_train = cv2.BOWKMeansTrainer(1024)
bow_extract = cv2.BOWImgDescriptorExtractor(descriptor, descriptor)
print("Done - ", time() - t0)

print("Initialise ThreadPoolExecutor")
t0 = time()
pool = ThreadPoolExecutor(20)
print("Done - ", time() - t0)


def get_descriptor(img_id):
    # If we have already computed the SIFT of this image, load it
    fn = '/home/data/LandmarkRetrieval/descriptor/' + img_id + '.sift.npy'
    if os.path.exists(fn):
        try:
            out = np.load(fn)
            if out is None:
                raise
            return (img_id, out)
        except:
            pass # if we can't load then compute and overwrite

    # Otherwise, compute the SIFT
    img = read_img(img_id)
    if img is None:
        return (img_id, None) # If the image doesn't exist don't save anything and return a None
    out = descriptor.compute(img, None) # We only need the features and not the keypoints

    # Save the SIFT
    np.save(fn, out)

    return (img_id, out)

if not os.path.exists('/home/data/LandmarkRetrieval/orb'):
    os.makedirs('/home/data/LandmarkRetrieval/orb')

print("Submitting SIFT jobs to pool")
futures = [pool.submit(get_orb, k) for k in tqdm(list(test_dict.keys()))]
print("Waiting for pool to finish")
for feat in tqdm(as_completed(futures), total = len(test_data)):
    out = feat.result()
    bow_train.add(out[1])
print("Done")

print("Cluster the features to get our vocabulary")
t0 = time()
voc = bow_train.cluster()
bow_extract.setVocabulary(voc)
print("Done - ", time() - t0)
print("Bag Of Words Vocabulary", np.shape(voc))

def get_histograms(img_id):
    # If we have already computed the SIFT of this image, load it
    fn = '/home/data/LandmarkRetrieval/descriptor/' + img_id + '.bow.npy'
    if os.path.exists(fn):
        try:
            out = np.load(fn)
            if out is None:
                raise
            return (img_id, out)
        except:
            pass # if we can't load then compute and overwrite

    # Otherwise, compute the SIFT
    img = read_img(img_id)
    if img is None:
        return (img_id, None) # If the image doesn't exist don't save anything and return a None
    out = descriptor.bow_extract(img, None) # We only need the features and not the keypoints

    # Save the SIFT
    np.save(fn, out)

    return (img_id, out)


print("Submitting SIFT jobs to pool")
futures = [pool.submit(get_orb, k) for k in tqdm(list(test_dict.keys()))]
features = []
print("Waiting for pool to finish")
for feat in tqdm(as_completed(futures), total = len(test_data)):
    out = feat.result()
    bow_train.add(out[1])
    features += [out]
print("Done")

features = dict(features)

def match(id1):
    result_df = pd.DataFrame(columns = ['id', 'images'])
    result_all = dict()

    sample = test_data.sample(frac = 0.01) # Compare every image with 10% of the dataset and take the 100 best matches
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
            matches = sorted(matches, key = lambda x:x.distance) # Only use the 50 best matches
            for k in range(min(len(matches), 50)):
                score_img += 1.0 / (matches[k].distance + 1e-7)

            score[id2] = score_img

        except:
            continue

    # Sort the matches and take the 100 best, note higher the score is better so we sort in descending order
    score = OrderedDict(sorted(score.items(), key = lambda t:t[1], reverse = True))
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
    result = out.result()
    result_df.loc[result[0]] = list(result)

result_df.to_csv('submission.csv')
print("Done")
