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

test_data = pd.read_csv('/home/data/LandmarkRetrieval/test.csv').sample(n = 1000)
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

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

descriptor = cv2.xfeatures2d.SIFT_create(nfeatures = 100)
bow_train = cv2.BOWKMeansTrainer(512)
bow_extract = cv2.BOWImgDescriptorExtractor(descriptor, flann)
print("Done - ", time() - t0)

print("Initialise ThreadPoolExecutor")
t0 = time()
pool = ThreadPoolExecutor(10)
print("Done - ", time() - t0)


def get_descriptor(img_id):
    # If we have already computed the SIFT of this image, load it
    fn = '/home/data/LandmarkRetrieval/sift/' + img_id + '.npy'
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = descriptor.detectAndCompute(img, None)[1].astype(np.float32) # We only need the features and not the keypoints

    # Save the SIFT
    np.save(fn, out)

    return (img_id, out)

if not os.path.exists('/home/data/LandmarkRetrieval/sift'):
    os.makedirs('/home/data/LandmarkRetrieval/sift')

print("Submitting SIFT jobs to pool")
futures = [pool.submit(get_descriptor, k) for k in tqdm(list(test_dict.keys()))]
print("Waiting for pool to finish")
for feat in tqdm(as_completed(futures), total = len(test_data)):
    out = feat.result()
    if out[1] is not None:
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
    fn = '/home/data/LandmarkRetrieval/bow/' + img_id + '.npy'
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = bow_extract.compute(img, descriptor.detect(img)).astype(np.float32) # We only need the features and not the keypoints

    # Save the SIFT
    np.save(fn, out)

    return (img_id, out)


if not os.path.exists('/home/data/LandmarkRetrieval/bow'):
    os.makedirs('/home/data/LandmarkRetrieval/bow')

print("Submitting BOW jobs to pool")
futures = [pool.submit(get_histograms, k) for k in tqdm(list(test_dict.keys()))]
features = []
print("Waiting for pool to finish")
for feat in tqdm(as_completed(futures), total = len(test_data)):
    features += [feat.result()]
print("Done")


print("Create feature dictionary")
features = dict(features)
print("Done - ", time() - t0)


def match(id1):
    result_df = pd.DataFrame(columns = ['id', 'images'])
    result_all = dict()
    #sample = test_data.sample(frac = 0.01) # Compare every image with 10% of the dataset and take the 100 best matches
    sample = test_data
    score = OrderedDict()
    try:
        ds1 = features[id1] # If we can't get the features of the image we're looking at, return nothing
    except:
        return (id1, '')
    for _, row in sample.iterrows():
        id2 = row['id']
        #print(id2)
        if id1 == id2:
            continue # Skip if we're comparing the same image
        try: # If we can't get the features of the image we're trying to match with or we can't match with it, skip it
            ds2 = features[id2]
            score[id2] = np.mean((ds1 - ds2) ** 2)
        except:
            continue
    # Sort the matches and take the 100 best, note higher the score is better so we sort in descending order
    score = OrderedDict(sorted(score.items(), key = lambda t:t[1]))
    #print(score)
    out_str = ''
    for s in list(score.keys())[:min(100, len(score.keys()))]:
        out_str = out_str + s + ' '
    out_str = out_str[:-1]
    return (id1, out_str)

#print(match(list(features.keys())[1]))
#exit()

if os.path.exists('./vis'):
    shutil.rmtree('./vis')
os.makedirs('./vis')

print("Submitting  matching jobs to pool")
t0 = time()
for i in range(1):
    print(match(list(test_dict.keys())[i]))

print(time() - t0)
futures = [pool.submit(match, k) for k in tqdm(list(test_dict.keys()))]
print("Waiting for pool to finish")
j = 0
for out in tqdm(as_completed(futures), total = len(test_data)):
    result = out.result()
    #if result[1] == '':
        #continue
    #matched = result[1].split(' ')
    #plt.figure(figsize = (10, 20))
    #plt.subplot(1, 2, 1)
    #plt.imshow(read_img(result[0]))
    #plt.subplot(1, 2, 2)
    #plt.imshow(read_img(matched[0]))
    #plt.savefig('./vis/' + str(j) + '.png', bbox_inches = 'tight')
    result_df.loc[result[0]] = list(result)
    #j += 1

result_df.to_csv('submission.csv')
print("Done")
