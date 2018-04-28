import time, os
import shutil

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from multiprocessing.pool import Pool, ThreadPool

from tqdm import tqdm

from data_utils import grab, sample

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train_data = pd.read_csv('/home/data/LandmarkRetrieval/index.csv')
test_data = pd.read_csv('/home/data/LandmarkRetrieval/test.csv')
submission = pd.read_csv('/home/data/LandmarkRetrieval/sample_submission.csv')

print("Training data size",train_data.shape)
print("test data size",test_data.shape)

def sample(i):
    img = None
    while(img is None):
        try:
            img = grab(train_data.sample(1).iloc[0, 1])
        except:
            img = None

    return img

max_samples = 200
pool = ThreadPool(20)
train_imgs = [imgs for imgs in tqdm(pool.imap_unordered(sample, range(max_samples)), total = max_samples)]

# Initiate ORB detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with ORB
print('Start sift')
t0 = time.time()
features = [sift.detectAndCompute(img, None) for img in train_imgs]
print("Orb time: ", time.time() - t0)

max_vis = 100


# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

permutation = np.zeros(max_vis, dtype = np.int32)
score = np.zeros(max_vis, dtype = np.float32)
for i, (kp1, ds1) in tqdm(enumerate(features[:max_vis]), total = max_vis):
    for j, (kp2, ds2) in enumerate(features):
        if j == i:
            continue # Skip if we're comparing the same image
        # Match every image in the trainset with every other image in the trainset
        try:
            m = flann.knnMatch(ds1, ds2, k = 2)
        except:
            continue
        # Select the good matches and use them to calculate a score for the overall match of an image
        score_img = 0
        for k in range(len(m)):
            if len(m[k]) != 2:
                continue
            if m[k][0].distance < 0.7 * m[k][1].distance:
                # score is the sum of reciprocal of distances of all matched points
                score_img += 1.0 / (m[k][0].distance + 1e-7)

        if score_img > score[i]:
            score[i] = score_img
            permutation[i] = j

print(permutation)
print(score)

if os.path.exists('./vis'):
    shutil.rmtree('./vis')
os.makedirs('./vis')

i = 0
while (i < max_vis):
    if (score[i] == 0):
        i += 1
        continue
    plt.figure(figsize = (10, 5))
    plt.subplot(1, 2, 1)
    plt.title(score[i])
    plt.imshow(train_imgs[i])
    plt.subplot(1, 2, 2)
    plt.imshow(train_imgs[int(permutation[i])])
    print(i)
    plt.savefig('./vis/' + str(i) + '.png', bbox_inches = 'tight')
    i += 1
