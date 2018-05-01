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

# 14950 distinct landmarks in recog

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
orb = cv2.ORB_create(nfeatures = 100)

# find the keypoints and descriptors with ORB
# kp, des = orb.detectAndCompute(img,None)
def getOrb(img):
    return orb.detectAndCompute(img, None)
print('Start Orb')
t0 = time.time()
#features = [feat for feat in pool.map(getOrb, train_imgs, chunksize = 20)]
features = [getOrb(img) for img in tqdm(train_imgs)]
print("Orb time: ", time.time() - t0)

max_vis = 100

# features = [orb.detectAndCompute(img, None) for img in train_imgs]

#FLANN_INDEX_LSH = 6
#index_params= dict(algorithm = FLANN_INDEX_LSH,
                   #table_number = 6, # 12
                   #key_size = 12,     # 20
                   #multi_probe_level = 1) #2
#search_params = dict(checks=50)   # or pass empty dictionary

#flann = cv2.FlannBasedMatcher(index_params, search_params)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

if os.path.exists('./vis'):
    shutil.rmtree('./vis')
os.makedirs('./vis')

permutation = np.zeros(max_vis, dtype = np.int32)
score = np.zeros(max_vis, dtype = np.float32)
for i, (kp1, ds1) in tqdm(enumerate(features[:max_vis]), total = max_vis):
    best_match = None

    for j, (kp2, ds2) in enumerate(features):
        if j == i:
            continue # Skip if we're comparing the same image
        # Match every image in the trainset with every other image in the trainset
        try:
            #matches = flann.knnMatch(ds1, ds2, k = 2)
            matches = bf.match(ds1, ds2)
        except:
            continue
        matches = sorted(matches, key = lambda x:x.distance)
        # Select the good matches and use them to calculate a score for the overall match of an image
        score_img = 0
        for k in range(min(len(matches), 50)):
            #if len(matches[k]) != 2:
                #continue
            #if matches[k][0].distance < 0.7 * matches[k][1].distance:
                # score is the sum of reciprocal of distances of all matched points
                #score_img += 1.0 / (matches[k][0].distance + 1e-7)
            score_img += 1.0 / (matches[k].distance + 1e-7)

        if score_img > score[i]:
            best_match = matches
            score[i] = score_img
            permutation[i] = j

    plt.figure()
    img_show = None
    img_show = cv2.drawMatches(train_imgs[i], kp1, train_imgs[permutation[i]], features[permutation[i]][0], best_match[:10], img_show, flags = 2)
    plt.imshow(img_show)
    plt.title(score[i])
    plt.savefig('./vis/' + str(i) + '.png', bbox_inches = 'tight')

print(permutation)
print(score)

#if os.path.exists('./vis'):
    #shutil.rmtree('./vis')
#os.makedirs('./vis')

#i = 0
#while (i < max_vis):
    #if (score[i] == 0):
        #i += 1
        #continue
    #plt.figure(figsize = (10, 5))
    #plt.subplot(1, 2, 1)
    #plt.title(score[i])
    #plt.imshow(train_imgs[i])
    #plt.subplot(1, 2, 2)
    #plt.imshow(train_imgs[int(permutation[i])])
    #print(i)
    #plt.savefig('./vis/' + str(i) + '.png', bbox_inches = 'tight')
    #i += 1
