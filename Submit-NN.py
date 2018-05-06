import os
import math

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet, resnet50

from queue import Queue
from threading import Thread
import tqdm
from tqdm import tqdm
tqdm.monitor_interval = 0

from data_utils import grab, sample, read_img, CSVDataset
from models import MainNetwork, FeatureNetwork, FeatureNetwork, FinalNetwork, CombinedNetwork
from train_utils import train_iter, test_iter, make_queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train_data = pd.read_csv('/home/data/LandmarkRetrieval/train_clean.csv') # This has just under 100k images
test_data = pd.read_csv('/home/data/LandmarkRetrieval/test_clean.csv')[:100]
result_recog_df = pd.DataFrame(index = np.arange(len(test_data)), columns = ['id', 'landmarks'])

test_set = CSVDataset(test_data, '/home/data/LandmarkRetrieval/test/', submission = True)

test_loader = DataLoader(test_set, batch_size = 16, shuffle = False, num_workers = 6, pin_memory = True)

net = CombinedNetwork(int(np.max(train_data['landmark_id'])) + 1).cuda()
net.use_attention = False
net.train(mode = False)
net.load_state_dict(torch.load("network.nn"))

test_queue, test_worker = make_queue(test_loader, submission = True)

pb = tqdm(total = len(test_set))

i = 0
while test_worker.is_alive() or not test_queue.empty():

    data, idx = test_queue.get()
    out = net(data.float() / 255.0)[0]
    for j in range(out.shape[0]):
        prop, category = torch.max(out[j], dim = 0)
        category = int(category.data.cpu().numpy())
        prop = math.exp(prop.data.cpu().numpy())

        result_recog_df.iloc[i] = {'id' : idx[j], 'landmarks' : str(category) + ' ' + str(prop)}
        i += 1

    pb.update(data.size()[0])

pb.close()
test_queue.join()
print(result_recog_df)
result_recog_df.to_csv("submission_recog.csv", index = False)
