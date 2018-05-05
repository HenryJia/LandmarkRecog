import os

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
test_data = pd.read_csv('/home/data/LandmarkRetrieval/test_clean.csv')

result_recog_df = pd.DataFrame(index = test_data['id'].tolist(), columns = ['id', 'images'])

test_set = CSVDataset(test_data, '/home/data/LandmarkRetrieval/test/', submission = True)

test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 6, pin_memory = True)

net = CombinedNetwork(int(np.max(train_data['landmark_id'])) + 1).cuda()
net.use_attention = False
net.train(mode = False)
net.load_state_dict(torch.load("network.nn"))

test_queue, test_worker = make_queue(test_loader)

pb = tqdm(total = len(test_set))

out = []
while test_worker.is_alive() or not test_queue.empty():

    data, _ = test_queue.get()
    out += [net(data.float() / 255.0)[0]]

    pb.update(data.size()[0])

pb.close()
test_queue.join()

for i in range(len(result_recog_df)):
    out_cpu = torch.max(out[i], dim = 1).data.cpu().numpy()
    result_str = str(out_cpu[0] + ' ' + out_cpu[1])

    result_recog_df.iloc[i]['landmark'] = result_str
