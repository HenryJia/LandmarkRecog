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
from models import MainNetwork, FeatureNetwork, FeatureNetwork, FinalNetwork, CombinedNetwork, BoostedNearestClass
from train_utils import train_iter, test_iter, make_queue, split_validation, gap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#all_data = pd.read_csv('/home/data/LandmarkRetrieval/train_clean.csv') # This has just under 100k images
#train_data, val_data = split_validation(all_data, 0.8)
train_data = pd.read_csv('/home/data/LandmarkRetrieval/train_split.csv').sample(100000)
val_data = pd.read_csv('/home/data/LandmarkRetrieval/val_split.csv')
print('Training samples: ', len(train_data), '\n', train_data.head())
print('Validation samples: ', len(val_data), '\n', val_data.head())

train_set = CSVDataset(train_data, '/home/data/LandmarkRetrieval/train/')
val_set = CSVDataset(val_data, '/home/data/LandmarkRetrieval/train/')

# 16 seems to be the maximum batchsize we can do parallel load and train with
val_loader = DataLoader(val_set, batch_size = 4, shuffle = True, num_workers = 6, pin_memory = True)

# Build our base model with pretrained weights
#classes = int(max(np.max(val_data['landmark_id']), np.max(train_data['landmark_id']))) + 1
classes = 14951
net = CombinedNetwork(classes).cuda()
net.use_attention = True
net.load_state_dict(torch.load('./archived_nn/network-epoch1.nn'))

model = BoostedNearestClass(net, classes)
model.train_means(train_set)
model.save()
model.load()
model.train_booster(train_set, max_samples = int(1e4))
model.save()
model.load()

print("Validating")
accuracy_nearest_avg = 0
accuracy_net_avg = 0
accuracy_all_avg = 0
metric_avg = 0
pb = tqdm(total = len(val_set))
val_queue, val_worker = make_queue(val_loader)
while val_worker.is_alive() or not val_queue.empty():
    data, targets = val_queue.get()

    out, out_nearest, out_net = model.run_all(data)

    accuracy_nearest = torch.mean((torch.max(out_nearest, dim = 1)[1] == targets).float())
    accuracy_net = torch.mean((torch.max(out_net, dim = 1)[1] == targets).float())
    accuracy_all = torch.mean((torch.max(out, dim = 1)[1] == targets).float())
    metric = gap(out, targets)

    accuracy_nearest_avg = 0.9 * accuracy_nearest_avg + 0.1 * accuracy_nearest.data.cpu().numpy()
    accuracy_net_avg = 0.9 * accuracy_net_avg + 0.1 * accuracy_net.data.cpu().numpy()
    accuracy_all_avg = 0.9 * accuracy_all_avg + 0.1 * accuracy_all.data.cpu().numpy()
    metric_avg = 0.9 * metric_avg + 0.1 * metric.data.cpu().numpy()
    pb.set_postfix(accuracy_nearest_avg = accuracy_nearest_avg,
                   accuracy_net_avg = accuracy_net_avg, accuracy_all_avg = accuracy_all_avg,
                   gap = metric_avg)
    pb.update(data.size()[0])

print('Cluster validation accuracy: ', accuracy_nearest_avg)
print('Network validation accuracy: ', accuracy_net_avg)
print('Boosted validation accuracy: ', accuracy_all_avg)
