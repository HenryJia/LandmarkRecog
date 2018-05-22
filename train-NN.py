import os
from math import gcd

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import ResNet, resnet50

from queue import Queue
from threading import Thread
import tqdm
from tqdm import tqdm
tqdm.monitor_interval = 0

from data_utils import grab, sample, read_img, CSVDataset
from models import MainNetwork, FeatureNetwork, FeatureNetwork, FinalNetwork, CombinedNetwork
from train_utils import train_iter, test_iter, make_queue, split_validation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train_data = pd.read_csv('/home/data/LandmarkRetrieval/train_split.csv')
val_data = pd.read_csv('/home/data/LandmarkRetrieval/val_split.csv')
classes = int(max(np.max(val_data['landmark_id']), np.max(train_data['landmark_id']))) + 1

print('Training samples: ', len(train_data), '\n', train_data.head())
print('Validation samples: ', len(val_data), '\n', val_data.head())

print('Setting up Dataloaders')
train_set = CSVDataset(train_data, '/home/data/LandmarkRetrieval/train/')
val_set = CSVDataset(val_data, '/home/data/LandmarkRetrieval/train/')

# 16 seems to be the maximum batchsize we can do parallel load and train with
train_loader = DataLoader(train_set, batch_size = 16, shuffle = True, num_workers = 6, pin_memory = True)
val_loader = DataLoader(val_set, batch_size = 16, shuffle = True, num_workers = 6, pin_memory = True)

# Build our base model with pretrained weights
net = CombinedNetwork(classes).cuda()

criterion = nn.NLLLoss().cuda()
#main_optim, attention_optim = net.get_optims()
main_optim = Adam(net.parameters(), lr = 3e-4)


print('Training')
net.use_attention = True
for epoch in range(3, 10):

    print('Epoch ', epoch + 1, ', beginning train')
    pb = tqdm(total = len(train_set))
    pb.set_description('Epoch ' + str(epoch + 1))

    train_queue, train_worker = make_queue(train_loader)

    loss_avg = 0 # Keep an exponential running avg
    accuracy_avg = 0
    metric_avg = 0

    net.train()
    while train_worker.is_alive() or not train_queue.empty():

        data, targets = train_queue.get()
        out, loss, accuracy, metric = train_iter(data, targets, net, criterion, main_optim)

        loss_avg = 0.9 * loss_avg + 0.1 * loss.data.cpu().numpy()
        accuracy_avg = 0.9 * accuracy_avg + 0.1 * accuracy.data.cpu().numpy()
        metric_avg = 0.9 * metric_avg + 0.1 * metric.data.cpu().numpy()
        pb.update(data.size()[0])
        pb.set_postfix(loss = loss_avg, accuracy = accuracy_avg, gap = metric_avg, queue_empty = train_queue.empty())

    train_queue.join()
    pb.close()

    print('Epoch ', epoch + 1, ', beginning test')
    pb = tqdm(total = len(val_set))
    pb.set_description('Epoch ' + str(epoch + 1))

    val_queue, val_worker = make_queue(val_loader)

    loss_avg = 0 # Keep an exponential running avg
    accuracy_avg = 0
    metric_avg = 0

    net.eval()
    while val_worker.is_alive() or not val_queue.empty():

        data, targets = val_queue.get()
        out, loss, accuracy, metric = test_iter(data, targets, net, criterion)

        loss_avg = 0.9 * loss_avg + 0.1 * loss.data.cpu().numpy()
        accuracy_avg = 0.9 * accuracy_avg + 0.1 * accuracy.data.cpu().numpy()
        metric_avg = 0.9 * metric_avg + 0.1 * metric.data.cpu().numpy()
        pb.update(data.size()[0])
        pb.set_postfix(loss = loss_avg, accuracy = accuracy_avg, gap = metric_avg, queue_empty = val_queue.empty())

    pb.close()
    val_queue.join()
    print('Test loss and accuracy ', (loss_avg, accuracy_avg))

    torch.save(net.state_dict(), 'network-epoch' + str(epoch) + '.nn')
