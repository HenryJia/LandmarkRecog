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
mask = np.random.rand(len(train_data)) < 0.2

test_data = train_data[mask]
train_data = train_data[~mask]

train_set = CSVDataset(train_data, '/home/data/LandmarkRetrieval/train/')
test_set = CSVDataset(test_data, '/home/data/LandmarkRetrieval/train/')

# 16 seems to be the maximum batchsize we can do parallel load and train with
train_loader = DataLoader(train_set, batch_size = 16, shuffle = True, num_workers = 6, pin_memory = True)
test_loader = DataLoader(test_set, batch_size = 16, shuffle = True, num_workers = 6, pin_memory = True)

# Build our base model with pretrained weights
classes = int(max(np.max(test_data['landmark_id']), np.max(train_data['landmark_id']))) + 1
net = CombinedNetwork(classes).cuda()
net.load_state_dict(torch.load("network-1.nn"))
torch.save(net.state_dict(), 'network.nn')

criterion = nn.NLLLoss().cuda()
#main_optim, attention_optim = net.get_optims()
main_optim = Adam(net.parameters(), lr = 3e-4)


print('Training')
net.use_attention = True
for epoch in range(5):
    net.load_state_dict(torch.load("network.nn"))

    print('Epoch ', epoch + 1, ', beginning train')
    pb = tqdm(total = len(train_set))
    pb.set_description('Epoch ' + str(epoch + 1))

    train_queue, train_worker = make_queue(train_loader)

    loss_avg = 0 # Keep an exponential running avg
    accuracy_avg = 0
    metric_avg = 0

    net.train(mode = True)
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
    pb = tqdm(total = len(test_set))
    pb.set_description('Epoch ' + str(epoch + 1))

    test_queue, test_worker = make_queue(test_loader)

    loss_avg = 0 # Keep an exponential running avg
    accuracy_avg = 0
    metric_avg = 0

    net.train(mode = False)
    while test_worker.is_alive() or not test_queue.empty():

        data, targets = test_queue.get()
        out, loss, accuracy, metric = test_iter(data, targets, net, criterion)

        loss_avg = 0.9 * loss_avg + 0.1 * loss.data.cpu().numpy()
        accuracy_avg = 0.9 * accuracy_avg + 0.1 * accuracy.data.cpu().numpy()
        metric_avg = 0.9 * metric_avg + 0.1 * metric.data.cpu().numpy()
        pb.update(data.size()[0])
        pb.set_postfix(loss = loss_avg, accuracy = accuracy_avg, gap = metric_avg, queue_empty = test_queue.empty())

    pb.close()
    test_queue.join()
    print('Test loss and accuracy ', (loss_avg, accuracy_avg))

    torch.save(net.state_dict(), 'network.nn')
