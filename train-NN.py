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
from tqdm import tqdm

from data_utils import grab, sample, read_img, CSVDataset
from models import MainNetwork, FeatureNetwork, FeatureNetwork, FinalNetwork, CombinedNetwork

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train_data = pd.read_csv('/home/data/LandmarkRetrieval/train_clean.csv') # This has just under 100k images
mask = np.random.rand(len(train_data)) < 0.1

test_data = train_data[mask]
train_data = train_data[~mask]

train_set = CSVDataset(train_data, '/home/data/LandmarkRetrieval/train/')
test_set = CSVDataset(test_data, '/home/data/LandmarkRetrieval/train/')

# 32 or 16 seems to be the maximum batchsize we can do parallel load and train with
train_loader = DataLoader(train_set, batch_size = 16, shuffle = True, num_workers = 6, pin_memory = True)
test_loader = DataLoader(test_set, batch_size = 16, shuffle = True, num_workers = 6, pin_memory = True)

# Build our base model with pretrained weights
net = CombinedNetwork(int(max(np.max(test_data['landmark_id']), np.max(train_data['landmark_id']))) + 1).cuda()

criterion = nn.NLLLoss().cuda()
main_optim, attention_optim = net.get_optims()


def load_loop(q, loader):
    for i, (data_cpu, targets_cpu) in enumerate(loader):
        data = data_cpu.cuda(non_blocking = True)
        targets = targets_cpu.cuda(non_blocking = True)
        q.put((data, targets))

print('Training features')
net.use_attention = False
for epoch in range(10):

    train_queue = Queue(maxsize = 10)
    train_worker = Thread(target = load_loop, args=(train_queue, train_loader))
    train_worker.setDaemon(True)
    train_worker.start()

    print('Epoch ', epoch + 1, ', beginning train')
    pb = tqdm(total = len(train_set))
    pb.set_description('Epoch ' + str(epoch + 1))

    loss_avg = 0 # Keep an exponential running avg
    accuracy_avg = 0
    while train_worker.is_alive() or not train_queue.empty():

        data, targets = train_queue.get()
        out = net(data.float() / 255.0)[0]

        loss = criterion(out, targets)
        accuracy = torch.mean((torch.max(out, dim = 1)[1] == targets).float())
        net.zero_grad()
        loss.backward()
        main_optim.step()

        loss_avg = 0.9 * loss_avg + 0.1 * loss.data.cpu().numpy()
        accuracy_avg = 0.9 * accuracy_avg + 0.1 * accuracy.data.cpu().numpy()
        pb.update(data.size()[0])
        pb.set_postfix(loss = loss_avg, accuracy = accuracy_avg, queue_empty = train_queue.empty())

    train_queue.join()
    pb.close()

    torch.save(net.state_dict(), 'network.nn')

    test_queue = Queue(maxsize = 10)
    test_worker = Thread(target = load_loop, args=(test_queue, test_loader))
    test_worker.setDaemon(True)
    test_worker.start()

    loss_avg = 0 # Keep an exponential running avg
    accuracy_avg = 0
    while test_worker.is_alive() or not test_queue.empty():

        data, targets = test_queue.get()
        out = net(data.float() / 255.0)[0]

        loss = criterion(out, targets)
        accuracy = torch.mean((torch.max(out, dim = 1)[1] == targets).float())

        loss_avg = 0.9 * loss_avg + 0.1 * loss.data.cpu().numpy()
        accuracy_avg = 0.9 * accuracy_avg + 0.1 * accuracy.data.cpu().numpy()

    test_queue.join()
    print('Test loss and accuracy ', (loss_avg, accuracy_avg))


print('Training Attention')
net.use_attention = True
for epoch in range(10):

    train_queue = Queue(maxsize = 10)
    train_worker = Thread(target = load_loop, args=(train_queue, train_loader))
    train_worker.setDaemon(True)
    train_worker.start()

    print('Epoch ', epoch + 1, ', beginning train')
    pb = tqdm(total = len(train_set))
    pb.set_description('Epoch ' + str(epoch + 1))

    loss_avg = 0 # Keep an exponential running avg
    accuracy_avg = 0
    while train_worker.is_alive() or not train_queue.empty():

        data, targets = train_queue.get()
        out = net(data.float() / 255.0)[0]

        loss = criterion(out, targets)
        accuracy = torch.mean((torch.max(out, dim = 1)[1] == targets).float())
        net.zero_grad()
        loss.backward()
        attention_optim.step()

        loss_avg = 0.9 * loss_avg + 0.1 * loss.data.cpu().numpy()
        accuracy_avg = 0.9 * accuracy_avg + 0.1 * accuracy.data.cpu().numpy()
        pb.update(data.size()[0])
        pb.set_postfix(loss = loss_avg, accuracy = accuracy_avg, queue_empty = train_queue.empty())

    train_queue.join()
    pb.close()

    torch.save(net.state_dict(), 'network.nn')

    test_queue = Queue(maxsize = 10)
    test_worker = Thread(target = load_loop, args=(test_queue, test_loader))
    test_worker.setDaemon(True)
    test_worker.start()

    loss_avg = 0 # Keep an exponential running avg
    accuracy_avg = 0
    while test_worker.is_alive() or not test_queue.empty():

        data, targets = test_queue.get()
        out = net(data.float() / 255.0)

        loss = criterion(out, targets)
        accuracy = torch.mean((torch.max(out, dim = 1)[1] == targets).float())

        loss_avg = 0.9 * loss_avg + 0.1 * loss.data.cpu().numpy()
        accuracy_avg = 0.9 * accuracy_avg + 0.1 * accuracy.data.cpu().numpy()

    test_queue.join()
    print('Test loss and accuracy ', (loss_avg, accuracy_avg))
