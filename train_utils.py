import os

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

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

def gap(out, targets): # The Global Average Precision metric used for the competition
    prob, category = torch.max(out, dim = 1)
    _, indices = torch.sort(prob, descending = True)
    correct = (category == targets).float()[indices]
    precision = torch.cumsum(correct, dim = 0) / (torch.arange(correct.shape[0]).cuda() + 1).float()
    return torch.mean(precision * correct)

def test_iter(data, targets, net, criterion):
    out = net(data.float() / 255.0)[0]

    loss = criterion(out, targets)
    accuracy = torch.mean((torch.max(out, dim = 1)[1] == targets).float())
    metric = gap(out, targets)

    return out, loss, accuracy, metric


def train_iter(data, targets, net, criterion, optim):
    out, loss, accuracy, metric = test_iter(data, targets, net, criterion)

    net.zero_grad()
    loss.backward()
    optim.step()

    return out, loss, accuracy, metric


def load_loop(q, loader, submission = False):
    for i, (data_cpu, targets_cpu) in enumerate(loader):
        data = data_cpu.cuda(non_blocking = True)
        if submission:
            targets = targets_cpu # They're in fact the ids
        else:
            targets = targets_cpu.cuda(non_blocking = True)
        q.put((data, targets))
        q.task_done()


def make_queue(loader, maxsize = 10, submission = False):
    data_queue = Queue(maxsize = maxsize)
    worker = Thread(target = load_loop, args=(data_queue, loader, submission))
    worker.setDaemon(True)
    worker.start()

    return data_queue, worker


def split_validation(all_df, ratio_train):
    labels = np.unique(all_df['landmark_id'])

    print("Splitting into validation and test set, keeping consistent label ratios")
    train_indices = []
    val_indices = []
    for l in tqdm(labels.tolist()):
        l_indices = list(all_df[all_df['landmark_id'] == l].index)
        tl_indices, vl_indices = train_test_split(l_indices, test_size = 1 - ratio_train)
        train_indices += tl_indices
        val_indices += vl_indices

    return all_df.loc[train_indices], all_df.loc[val_indices]
