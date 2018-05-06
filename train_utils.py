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


def test_iter(data, targets, net, criterion):
    out = net(data.float() / 255.0)[0]

    loss = criterion(out, targets)
    accuracy = torch.mean((torch.max(out, dim = 1)[1] == targets).float())

    return out, loss, accuracy


def train_iter(data, targets, net, criterion, optim):
    out, loss, accuracy = test_iter(data, targets, net, criterion)

    net.zero_grad()
    loss.backward()
    optim.step()

    return out, loss, accuracy


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
