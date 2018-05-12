from collections import OrderedDict

import numpy as np

from train_utils import train_iter, test_iter, make_queue, split_validation, gap

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.models import ResNet, resnet50

from sklearn.linear_model import LogisticRegression

import tqdm
from tqdm import tqdm
tqdm.monitor_interval = 0

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MainNetwork(ResNet):
    def __init__(self, base, **kwargs):
        super(MainNetwork, self).__init__(Bottleneck, [3, 4, 6, 3], **kwargs)
        self.base = base

        for m, b in zip(self.parameters(), base.parameters()):
            m.data = b.data

        self.fc = None
        self.avgpool = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class FeatureNetwork(nn.Module):
    def __init__(self):
        super(FeatureNetwork, self).__init__()

        self.conv1 = nn.Conv2d(2048, 512, kernel_size = 1, bias = True)
        self.conv2 = nn.Conv2d(512, 512, kernel_size = 3, padding = 2, bias = True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        return self.conv2(x)


class FinalNetwork(nn.Module):
    def __init__(self, classes):
        super(FinalNetwork, self).__init__()

        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout1(x)
        x = F.elu(self.fc1(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc2(x), dim = 1)


class CombinedNetwork(nn.Module):
    def __init__(self, classes):
        super(CombinedNetwork, self).__init__()
        self.base = resnet50(pretrained = True)
        self.main = MainNetwork(self.base)
        self.attention = FeatureNetwork()
        self.feature = FeatureNetwork()
        self.final = FinalNetwork(classes)

        self.use_attention = True

    def forward(self, x):
        out_main = self.main(x)
        out_feat = self.feature(out_main)

        if self.use_attention:
            out_attn = self.attention(out_main)
            out_attn = out_attn.view(out_attn.shape[0], out_attn.shape[1], -1) # Flatten the spatial dimension to do a softmax on
            out_attn = F.softmax(out_attn, dim = 2)
            out_attn = out_attn.view_as(out_feat)

            out_merged = out_attn * out_feat
            out_merged = torch.sum(torch.sum(out_merged, dim = -1), dim = -1)
        else:
            out_merged = torch.mean(torch.mean(out_feat, dim = -1), dim = -1)

        out_final = self.final(out_merged)

        results = [out_final, out_merged, out_feat]
        if self.use_attention:
            results += [out_attn]
        return results

    def get_optims(self):
        main_params = list(self.main.parameters()) + list(self.feature.parameters()) + list(self.final.parameters())
        return Adam(main_params, lr = 3e-4), Adam(self.attention.parameters())


# Boost a model with a nearest neighbour/class on the feature vectors
class BoostedNearestClass(object): # Note this is NOT a PyTorch model so the normal pytorch serialisation will NOT work
    def __init__(self, net, classes):
        self.net = net
        self.classes = classes
        self.class_statistics = {}
        self.booster = nn.Linear(2, 1).cuda() # Just use default parameters for now


    def train(self, train_set, max_class = 100, max_samples_boost = 100):
        print("Computing features for each class in the trainset")
        pb = tqdm(total = len(train_set))
        train_loader = DataLoader(train_set, batch_size = 16, shuffle = True, num_workers = 6, pin_memory = True)
        train_queue, train_worker = make_queue(train_loader)

        class_feat = {}
        self.net.eval()
        while train_worker.is_alive() or not train_queue.empty():

            data, targets = train_queue.get()

            data_selected = []
            targets_selected = []
            for i in range(targets.shape[0]):
                if i not in class_feat or class_feat[i].shape[0] < max_class:
                    data_selected += [data[i]]
                    targets_selected += [targets[i]]

            data = torch.stack(data_selected, dim = 0)
            targets = torch.stack(targets_selected, dim = 0)

            out_final, out_merged, out_feat, out_attn = self.net(data.float() / 255.0)
            for i in range(out_final.shape[0]):
                prob, category = torch.max(out_final[i], dim = 0)
                category = int(category.data.cpu().numpy())

                if category in class_feat:
                    class_feat[category] = torch.cat([class_feat[category], out_merged[None, i]], dim = 0).data
                else:
                    class_feat[category] = out_merged[None, i].data
            pb.update(data.size()[0])

        train_queue.join()
        pb.close()

        print("Computing centroids")
        for k in tqdm(class_feat):
            if len(class_feat[k]) == 1: # We can't really compute a standard deviation so set it to 1
                self.class_statistics[k] = (class_feat[k][0], torch.ones(class_feat[k][0].shape).cuda())
            else:
                self.class_statistics[k] = (torch.mean(class_feat[k], dim = 0), torch.std(class_feat[k], dim = 0))

        idx = list(self.class_statistics.keys())
        stat = self.class_statistics.values()
        mu, sigma = zip(*stat)

        idx = torch.LongTensor(idx).cuda()
        mu = torch.stack(mu, dim = 0)
        sigma = torch.stack(sigma, dim = 0)
        self.class_statistics = [idx, [mu, sigma]]

        print("Fitting the booster")
        criterion = nn.NLLLoss()
        booster_optim = Adam(self.booster.parameters(), lr = 3e-4)

        loss_avg = 0 # Keep an exponential running avg
        accuracy_avg = 0
        metric_avg = 0

        # We don't need very much data since we have literally 2 datapoints
        sampler = SubsetRandomSampler(np.random.permutation(len(train_set))[:max_samples_boost].tolist())
        train_loader = DataLoader(train_set, batch_size = 16, sampler = sampler, num_workers = 6, pin_memory = True)
        train_queue, train_worker = make_queue(train_loader)


        pb = tqdm(total = max_samples_boost)
        while train_worker.is_alive() or not train_queue.empty():
            data, targets = train_queue.get()

            out_booster, out_net = self.run_all(data)

            loss = criterion(out_booster, targets)
            accuracy = torch.mean((torch.max(out_booster, dim = 1)[1] == targets).float())
            metric = gap(out_booster, targets)

            self.booster.zero_grad()
            loss.backward()
            booster_optim.step()

            loss_avg = 0.9 * loss_avg + 0.1 * loss.data.cpu().numpy()
            accuracy_avg = 0.9 * accuracy_avg + 0.1 * accuracy.data.cpu().numpy()
            metric_avg = 0.9 * metric_avg + 0.1 * metric.data.cpu().numpy()
            pb.update(data.size()[0])
            pb.set_postfix(loss = loss_avg, accuracy = accuracy_avg, gap = metric_avg)

        pb.close()
        train_queue.join()


    def run_neighbours(self, data):
        out_net, out_merged, out_feat, out_attn = self.net(data.float() / 255.0)

        idx, (mu, sigma) = self.class_statistics
        distance = torch.sqrt(torch.sum((out_merged.unsqueeze(1) - mu) ** 2 / sigma ** 2, dim = 2))
        _, min_distance = torch.min(distance, dim = 1)

        out_nearest = torch.zeros_like(out_net)
        for i in range(min_distance.shape[0]):
            out_nearest[i, idx[min_distance[i]]] = 1

        return out_nearest, out_net


    def run_all(self, data):
        out_nearest, out_net = self.run_neighbours(data)
        out_nearest_onehot = torch.zeros_like(out_net)[:, :, None]
        out_net = torch.exp(out_net)[:, :, None] # Get the actual probabilities

        in_booster = torch.cat([out_nearest_onehot, out_net], dim = 2)
        out_booster = F.log_softmax(torch.squeeze(self.booster(in_booster)), dim = 1)

        return out_booster, out_net


    def save(self):
        torch.save(self.class_statistics[0], 'idx.pytorch')
        torch.save(self.class_statistics[1][0], 'mu.pytorch')
        torch.save(self.class_statistics[1][1], 'sigma.pytorch')

        torch.save(self.booster.state_dict(), 'booster.nn')


    def load(self):
        idx = torch.load('idx.pytorch')
        mu = torch.load('mu.pytorch')
        sigma = torch.load('sigma.pytorch')
        self.class_statistics = [idx, [mu, sigma]]

        self.booster.load_state_dict(torch.load('booster.nn'))
