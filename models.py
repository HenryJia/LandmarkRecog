import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet, resnet50


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
