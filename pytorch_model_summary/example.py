# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch.nn as nn
import torchvision.models as tv_models
from attention_network_def import *
from model_summary import model_summary

import pdb
pdb.set_trace()

class CCustomNet(nn.Module):
    def __init__(self):
        super(CCustomNet, self).__init__()

        a = list()
        a.append(('conv_0', nn.Conv2d(3, 16, 3, 2, 1)))
        a.append(('bn_0', nn.BatchNorm2d(16)))
        a.append(('relu_0', nn.ReLU(inplace=True)))
        b = list()
        b.append(('conv_1', nn.Conv2d(16, 32, 3, 1, 1)))
        b.append(('bn_1', nn.BatchNorm2d(32)))
        b.append(('relu_1', nn.ReLU(inplace=True)))
        c = list()
        c.append(('conv_1', nn.Conv2d(32, 32, 3, 1, 1)))

        a = nn.Sequential(OrderedDict(a))
        b = nn.Sequential(OrderedDict(b))
        c = nn.Sequential(OrderedDict(c))
        self.d = nn.Sequential(OrderedDict(a=a, b=b, c=c))

    def forward(self, x):
        x = self.d(x)
        return x


def main():

    model = Feat_Net(in_channels=3, img_dim=112, net_mode='irse', p=1, t=2, r=1, attention_stages=(3, 6, 2))
    #model.fc = MarginCosineProduct(512, 93419, m=0.4, scale=64)
    input_size = (3, 112, 112)

    model_summary(model, input_size, query_granularity=1)


if __name__ == "__main__":
    main()
