from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.onnx


import numpy as np

import sys
import sklearn
from sklearn.preprocessing import normalize
from collections import OrderedDict


from network_def import *

import pdb
pdb.set_trace()


args = None

def parse_args():
    parser = argparse.ArgumentParser(description='export onnx model')
    parser.add_argument('--gpu-id', type=str, default='0', help='gpu id used for inference')
    parser.add_argument('--block', type=str, default='irse', help='block type of backbone')
    parser.add_argument('--input-size', type=int, default=112, help='input size of network')
    parser.add_argument('--torch-model', type=str, required=True, default='', help='torch model path')
    parser.add_argument('--onnx-model', type=str, required=True, default='', help='onnx model directory')
    args = parser.parse_args()
    return args



def export_onnx(args):
    ########### devices ##############
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 
    num_dev = torch.cuda.device_count()
    if num_dev >=1:
        print("Let's use ", num_dev , 'GPUs!')

    ########## Model #################
    model = Feat_Net(in_channels=3, img_dim=args.input_size, net_mode=args.block, p=1, t=2, r=1, attention_stages=(3, 6, 2)) 
    model = nn.DataParallel(model)

    if os.path.isfile(args.torch_model):
        print ("=> loading checkpoint '{}'".format(args.torch_model))
        checkpoint = torch.load(args.torch_model)
        model.load_state_dict(checkpoint)
    model.train(False)
    x = torch.randn(1,3,112,112,device=torch.device('cuda'))
    output = torch.onnx._export(model, x, args.onnx_model, export_params=True)#, opset_version=11


def main():
    global args
    args = parse_args()
    print(args)
    export_onnx(args)

if __name__ == "__main__":
    main()
