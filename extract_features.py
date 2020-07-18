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
import cv2
import numpy as np
import scipy.io as sio
import time

import struct
import sys
import sklearn
from sklearn.preprocessing import normalize
from collections import OrderedDict

from ImageDataset import *
from network_def import *
from funcs import *

import pdb
pdb.set_trace()
args = None


## add some function
def parse_args():
    parser = argparse.ArgumentParser(description='Extract face features')
    # general
    parser.add_argument('--gpu-id', type=str, default='0,1', help='gpu id used for inference')
    parser.add_argument('--batch-size', type=int, default=200, help='batch size of test images when each inference')
    parser.add_argument('--block', type=str, default='', help='basic block for net')
    parser.add_argument('--input-size', type=int, default=112, help='input size')
    parser.add_argument('--ckpt-path', type=str, required=True, default='', help='pretrained model path')
    parser.add_argument('--root-path', type=str, required=True, default='', help='root directory of test images')
    parser.add_argument('--save-path', type=str, required=True, default='', help='save directory of feature')
    args = parser.parse_args()
    return args


def write_mat_to_bin(file_name,feat_mat):
    np.asarray(feat_mat,dtype='float').tofile(file_name)


def inference(args):
    ########### devices ##############
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 
    num_dev = torch.cuda.device_count()
    if num_dev >=1:
        print("Let's use ", num_dev , 'GPUs!')
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = True
    ########## Model #################
    model = Feat_Net(in_channels=3, img_dim=args.input_size, net_mode=args.block, p=1, t=2, r=1, attention_stages=(3, 6, 2)) 
    model = nn.DataParallel(model)
    # model_params = model.state_dict()
    # for k,v in model_params.items():
    #     print(k)
    weight_params = OrderedDict()
    if os.path.isfile(args.ckpt_path):
        print ("=> loading checkpoint '{}'".format(args.ckpt_path))
        checkpoint = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
        if 'train_attentionIR362.tmp' in args.ckpt_path:
            for k,v in checkpoint.items():
                #print(k)
                name = 'module.feat_net'+k[6:]
                weight_params[name] = v
            model.load_state_dict(weight_params)
        else:
            model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    ######### test set ##############

    imgdicts = search_dir(args.root_path)
    batch_size = args.batch_size * num_dev
    test_loader_dict = {}
    for key in imgdicts.keys():
        ## construct loader
        test_loader = torch.utils.data.DataLoader(
            test_dataloader(root=args.root_path, save_path=args.save_path, dir_name=key, imglist= '/data/datas/recognize/icartoon_face/personai_icartoonface_rectest/icartoonface_rectest_det.txt',
                         transform=transforms.Compose([transforms.Resize([112,112]),transforms.ToTensor()])),
            batch_size=batch_size, shuffle=False
        )
        test_loader_dict[key] = test_loader

    ######### inference ##############
    for key in test_loader_dict.keys():
        
        feat_all_addFlip = np.zeros([len(imgdicts[key])*2, 512], dtype='float32')
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader_dict[key]):
                input_var = Variable(images.cuda())
                target_var = Variable(labels.cuda())

                features = model(input_var)   
                feat_data = features.data.cpu()
                feat_norm = F.normalize(feat_data).numpy()

                feat_all_addFlip[batch_idx*batch_size: batch_idx*batch_size+feat_norm.shape[0]] = feat_norm

                if batch_idx % 10 == 0:
                    print('%s  ' % (time.strftime('%Y-%m-%d %H:%M:%S'), ) + str(batch_idx))

            end_time = time.time()
            print('Total time = %f'%(end_time-start_time))
            feat_all = (feat_all_addFlip[::2] + feat_all_addFlip[1::2])
            feat_all = sklearn.preprocessing.normalize(feat_all)
            write_mat_to_bin(os.path.join(args.save_path,key+".bin"),feat_all)

def main():
    global args
    args = parse_args()
    print(args)
    inference(args)

if __name__ == "__main__":
    main()

