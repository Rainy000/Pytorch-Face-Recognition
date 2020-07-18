from __future__ import print_function
import argparse
from collections import OrderedDict
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from ImageDataset import *
from funcs import *
from network_def import *
from mmd import *

import pdb
pdb.set_trace()


args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train face identification net')
    parser.add_argument('--block', type=str, default='irse', help='block type')
    parser.add_argument('--loss', type=str, default='softmax', help='loss function type')
    parser.add_argument('--gpu-id', type=str, default='0,1,2,3', help='gpu id used for train')
    parser.add_argument('--batch-size', type=int, default=48, help='batch size of trainset on each gpu')
    parser.add_argument('--source-num-class', type=int, default=93419, help='num of trainset id')
    parser.add_argument('--feat-scale', type=int, default=30, help='feature scales for train')
    parser.add_argument('--cos-m', type=float, default=0.0, help='margin for cos-softmax')
    parser.add_argument('--factor', type=float, default=5.0, help='lambda factor for regular-loss')
    parser.add_argument('--input-size', type=int, default=112, help='input size of model')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for sgd')
    parser.add_argument('--lr-step', type=str, default='', help='lr step')
    parser.add_argument('--max-step', type=int, default=10, help='maximum epoch')
    parser.add_argument('--resume', dest='resume_switch', default=False, help='if finetuning from pretrained model', action='store_true')
    parser.add_argument('--start-epoch', type=int, default=0, help=' resume epoch')
    parser.add_argument('--feat-model', type=str, default='', help='feat model path')
    parser.add_argument('--src-model', type=str, default='', help='src model path')
    parser.add_argument('--save-period', type=int, default=1, help='period of save checkpoint')
    parser.add_argument('--save-path', type=str, default='', help='path for model save')
    parser.add_argument('--save-name', type=str, default='', help='model save name')
    parser.add_argument('--log-dir', type=str, default='', help='train log directory')
    parser.add_argument('--source-dir', type=str, default='', help='source train set root directory')
    parser.add_argument('--target-dir', type=str, default='', help='target train set root directory')
    args = parser.parse_args()
    return args


def train_net(args):
    ###
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    ### device set 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        pass
        '''
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = False
        '''
    num_dev = torch.cuda.device_count()
    if num_dev >=1:
        print("Let's use ", num_dev , 'GPUs!')
    ### parameter set
    block = args.block
    lr = args.lr
    lr_step = []
    src_num_class = args.source_num_class
    margin = args.cos_m
    factor = args.factor
    scale = args.feat_scale
    for step in args.lr_step.strip().split(","):
        lr_step.append(int(step))
    num_epoch = args.max_step
    input_size = args.input_size
    batch_size = args.batch_size* num_dev
    start_epoch = args.start_epoch
    feat_path = args.feat_model
    src_path = args.src_model
    save_period = args.save_period
    save_path = args.save_path
    log_dir = args.log_dir
    save_name = args.save_name
    source_dir = args.source_dir
    target_dir = args.target_dir
    ### log set
    logger = Logger(log_dir)

    ### model construct 
    model = Feat_Net(in_channels=3, img_dim=input_size, net_mode=block, p=1, t=2, r=1, attention_stages=(3, 6, 2)) 
    if args.loss == 'cosine_margin':
        model_fc = MarginCosineProduct(in_features=512, out_features=num_class, scale=scale, m=margin)
    elif args.loss == 'arc_margin':
        model_fc = ArcMarginProduct(in_features=512, out_features=num_class, s=scale, m=margin, easy_margin=False)
    elif args.loss == 'sphere':
        model_fc = SphereProduct(in_features=512, out_features=num_class, m=4)
    elif args.loss == 'regular':
        src_model_fc = LMRegularProduct(in_features=512,out_features=src_num_class,scale=scale, m=margin, factor=factor)
    elif args.loss == 'softmax': 
        src_model_fc = torch.nn.Linear(512, src_num_class, bias=False)

        
    else:
        print('args.loss must be one of [cosine_margin, arc_margin, sphere]')

    model = model.to(device)
    model = nn.DataParallel(model)
    src_model_fc = src_model_fc.to(device)
    src_model_fc = nn.DataParallel(src_model_fc)

    ### finetuning 
    if args.resume_switch and os.path.isfile(feat_path):
        logger("=> loading checkpoint '{}'".format(feat_path))
        checkpoint = torch.load(feat_path,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint,strict=False)
        if os.path.isfile(src_path):
            logger("=> loading checkpoint '{}'".format(src_path))
            src_fc_checkpoint = torch.load(src_path, map_location=torch.device('cpu'))
            src_model_fc.load_state_dict(src_fc_checkpoint)



    ### optimizer and scheduler set
    optimizer = optim.SGD([{'params':model.parameters(),'lr':lr*1},{'params':src_model_fc.parameters(),'lr':lr*10}],
                                 lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=False)#{'params':model.parameters()}, 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_step, gamma=0.1)

    ### transform dict
    transform_dicts = {
        'train':transforms.Compose([
            transforms.Resize(112),
            #transforms.RandomCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
         ]),
        'test':transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),
         ])
    }
    ### source and target image dataloader
    source_dataset=None
    target_dataset=None


    if os.path.exists(source_dir):
        source_dataset = folder_dataloader(root=source_dir,shuffle=False, transform=transform_dicts['train'])
    if os.path.exists(target_dir):
        target_dataset = folder_dataloader(root=target_dir,shuffle=False, transform=transform_dicts['train'])

    source_train_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,drop_last=True)
    target_train_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,drop_last=True)

    ### criterion
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    mmd_criterion = MMD_loss()
    mmd_criterion.to(device)

    batch_time = AverageMeter()
    load_time = AverageMeter()
    source_top1 = AverageMeter()
    end_time = time.time()

    for epoch in range(num_epoch):
        running_loss = AverageMeter()
        mmd_running_loss = AverageMeter()
        mean_regular_loss = AverageMeter()
        model.train() 
        scheduler.step()
        target_generator = iter(enumerate(target_train_loader))
        source_len, target_len = len(source_train_loader), len(target_train_loader)
        target_idx = 0
        for batch_idx, (images, labels) in enumerate(source_train_loader): 
            _, (target_images, target_labels) = next(target_generator)
            load_time.update(time.time() - end_time)
            images = images.to(device)
            labels = labels.to(device)
            target_images = target_images.to(device)
            target_labels = target_labels.to(device)

            ## compute features
            feat = model(images)
            target_feat = model(target_images)

            ### compute class prob 
            
            if args.loss == 'softmax': 
                outputs = src_model_fc(feat)
            elif args.loss == 'regular':
                outputs,regular_loss = src_model_fc(feat,labels)
            else:
                outputs = src_model_fc(feat,labels)
            ### compute loss
            if args.loss == 'regular':
                source_loss = criterion(outputs,labels)+torch.sum(regular_loss)/regular_loss.numel()
            else:
                source_loss = criterion(outputs, labels)

            mmd_loss = mmd_criterion(feat, target_feat)
            loss = 0.25*mmd_loss + source_loss

            ### compute prec1
            source_prec, = compute_accuracy(outputs, labels, topk=(1,))

            source_top1.update(source_prec.item())

            ## backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(source_loss.item())
            mmd_running_loss.update(mmd_loss.item())

            if args.loss == 'regular':
                mean_regular_loss.update((torch.sum(regular_loss)/regular_loss.numel()).item())
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            lr_now = optimizer.param_groups[0]['lr']

            if batch_idx % 100 == 0:
                if args.loss == 'regular':
                    logger("Epoch-Iter [%d/%d][%d/%d] Time_tot/load [%f][%f] lr [%g] src_loss [%f] mmd_loss [%f] regular_loss [%f] src_top1@1 [%f]"%(
                        epoch + 1 + start_epoch, num_epoch + start_epoch, batch_idx, len(source_train_loader), batch_time.avg,
                        load_time.avg, lr_now, running_loss.avg, mmd_running_loss.avg, mean_regular_loss.avg, source_top1.avg,))
                else:
                    logger("Epoch-Iter [%d/%d][%d/%d] Time_tot/load [%f][%f] lr [%g] src_loss [%f] mmd_loss [%f] src_top1@1 [%f]"%(
                        epoch + 1 + start_epoch, num_epoch + start_epoch, batch_idx, len(source_train_loader), batch_time.avg,
                        load_time.avg, lr_now, running_loss.avg, mmd_running_loss.avg, source_top1.avg, ))
                running_loss.reset()
                mmd_running_loss.reset()
                mean_regular_loss.reset()

                load_time.reset()
                batch_time.reset()
                source_top1.reset()
            target_idx += 1
            if target_idx >= target_len:
                target_idx = 0
                target_generator = iter(enumerate(target_train_loader))

        if (epoch + 1 + start_epoch) % save_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_path, save_name+'_' + str(
                           epoch + 1 + start_epoch) + '.tmp'))
            torch.save(src_model_fc.state_dict(),
                       os.path.join(save_path,save_name+'_model_fc'+'_'+str(
                           epoch+1+start_epoch) + '.tmp'))


def main():
    global args
    args = parse_args()
    print(args)
    train_net(args)


if __name__ == "__main__":
    main()
