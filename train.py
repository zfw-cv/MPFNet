from __future__ import absolute_import, division, print_function
import cv2
import numbers
import math
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from util import utils
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
# import pytorch_ssim
import torchvision
from torch.autograd import Variable
from my_pytorch_mssim import pytorch_msssim
from util import Vidsom
from tqdm import tqdm

# from tensorboardX import SummaryWriter
device = torch.device("cuda:0")
# from net_multi_bokeh import model_bokeh_base
import multi_scale_bokeh

feed_width = 1024
feed_height = 1024
bokehnet = multi_scale_bokeh.multi_bokeh().to(device)

bokehnet.net.load_state_dict(torch.load('checkpoints/MPFNet.pth', map_location=device))


class bokehDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bok = pil.open(self.root_dir + self.data.iloc[idx, 0][1:]).convert('RGB')
        org = pil.open(self.root_dir + self.data.iloc[idx, 1][1:]).convert('RGB')

        bok = bok.resize((feed_width, feed_height), pil.LANCZOS)
        org = org.resize((feed_width, feed_height), pil.LANCZOS)
        if self.transform:
            bok_dep = self.transform(bok)
            org_dep = self.transform(org)
        return (bok_dep, org_dep)


def smooth_loss(im):
    return torch.mean(torch.abs(im[:, :, 1:, :] - im[:, :, :-1, :])) + torch.mean(
        torch.abs(im[:, :, :, 1:] - im[:, :, :, :-1]))


transform1 = transforms.Compose(
    [
        transforms.ToTensor(),
    ])

transform2 = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
    ])

transform3 = transforms.Compose(
    [
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
    ])

trainset1 = bokehDataset(csv_file='./data/train.csv', root_dir='./', transform=transform1)
trainset2 = bokehDataset(csv_file='./data/train.csv', root_dir='./', transform=transform2)
trainset3 = bokehDataset(csv_file='./data/train.csv', root_dir='./', transform=transform3)

# batch size changed
trainloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([trainset1, trainset2, trainset3]),
                                          batch_size=2,
                                          shuffle=True, num_workers=0)

testset = bokehDataset(csv_file='./data/test.csv', root_dir='./', transform=transform1)

testloader = torch.utils.data.DataLoader(testset, batch_size=2,
                                         shuffle=False, num_workers=0)
# learning_rate = 0.0001
learning_rate = 0.00001

optimizer = optim.Adam(list(bokehnet.parameters()), lr=learning_rate, betas=(0.9, 0.999))

sm = nn.Softmax(dim=1)
ssim_loss = pytorch_ssim.SSIM()
msssim_loss = pytorch_msssim.MS_SSIM()
MSE_LossFn = nn.MSELoss()
L1_LossFn = nn.L1Loss()


def multi_scale_loss(pred_frames, gt, alpha=0.1, w1=1, w2=2, w3=4):
    gt_lv2 = F.interpolate(gt, scale_factor=0.5, mode='bilinear')
    gt_lv3 = F.interpolate(gt, scale_factor=0.25, mode='bilinear')

    lv1_loss = L1_LossFn(gt, pred_frames[0]) + alpha * (1 - ssim_loss(gt, pred_frames[0]))
    lv2_loss = L1_LossFn(gt_lv2, pred_frames[1]) + alpha * (1 - ssim_loss(gt_lv2, pred_frames[1]))
    lv3_loss = L1_LossFn(gt_lv3, pred_frames[2]) + alpha * (1 - ssim_loss(gt_lv3, pred_frames[2]))

    loss = w1 * lv1_loss + w2 * lv2_loss + w3 * lv3_loss

    return loss


def train(dataloader):
    running_l1_loss = 0
    running_ms_loss = 0
    running_sal_loss = 0
    running_loss = 0
    for i, data in enumerate(tqdm(dataloader), 0):
        bok, org = data
        bok, org = bok.to(device), org.to(device)
        bok_pred = bokehnet(org)
        loss = (1 - msssim_loss(bok_pred, bok))
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % 100 == 0):
            print('Batch: ', i, '/', len(dataloader), ' Loss:', loss.item())
        if (i % 1600 == 0):
            torch.save(bokehnet.state_dict(), './checkpoints/model/MPFNet-' + str(epoch) + '-' + str(i) + '.pth')
            print(loss.item())
    print(running_l1_loss / len(dataloader))
    print(running_ms_loss / len(dataloader))
    print(running_loss / len(dataloader))


def val(dataloader):
    running_l1_loss = 0
    running_ms_loss = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader), 0):
            bok, org = data
            bok, org = bok.to(device), org.to(device)

            bok_pred = bokehnet(org)

            l1_loss = L1_LossFn(bok_pred[0], bok)
            ms_loss = 1 - ssim_loss(bok_pred, bok)

            running_l1_loss += l1_loss.item()
            running_ms_loss += ms_loss.item()

    print('Validation l1 Loss: ', running_l1_loss / len(dataloader))


start_ep = 1
for epoch in range(start_ep, 4000):
    print(epoch)

    train(trainloader)

    with torch.no_grad():
        val(testloader)

    # train(trainloader)