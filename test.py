from __future__ import absolute_import, division, print_function
import cv2

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.utils import save_image

from skimage.measure import compare_psnr, compare_ssim
from tqdm import tqdm

import math
import numbers
import sys

import matplotlib.pyplot as plt

from multi_scale_bokeh import multi_bokeh

device = torch.device("cuda:0")

feed_width = 1536
feed_height = 1024


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


bokehnet = multi_bokeh_two().to(device)

bokehnet.load_state_dict(torch.load('./checkpoints/MPFNet.pth', map_location=device), False)

import time

total_time = 0

with torch.no_grad():
    for i in tqdm(range(4400, 4694)):
        image_path = './Training/original/' + str(i) + '.jpg'

        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size

        org_image = input_image
        org_image = transforms.ToTensor()(org_image).unsqueeze(0)

        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        org_image = org_image.to(device)
        input_image = input_image.to(device)

        start_time = time.time()
        bok_pred = bokehnet(input_image)
        total_time += time.time() - start_time

        bok_pred = F.interpolate(bok_pred, (original_height, original_width), mode='bilinear')

        save_image(bok_pred, './output/result/' + str(i) + '.png')

        del bok_pred
        del input_image