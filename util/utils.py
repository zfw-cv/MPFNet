import torch
import torch.nn as nn
from torch.nn import *

import numpy as np
import os
from os import listdir
from os.path import join
import torchvision.transforms as transforms
from PIL import Image


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
	img = Image.open(filename).convert('RGB')
	if size is not None:
		if keep_asp:
			size2 = int(size * 1.0 / img.size[0] * img.size[1])
			img = img.resize((size, size2), Image.ANTIALIAS)
		else:
			img = img.resize((size, size), Image.ANTIALIAS)

	elif scale is not None:
		img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
	img = np.array(img).transpose(2, 0, 1)
	img = torch.from_numpy(img).float()
	return img



def tensor_save_rgbimage(tensor, filename, cuda=False):
	if cuda:
		img = tensor.clone().cpu().clamp(0, 255).numpy()
	else:
		img = tensor.clone().clamp(0, 255).numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	img = Image.fromarray(img)
	img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
	(b, g, r) = torch.chunk(tensor, 3)
	tensor = torch.cat((r, g, b))
	tensor_save_rgbimage(tensor, filename, cuda)


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def Write_test_perform( file_path , psnr , ssim , epoch):
    """test   psnr , ssim """
    if not isinstance(psnr , str):
        psnr = str(psnr)

    if not isinstance(ssim , str):
        ssim = str(ssim)
    if not isinstance(epoch , str):
        epoch = str(epoch)

    with open(file_path , "a") as  f :
        f.write("\n epoch: " + epoch + " , psnr :" + psnr + ",  ssim :" + ssim )



def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])




def load_all_image(path):
    return [join(path, x) for x in listdir(path) if is_image_file(x)]


def save_checkpoint(root ,model, epoch, model_stage ):

    model_out_path = root+"/%s/%d.pth" % (model_stage, epoch)
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    if not os.path.exists(root+"/%s" % (model_stage)):
        os.makedirs(root+"/%s" % (model_stage))

    torch.save({
        'epoch': epoch,
        'state_dict': state_dict}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.features(x)


def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std



def get_residue(tensor , r_dim = 1):
    """
    return residue_channle (RGB)
    """
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]  
    return res_channel

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr



def rgb2grad(img):
    '''in:batch_size 3 512 512  out:batch_size 1 512 512 
	in:tensor out:tensor'''
    R = img[ : ,:1]
    G = img[ : ,1:2]
    B = img[ : ,2:3]
    img = 0.299*R + 0.587*G + 0.114*B
    return img


import torch
import math
import sys
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

from torchvision.transforms import  functional as F

class RandomCrop(object):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):

        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img1 , img2 ):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img1 = F.pad(img1, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img1.size[0] < self.size[1]:
            img1 = F.pad(img1, (self.size[1] - img1.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img1.size[1] < self.size[0]:
            img1 = F.pad(img1, (0, self.size[0] - img1.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img1, self.size)

        return F.crop(img1, i, j, h, w) , F.crop(img2, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

