import torch
import torch.nn as nn
import torch.nn.functional as F
import pyblur.pyblur
from my_pytorch_mssim import pytorch_msssim
class FE(nn.Module):
    def __init__(self):
        super(FE, self).__init__()
        # Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        # Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        # Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        # Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        # Conv3
        x = self.layer9(x)
        x = self.layer10(x) + x
        x = self.layer11(x) + x
        return x


class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)
        # Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        # Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x


class multi_bokeh(nn.Module):
    def __init__(self):
        super(multi_bokeh, self).__init__()
        self.fe_lv1 = FE()
        self.fe_lv2 = FE()
        self.fe_lv3 = FE()

        self.gen_lv1 = Gen()
        self.gen_lv2 = Gen()
        self.gen_lv3 = Gen()

        self.rough_pre1 = pyblur.pyblur.DefocusBlur()
        self.rough_pre2 = pyblur.pyblur.DefocusBlur()
        self.rough_pre3 = pyblur.pyblur.DefocusBlur()

    def forward(self, orig_lv3):
        H = orig_lv3.size(2)
        W = orig_lv3.size(3)

        orig_lv2 = F.interpolate(orig_lv3, scale_factor=0.5, mode='bilinear')
        orig_lv1 = F.interpolate(orig_lv2, scale_factor=0.5, mode='bilinear')

        feature_lv2 = self.fe_lv1(orig_lv2)
        feature_lv1 = self.fe_lv1(orig_lv1)+F.interpolate(feature_lv2,scale_factor=0.5, mode='bilinear')
        # gen_lv1 = self.gen_lv1(feature_lv1)
        # rough_pre1=self.rough_pre1(orig_lv1,3)
        #
        # out_lv1 = orig_lv1 + residual_lv1

        # residual_lv1 = F.interpolate(residual_lv1, scale_factor=2, mode='bilinear')
        feature_lv12 = F.interpolate(feature_lv1, scale_factor=2, mode='bilinear')
        # feature_lv2 = self.fe_lv2(orig_lv2 + residual_lv1)
        gen_lv12 = self.gen_lv2(feature_lv12 + feature_lv2)
        gen_lv12 = self.gen_lv1(feature_lv12)
        # out_lv2 = orig_lv2 + residual_lv2

        feature_lv3 = self.fe_lv3(orig_lv3)
        feature_lv23 = self.fe_lv2(orig_lv2) + F.interpolate(feature_lv3, scale_factor=0.5, mode='bilinear')
        feature_lv3=F.interpolate(feature_lv23, scale_factor=2, mode='bilinear')

        gen_lv1 = self.gen_lv1(feature_lv1)
        gen_lv12 = self.gen_lv1(feature_lv12)
        gen_lv23 = self.gen_lv1(feature_lv23)
        gen_lv3 = self.gen_lv1(feature_lv3)

        rough_pre1 = self.rough_pre1(orig_lv3,3)
        rough_pre2 = self.rough_pre1(orig_lv2, 5)
        rough_pre3 = self.rough_pre1(orig_lv1, 7)

        ssim1 = pytorch_msssim.SSIM(gen_lv1,rough_pre1)
        ssim12=pytorch_msssim.SSIM(gen_lv12,rough_pre2)
        ssim23=pytorch_msssim.SSIM(gen_lv23,rough_pre2)
        ssim3=pytorch_msssim.SSIM(gen_lv3,rough_pre3)
        overall=((1-ssim1)/2+(1-ssim12)/2+(1-ssim23)/2+(1-ssim3)/2)*0.75
        w1=ssim1/overall
        w12=ssim12/overall
        w23=ssim23/overall
        w3=ssim3/overall

        bokeh_image=w1*gen_lv1+w12*gen_lv12+w23+gen_lv23+w3*gen_lv3

        return bokeh_image
