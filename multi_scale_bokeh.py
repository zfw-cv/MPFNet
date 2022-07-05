import torch
import torch.nn as nn 
import torch.nn.functional as F

from model_bokeh_base import multi_bokeh

class multi_bokeh(nn.Module):
    def __init__(self):
        super(multi_bokeh_two,self).__init__()
        self.net = multi_bokeh()
 

    def forward(self,x):
        out1 = self.net1(x)

        return out1
