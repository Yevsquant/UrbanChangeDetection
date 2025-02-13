import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class L2pooling(nn.Module):
  def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
    super(L2pooling, self).__init__()
    self.padding = (filter_size - 2 )//2
    self.stride = stride
    self.channels = channels
    a = np.hanning(filter_size)[1:-1]
    g = torch.Tensor(a[:,None]*a[None,:])
    g = g/torch.sum(g)
    self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

  def forward(self, input):
    input = input**2
    out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
    return (out+1e-12).sqrt()