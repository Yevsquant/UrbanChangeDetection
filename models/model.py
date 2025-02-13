import torch
import torch.nn as nn
from torchvision import models
from l2_pooling import L2pooling

class DISTS(torch.nn.Module):
  def __init__(self):
    super(DISTS, self).__init__()
    """
    All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. More details about torchvision.models at: https://pytorch.org/vision/0.10/models.html
    A detailed explanation of vgg16 structure is discussed here: https://blog.csdn.net/Geek_of_CSDN/article/details/84343971. Also the paper: https://arxiv.org/pdf/1409.1556.pdf
    """

    vgg_pretrained_features = models.vgg16(pretrained=True).features

    self.stage1 = nn.Sequential()
    self.stage2 = nn.Sequential()
    self.stage3 = nn.Sequential()
    self.stage4 = nn.Sequential()
    self.stage5 = nn.Sequential()

    for x in range(0,4):
      self.stage1.add_module(str(x), vgg_pretrained_features[x])
    self.stage2.add_module(str(4), L2pooling(channels=64))
    for x in range(5, 9):
      self.stage2.add_module(str(x), vgg_pretrained_features[x])
    self.stage3.add_module(str(9), L2pooling(channels=128))
    for x in range(10, 16):
      self.stage3.add_module(str(x), vgg_pretrained_features[x])
    self.stage4.add_module(str(16), L2pooling(channels=256))
    for x in range(17, 23):
      self.stage4.add_module(str(x), vgg_pretrained_features[x])
    self.stage5.add_module(str(23), L2pooling(channels=512))
    for x in range(24, 30):
      self.stage5.add_module(str(x), vgg_pretrained_features[x])

    for param in self.parameters():
      param.requires_grad = False

    self.fc1 = nn.Linear(2950, 1000)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout()
    self.fc2 = nn.Linear(1000, 500)
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout()
    self.fc3 = nn.Linear(500, 250)
    self.relu3 = nn.ReLU()
    self.dropout3 = nn.Dropout()
    self.fc4 = nn.Linear(250, 100)
    self.relu4 = nn.ReLU()
    self.dropout4 = nn.Dropout()
    self.fc5 = nn.Linear(100, 50)
    self.relu5 = nn.ReLU()
    self.dropout5 = nn.Dropout()
    self.fc6 = nn.Linear(50, 10)
    self.relu6 = nn.ReLU()
    self.dropout6 = nn.Dropout()
    self.fc7 = nn.Linear(10, 2)

    self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
    self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

    self.chns = [3,64,128,256,512,512]

  def shrinkMask(self, mask, shrinkfactors = (0.5, 0.5)):
    h_factor, w_factor = shrinkfactors
    maskh = round(mask.size()[2] * h_factor)
    maskw = round(mask.size()[3] * w_factor)
    posb, posc, posh, posw = (mask!=0).nonzero(as_tuple=True)
    posh = torch.round(posh * h_factor).int()
    posh[posh >= maskh] = maskh - 1
    posw = torch.round(posw * w_factor).int()
    posw[posw >= maskw] = maskw - 1
    pos = [(a.item(), b.item(), c.item(), d.item()) for a, b, c, d in zip(posb, posc, posh, posw)]
    pos = list(set(pos))
    posb, posc, posh, posw = zip(*pos)
    mask = torch.zeros(mask.size()[0], 1, maskh, maskw, dtype = torch.int8)
    mask[posb, posc, posh, posw] = 1
    return mask.type(torch.bool)

  def forward_once(self, x, maskx):
    featx = x * maskx

    h = (x-self.mean)/self.std
    h = self.stage1(h)
    h_relu1_2 = h * maskx

    h = self.stage2(h)
    maskx1 = self.shrinkMask(maskx).to(h.device)
    h_relu2_2 = h * maskx1

    h = self.stage3(h)
    maskx2 = self.shrinkMask(maskx1).to(h.device)
    h_relu3_3 = h * maskx2

    h = self.stage4(h)
    maskx3 = self.shrinkMask(maskx2).to(h.device)
    h_relu4_3 = h * maskx3

    h = self.stage5(h)
    maskx4 = self.shrinkMask(maskx3).to(h.device)
    h_relu5_3 = h * maskx4

    return [(featx, maskx), (h_relu1_2, maskx), (h_relu2_2, maskx1), (h_relu3_3, maskx2), (h_relu4_3, maskx3), (h_relu5_3, maskx4)]

  def forward(self, x, y, mask_1, mask_2, target, istest, require_grad=False):
    if require_grad:
      res0 = self.forward_once(x, mask_1)
      res1 = self.forward_once(y, mask_2)
      feats0, mask0 = map(list,zip(*res0))
      feats1, mask1 = map(list,zip(*res1))
    else:
      with torch.no_grad():
        res0 = self.forward_once(x, mask_1)
        res1 = self.forward_once(y, mask_2)
        feats0, mask0 = map(list,zip(*res0))
        feats1, mask1 = map(list,zip(*res1))

    dist1 = torch.empty(0, device=x.device)
    dist2 = torch.empty(0, device=x.device)
    c1 = 1e-6
    c2 = 1e-6

    for k in range(len(self.chns)):
      area0 = mask0[k].sum([2,3], keepdim=True)
      x_mean = feats0[k].sum([2,3], keepdim=True) / area0
      y_mean = feats1[k].sum([2,3], keepdim=True) / area0
      S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)

      # Adapt from dist1 = torch.cat((dist1, S1.squeeze()), 1) for one image
      S1_squeezed = S1.squeeze()
      if S1_squeezed.dim() == 1:
        S1_squeezed = S1_squeezed.unsqueeze(0)
      dist1 = torch.cat((dist1, S1_squeezed), 1)

      x_var = ((feats0[k]-x_mean * mask0[k])**2).sum([2,3], keepdim=True) / area0
      y_var = ((feats1[k]-y_mean * mask0[k])**2).sum([2,3], keepdim=True) / area0
      xy_cov = (feats0[k]*feats1[k]).sum([2,3], keepdim=True) / area0 - x_mean*y_mean
      S2 = (2*xy_cov+c2)/(x_var+y_var+c2)

      # dist2 = torch.cat((dist2, S2.squeeze()), 1)
      S2_squeezed = S2.squeeze()
      if S2_squeezed.dim() == 1:
        S2_squeezed = S2_squeezed.unsqueeze(0)
      dist2 = torch.cat((dist2, S2_squeezed), 1)

    dist12 = torch.cat((dist1, dist2), 1)

    score = self.fc1(dist12)
    score = self.relu1(score)
    score = self.dropout1(score)
    score = self.fc2(score)
    score = self.relu2(score)
    score = self.dropout2(score)
    score = self.fc3(score)
    score = self.relu3(score)

    score = self.dropout3(score)
    score = self.fc4(score)
    score = self.relu4(score)
    score = self.dropout4(score)
    score = self.fc5(score)
    score = self.relu5(score)
    score = self.dropout5(score)
    score = self.fc6(score)
    score = self.relu6(score)
    score = self.dropout6(score)
    score = self.fc7(score)

    if istest:
      loss = None

    print(score)

    return score