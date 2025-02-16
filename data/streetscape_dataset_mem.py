from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class StreetscapeDataset(Dataset):
  def __init__(self, imgpairfile, image_path, mask_path, istest, resize=False):
    attr = imgpairfile
    self.indlist = list(attr.loc[:,'temp_id'])
    self.imglist = list(image_path + "/" + attr.loc[:,'image'])
    self.masklist1 = list(mask_path + "/" + attr.loc[:,'min_panoid_mask'])
    self.masklist2 = list(mask_path + "/" + attr.loc[:,'max_panoid_mask'])
    self.targetlist = [-1] * len(self.indlist) if istest else list(attr.loc[:, 'label'])
    self.resize = resize
    self.istest = istest

  def __len__(self):
    return len(self.indlist)

  def __getitem__(self, idx):
    merged_image = Image.open(self.imglist[idx])

    middle_point = merged_image.width // 2

    img1 = merged_image.crop((0, 0, middle_point, merged_image.height))
    img2 = merged_image.crop((middle_point, 0, merged_image.width, merged_image.height))

    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    mask1 = Image.open(self.masklist1[idx]).convert("L")
    mask2 = Image.open(self.masklist2[idx]).convert("L")

    if self.resize:
      if min(img1.size)>256:
        img1 = transforms.functional.resize(img1,256)
      if min(img2.size)>256:
        img2 = transforms.functional.resize(img2,256)
      if min(mask1.size)>256:
        mask1 = transforms.functional.resize(mask1,256)
      if min(mask2.size)>256:
        mask2 = transforms.functional.resize(mask2,256)

    img1 = transforms.ToTensor()(img1)
    img2 = transforms.ToTensor()(img2)
    mask1 = torch.as_tensor(np.array(mask1), dtype=torch.bool).unsqueeze(0)
    mask2 = torch.as_tensor(np.array(mask2), dtype=torch.bool).unsqueeze(0)

    return (self.indlist[idx], img1, img2, mask1, mask2, self.targetlist[idx])