from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class StreetscapeDataset(Dataset):
    def __init__(self, imgpairfile, istest, resize=False):
        attr = imgpairfile
        self.imglist1 = attr.loc[:, 'min image location']
        self.imglist2 = attr.loc[:, 'max image location']
        self.masklist1 = attr.loc[:, 'min_panoid_mask location']
        self.masklist2 = attr.loc[:, 'max_panoid_mask location']
        self.indlist = attr.loc[:, "temp_id"]
        self.targetlist = [-1] * len(self.indlist) if istest else list(attr.loc[:, 'label'])
        self.resize = resize
        self.istest = istest

    def __len__(self):
        return len(self.indlist)

    def __getitem__(self, idx):
        img1_path = self.imglist1[idx]
        img2_path = self.imglist2[idx]
        mask1_path = self.masklist1[idx]
        mask2_path = self.masklist2[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        mask1 = Image.open(mask1_path).convert("L")
        mask2 = Image.open(mask2_path).convert("L")

        if self.resize:
            if min(img1.size) > 256:
                img1 = transforms.functional.resize(img1, 256)
            if min(img2.size) > 256:
                img2 = transforms.functional.resize(img2, 256)
            if min(mask1.size) > 256:
                mask1 = transforms.functional.resize(mask1, 256)
            if min(mask2.size) > 256:
                mask2 = transforms.functional.resize(mask2, 256)

        img1 = transforms.ToTensor()(img1)
        img2 = transforms.ToTensor()(img2)
        mask1 = torch.as_tensor(np.array(mask1), dtype=torch.bool).unsqueeze(0)
        mask2 = torch.as_tensor(np.array(mask2), dtype=torch.bool).unsqueeze(0)

        return (self.indlist[idx], img1, img2, mask1, mask2, self.targetlist[idx])
