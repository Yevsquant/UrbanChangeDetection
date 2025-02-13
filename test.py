from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.backends import cudnn
from tqdm import tqdm
import math
import pandas as pd
from .data.streetscape_dataset import StreetscapeDataset
from .models.model import DISTS

def test(image_path: str, mask_path: str, df_test, model, istest: bool = True, resize: bool = True):
    istest = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")

    sdataset = StreetscapeDataset(df_test, image_path, mask_path, istest, resize)
    sloader = DataLoader(sdataset, batch_size=16, shuffle=False, pin_memory=True)
    model.eval()

    output = pd.DataFrame(columns=['ID', 'predicted_label'])
    with torch.no_grad():
        for ind, x, y, mask1, mask2, target in tqdm(sloader):
            ind, x, y, mask1, mask2, target = ind.to(device), x.to(device), y.to(device), mask1.to(device), mask2.to(device), target.to(device)
            score = model(x, y, mask1, mask2, target, istest=True)
            label = torch.argmax(score.detach().cpu(), 1, keepdim=True)
            tempdf = torch.cat((torch.unsqueeze(ind.float(), 1), label.float()), 1)
            tempdf = pd.DataFrame(tempdf.numpy(), columns=['ID', 'predicted_label'])
            output = pd.concat([output, tempdf])

    output['ID'] = output['ID'].astype(int)
    output['predicted_label'] = output['predicted_label'].astype(int)
    output = output.reset_index(drop=True, inplace=False)
    output['correct_label'] = df_test['label'].astype(int).reset_index(drop=True, inplace=False)
    print("Testing/inference task is done!")
    output.to_csv('labels.csv', index = False)

    label_equality = (output['predicted_label'] == output['correct_label']).astype(int)
    acc = sum(label_equality) / len(label_equality)
    acc