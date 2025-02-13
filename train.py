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
import torch.optim as optim
import math
from .data.streetscape_dataset import StreetscapeDataset
from .models.model import DISTS

def train(image_path: str, mask_path: str, df_train, df_val, istest: bool = False, resize: bool = True):

    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")

    train_dataset = StreetscapeDataset(df_train, image_path, mask_path, istest, resize)
    val_dataset = StreetscapeDataset(df_val, image_path, mask_path, istest, resize)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = DISTS()
    #pretrained_dict = torch.load('./saved_pth/bestAfter10.pth')
    #model.load_state_dict(pretrained_dict)
    model.train()

    #loss_function = nn.CrossEntropyLoss()
    tentative_weights = torch.tensor([1.0, math.sqrt(7.93)]) # 7.93 is the approximate ratio of label counts, too large weight values may cause gradient explosion
    tentative_criterion = nn.CrossEntropyLoss(weight=tentative_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Slow
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            id, img1, img2, mask1, mask2, targets = data
            print(id)
            optimizer.zero_grad()

            outputs = model(img1, img2, mask1, mask2, targets, istest=False, require_grad=True)
            loss = tentative_criterion(outputs, targets)
            train_loss_history.append(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # evaluate the training acc
        model.eval()
        num_correct, num_samples = 0, 0
        with torch.no_grad():
            for i, data in enumerate(train_loader, 0):
                id, img1, img2, mask1, mask2, targets = data
                scores = model(img1, img2, mask1, mask2, targets, istest=False, require_grad=False)
                label_pred = torch.argmax(scores.detach().cpu(), 1)
                num_correct += (label_pred == targets).sum()
                num_samples += label_pred.size(0)
            acc = float(num_correct) / num_samples
            train_acc_history.append(acc)

        # evaluate the validation acc
        model.eval()
        val_loss = 0.0
        val_correct, val_samples = 0, 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                id, img1, img2, mask1, mask2, targets = data
                scores = model(img1, img2, mask1, mask2, targets, istest=False, require_grad=False)

                # cal the val loss
                loss = tentative_criterion(scores, targets)
                val_loss_history.append(loss.item())

                label_pred = torch.argmax(scores.detach().cpu(), 1)
                val_correct += (label_pred == targets).sum()
                val_samples += label_pred.size(0)
            acc = float(val_correct) / val_samples
            val_acc_history.append(acc)
        torch.save(model.state_dict(), f'10_epochs.pth')
        print("Finished Training Epoch: "+str(epoch+1))
