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
from .data.streetscape_dataset_nyc import StreetscapeDataset
from .models.model import DISTS

def train(df_train, df_val, num_epochs: int = 19, resize: bool = True):
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")


    # Load data
    train_dataset = StreetscapeDataset(df_train, False, resize)
    val_dataset = StreetscapeDataset(df_val, False, resize)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = DISTS().to(device)
    model.train()

    # Set loss function and optimizer
    # 3.37 is the approximate ratio of label counts. sqrt(3.37) = 1.83
    tentative_weights = torch.tensor([1.0, 1.83], dtype=torch.float32).to(device)
    loss_function = nn.CrossEntropyLoss(weight=tentative_weights.to(device))
    optimizer_1 = optim.Adam(model.parameters(), lr=0.0001)
    optimizer_2 = optim.Adam(model.parameters(), lr=0.00001)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        model.train()

        # Train a batch of data
        for i, data in enumerate(train_loader, 0):
            id, img1, img2, mask1, mask2, targets = data
            img1, img2, mask1, mask2, targets = (
                img1.to(device),
                img2.to(device),
                mask1.to(device),
                mask2.to(device),
                targets.to(device),
            )

            optimizer = optimizer_1 if epoch != 18 else optimizer_2
            optimizer.zero_grad()

            outputs = model(img1, img2, mask1, mask2, targets, istest=False, require_grad=True)
            loss = loss_function(outputs, targets)
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
                num_correct += (label_pred == targets.cpu()).sum().item()
                num_samples += label_pred.size(0)
            acc = num_correct / num_samples
            train_acc_history.append(acc)

        # evaluate the validation acc
        model.eval()
        val_loss = 0.0
        val_correct, val_samples = 0, 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                id, img1, img2, mask1, mask2, targets = data
                scores = model(img1, img2, mask1, mask2, targets, istest=False, require_grad=False)

                # calculate the validation loss
                loss = loss_function(scores, targets)
                val_loss_history.append(loss.item())

                label_pred = torch.argmax(scores.detach().cpu(), 1)
                val_correct += (label_pred == targets.cpu()).sum().item()
                val_samples += label_pred.size(0)
            acc = val_correct / val_samples
            val_acc_history.append(acc)

        torch.save(model.state_dict(), 'trainset.pth')
    
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history, model
