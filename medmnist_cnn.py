#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:35:15 2024

@author: vigo
"""


import os
import sys
import argparse
import time
from tqdm import trange
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

import medmnist
from medmnist import INFO, Evaluator

from models_classification import MLP, ResNet9, ConvNet


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

print(device)

#sys.exit()

torch.manual_seed(27)


# Hyper-parameters 
batch_size =32
n_epochs = 10
learning_rate = 1e-3



data_flag = 'organamnist'
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

# #To work with 28x28x1 size images
#data_transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[.5], std=[.5])])

### To work with full size images
data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])            


# organamnist (Abdominal CT images; 	Multi-Class (11);	58,830 images in total with trian/validation/test: 34,561 / 6,491 / 17,778)
train_data = DataClass(split='train', transform=data_transform, download=True, as_rgb=False)
val_data = DataClass(split='val', transform=data_transform, download=True, as_rgb=False)
#train_data=torch.utils.data.ConcatDataset([train_dataset, val_dataset])
test_data = DataClass(split='test', transform=data_transform, download=True, as_rgb=False)

train_loader = data.DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True)

val_loader = data.DataLoader(dataset=val_data,
                             batch_size=batch_size,
                             shuffle=False)

test_loader = data.DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                shuffle=False)



# disply some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
   
# show images
imshow(torchvision.utils.make_grid(images))


def train(model, device, dataloader, criterion, optimizer):
    model.train()
    train_correct = 0
    n_total_steps = len(train_loader.sampler)
    for (images, targets) in dataloader:
        images, targets = images.to(device), torch.squeeze(targets, 1).long().to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scores, predictions = torch.max(outputs.data, 1)
        train_correct += (predictions == targets).sum().item() 
            
    # Calculate training accuracy
    train_acc = train_correct / n_total_steps * 100
    return train_acc
        

def val(model, device, dataloader, criterion):
    n_total_samples = 0
    val_loss = 0
    val_correct=0
    model.eval()
    for (images, targets) in dataloader:
        images, targets = images.to(device), torch.squeeze(targets, 1).long().to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        
        scores, predictions = torch.max(outputs.data, 1)
        val_correct += (predictions == targets).sum().item() 
        n_total_samples += targets.size(0)
        
    val_acc = val_correct / n_total_samples * 100
    avg_val_loss = val_loss / len(dataloader)
    return avg_val_loss, val_acc




model = ConvNet(inchannels=n_channels, numclasses=n_classes).to(device)
#model = ResNet9(in_channels=n_channels, num_classes=n_classes).to(device)


# Pretrained Models

# #Resnet18/50/101
model = models.resnet18(pretrained=True)
# Modify the first convolutional layer to accept a single channel
model.conv1 = nn.Conv2d(
    in_channels=n_channels,      # Change number of input channels to 1
    out_channels=64,    # Keep the same number of output channels
    kernel_size=(7, 7), # Keep the same kernel size
    stride=(2, 2),      # Keep the same stride
    padding=(3, 3),     # Keep the same padding
    bias=False          # Keep the same bias setting
)
model.fc = nn.Linear(model.fc.in_features, n_classes)
model.to(device)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

# Early stopping parameters
best_val_loss = float('inf')
patience = 5
patience_counter = 0

# Path to save the best model
best_model_path = "best_model.pth"

for e in range(n_epochs):
    train_acc = train(model, device, train_loader, criterion, optimizer)
    val_loss, val_acc = val(model, device, val_loader, criterion)
    print(f'Epoch [{e+1}/{n_epochs}], Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%, Val Loss: {val_loss:.4f}')
       
    # Check for improvement in validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_params = model.state_dict()  # Save the current model parameters
        counter = 0
    else:
        counter += 1
           
    # If no improvement for 'patience' epochs, stop training
    if counter >= patience:
        print(f'No improvement in validation loss for {patience} epochs. Early stopping...')
        break


# Load the best model parameters
if best_model_params is not None:
    model.load_state_dict(best_model_params)


test_loss, test_acc = val(model, device, test_loader, criterion)
print(f'Test Accuracy: {test_acc:.2f}%')
   
















