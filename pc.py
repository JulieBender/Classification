#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 19:42:57 2024

@author: vigo
"""


import os
import sys
import argparse
import time
from tqdm import trange
import numpy as np
import PIL
import numpy as np

from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
import matplotlib.pyplot as plt

from models_classification import ResNet9, ConvNet

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

print(device)

# Hyper-parameters 
batch_size = 32
n_epochs = 20
learning_rate = 1e-4



# Specify the root folders for neoplastic and nonneoplastic images
neoplastic_folder = '/Users/juliebender/code/AIHealthcare/PCData/Neoplasia'
nonneoplastic_folder = '/Users/juliebender/code/AIHealthcare/PCData/Non-neoplasia'  


# Combine the data from neoplastic and nonneoplastic folders, excluding .DS_Store files
neoplastic_data = [(os.path.join(neoplastic_folder, file), 0) for file in os.listdir(neoplastic_folder) if not file.startswith('.')]
nonneoplastic_data = [(os.path.join(nonneoplastic_folder, file), 1) for file in os.listdir(nonneoplastic_folder) if not file.startswith('.')]
data = neoplastic_data + nonneoplastic_data


# Create the train dataset using train_data.txt
train_data = []
train_ids = set()  # Define train_ids before adding elements to it
# Assuming train_data.txt contains data in the format: id \t ccenum \t class_type
with open('train_data.txt', 'r') as file:
    for line in file:
        # Assuming train_data.txt contains data in the format: id \t ccenum \t class_type
        sdkid = line.strip().split(',')[0]
        train_ids.add(sdkid)


# Iterate through your neoplastic and nonneoplastic data
for filepath, label in data:
    filename = os.path.basename(filepath)
    sdkid_train = filename.split('_')[0]  # Assuming sdkid is the prefix before '_'
    #print(sdkid_data)
    if sdkid_train in train_ids:
        train_data.append((filepath, label))



# Create the valid dataset using valid_data.txt
valid_data = []
valid_ids = set()  # Define train_ids before adding elements to it
# Assuming train_data.txt contains data in the format: id \t ccenum \t class_type
with open('valid_data.txt', 'r') as file:
    for line in file:
        # Assuming train_data.txt contains data in the format: id \t ccenum \t class_type
        sdkid = line.strip().split(',')[0]
        valid_ids.add(sdkid)

# Iterate through your neoplastic and nonneoplastic data
for filepath, label in data:
    filename = os.path.basename(filepath)
    sdkid_val = filename.split('_')[0]  # Assuming sdkid is the prefix before '_'
    #print(sdkid_data)
    if sdkid_val in valid_ids:
        valid_data.append((filepath, label))
              
# Create the test dataset using train_data.txt
test_data = []
test_ids = set()  # Define train_ids before adding elements to it
# Assuming train_data.txt contains data in the format: id \t ccenum \t class_type
with open('test_data.txt', 'r') as file:
    for line in file:
        # Assuming train_data.txt contains data in the format: id \t ccenum \t class_type
        sdkid = line.strip().split(',')[0]
        test_ids.add(sdkid)

# Iterate through your neoplastic and nonneoplastic data
for filepath, label in data:
    filename = os.path.basename(filepath)
    sdkid_test = filename.split('_')[0]  # Assuming sdkid is the prefix before '_'
    #print(sdkid_data)
    if sdkid_test in test_ids:
        test_data.append((filepath, label))


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, os.path.basename(img_path)
    
    

# Define a variety of augmentations without resizing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    #transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    #transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize images 
])

# Create the custom dataset instances for training, validation, and testing
train_dataset = CustomDataset(data=train_data, transform=transform)
valid_dataset = CustomDataset(data=valid_data, transform=transform)
test_dataset = CustomDataset(data=test_data, transform=transform)



# Create DataLoader for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



def train(model, device, dataloader, criterion, optimizer):
    model.train()
    train_correct = 0
    n_total_steps = len(dataloader.sampler)
    for images, targets, image_names in dataloader:
        images, targets = images.to(device), targets.to(device).float().view(-1, 1) 
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predictions = (torch.sigmoid(outputs) > 0.65).float()
        train_correct += (predictions == targets).sum().item()
            
    # Calculate training accuracy
    train_acc = train_correct / n_total_steps * 100
    return train_acc

def val(model, device, dataloader, criterion):
    n_total_samples = 0
    val_loss = 0
    val_correct = 0
    model.eval()
    with torch.no_grad():
        for images, targets, image_names in dataloader:
            images, targets = images.to(device), targets.to(device).float().view(-1, 1) 
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            predictions = (torch.sigmoid(outputs) > 0.65).float()
            val_correct += (predictions == targets).sum().item()
            n_total_samples += targets.size(0)
        
    val_acc = val_correct / n_total_samples * 100
    avg_val_loss = val_loss / len(dataloader)
    return avg_val_loss, val_acc

# Initialize the model, criterion, optimizer
model = ResNet9(in_channels=3, num_classes=1).to(device)  # Single output for binary classification
criterion = nn.BCEWithLogitsLoss()
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
        torch.save(model.state_dict(), best_model_path)  # Save the current model parameters
        patience_counter = 0
    else:
        patience_counter += 1
        print(f'Early stopping counter: {patience_counter}/{patience}')
           
    # If no improvement for 'patience' epochs, stop training
    if patience_counter >= patience:
        print(f'No improvement in validation loss for {patience} epochs. Early stopping...')
        break

# Load the best model parameters
model.load_state_dict(torch.load(best_model_path))

test_loss, test_acc = val(model, device, test_loader, criterion)
print(f'Test Accuracy: {test_acc:.2f}%')

   
















