#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:49:30 2024

@author: vigo
"""

import torch.nn as nn
import torch.nn.functional as F

# Fully connected neural network with one hidden layer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  
        self.l2 = nn.Linear(hidden_size, hidden_size // 2)  # New layer
        self.l3 = nn.Linear(hidden_size // 2, num_classes) 
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)  #dropout
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out) 
        return out
    
    
class ConvNet(nn.Module):
    def __init__(self, inchannels, numclasses):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 6, 9)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 7)
        self.conv3 = nn.Conv2d(12, 16, 4)
        self.fc1 = nn.Linear(16 * 24 * 24, 256)
        #self.bn1 = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, numclasses)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 108, 108
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 12, 51, 51
        x = self.pool(F.relu(self.conv3(x)))  # -> n, 16, 24, 24
        x = x.view(-1, 16 * 24 * 24)            # -> n, 25088
        x = F.relu(self.fc1(x))               # -> n, 256
        #x = F.relu(self.bn1(self.fc1(x)))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 128
        x = self.fc3(x)                       # -> n, 10
        return x
    

#ResNet9
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out