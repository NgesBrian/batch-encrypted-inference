
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Define the Basic Residual Block used in ResNet-18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut layer
        self.shortcut_conv = None
        self.shortcut_bn = None
        if stride != 1 or in_channels != out_channels:
            # Shortcut layer for matching dimensions if needed
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = x
        
        # First conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Apply shortcut if dimensions do not match
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(x)
            shortcut = self.shortcut_bn(shortcut)
        
        out += shortcut
        out = F.relu(out)
        return out


# Define the ResNet-18 model
class ResNet20(nn.Module):
    def __init__(self, channel_values, num_classes=10):
        super(ResNet20, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, channel_values[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_values[0])
        
        # Define each residual block with two BasicBlocks each
        self.layer1_block1 = BasicBlock(channel_values[0], channel_values[0])
        self.layer1_block2 = BasicBlock(channel_values[0], channel_values[0])
        self.layer1_block3 = BasicBlock(channel_values[0], channel_values[0])
        
        self.layer2_block1 = BasicBlock(channel_values[0], channel_values[1], stride=2)
        self.layer2_block2 = BasicBlock(channel_values[1], channel_values[1])
        self.layer2_block3 = BasicBlock(channel_values[1], channel_values[1])
        
        self.layer3_block1 = BasicBlock(channel_values[1], channel_values[2], stride=2)
        self.layer3_block2 = BasicBlock(channel_values[2], channel_values[2])
        self.layer3_block3 = BasicBlock(channel_values[2], channel_values[2])
        
        # Fully connected layer
        # self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(channel_values[2], num_classes)

    def forward(self, x):
        # Initial conv + batch norm + ReLU + max pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Layer 1 (64 -> 64 channels, stride=1)
        x = self.layer1_block1(x)
        x = self.layer1_block2(x)
        x = self.layer1_block3(x)
        
        # Layer 2 (64 -> 128 channels, stride=2)
        x = self.layer2_block1(x)
        x = self.layer2_block2(x)
        x = self.layer2_block3(x)
        
        # Layer 3 (128 -> 256 channels, stride=2)
        x = self.layer3_block1(x)
        x = self.layer3_block2(x)
        x = self.layer3_block3(x)
        
        # Global average pooling and fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
