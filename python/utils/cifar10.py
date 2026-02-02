
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

batch_size = 32
#transform the data and load it into the training and test dataset. 
training_transform = transforms.Compose(
            [
            transforms.RandomCrop(padding=4, size=32),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])     
        ])
validation_transform = transforms.Compose(
            [
			transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])     
        ])

train_dataset = torchvision.datasets.CIFAR10(
    root='./../datas/cifar10', 
    train=True,
    download=True,
    transform=training_transform,
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./../datas/cifar10',
    train=False,
    download=True,
    transform=validation_transform,
)

# Extract data and labels
cifar10_features = train_dataset.data  # This is already in shape (N, 32, 32, 3)
cifar10_labels = np.array(train_dataset.targets)  # Targets are in a list, convert to numpy array

# Split the CIFAR-10 data into train and test sets using sklearn's train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    cifar10_features, cifar10_labels, test_size=0.2, shuffle=True, random_state=42
)

# Convert to float64 if necessary (PyTorch expects float32, so double-check your needs)
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')

# Check the shapes of the train and test sets
print("Shape of x_train:", x_train.shape) 
print("Shape of x_test:", x_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True
                            )

test_loader = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False
                            )
