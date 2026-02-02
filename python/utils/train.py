
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

# This is the training function. Train, show the loss and accuracy for every epoch.
def train_model_function(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)  # Move the model to the specified device (GPU or CPU)
    
    # Lists to store loss and accuracy for each epoch
    epoch_losses = []
    epoch_accuracies = []
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= num_epochs, eta_min=0)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0 
        
        # Create a tqdm progress bar for the training process
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
        
        # Iterate through the batches of the training dataset
        for batch_idx, (inputs, targets) in progress_bar:
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # print(f'target:{targets.shape}')
            # print(f'inputs: {inputs.shape}')
            
            
            # Forward pass
            outputs = model(inputs)
            # print(f'outputs: {outputs.shape}')
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for the epoch and calculate the
            running_loss += loss.item()
            
            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # Update tqdm progress bar with loss and accuracy
            # progress_bar.set_postfix(loss=loss.item(), accuracy=correct / total * 100)
        
        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100  # Convert to percentage
        
        # Store the loss and accuracy
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        schedular.step()
        
        # Print statistics for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
    print('Training completed!')
    
    # Return the losses and accuracies
    return epoch_losses, epoch_accuracies

# Inference function
def inference_model_function(model, data_loader, device='cpu'):
    model.to(device)  # Move the model to the specified device (CPU or GPU)
    model.eval()  # Set the model to evaluation mode (turns off dropout, batch norm, etc.)
    
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for inputs, labels in data_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass through the model
            outputs = model(inputs)
            
            # Get the predicted class and accumilate (with the highest score)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy
    accuracy = (correct / total )* 100  # Multiply by 100 to get percentage
    return predictions, accuracy


def predict_single_image_function(model, image, device='cpu'):
    """Predict the class of a single image."""
    # Set the model to evaluation mode
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
    return predicted.item()
