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
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image of size (H, W, C).
        Returns:
            PIL Image: Image with n_holes of dimension length x length cut out of it.
        """
        img = np.array(img)
        h, w, _ = img.shape

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = np.expand_dims(mask, axis=-1)
        img = img * mask

        img = Image.fromarray(img.astype(np.uint8))

        return img


# This is the training function. Train, show the loss and accuracy for every epoch.
def train_model_function(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    
    # Lists to store loss and accuracy for each epoch
    epoch_losses = []
    epoch_accuracies = []
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= num_epochs, eta_min=0)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create a tqdm progress bar for the training process
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
        
        # Iterate through the batches of the training dataset
        for batch_idx, (inputs, targets) in progress_bar:
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for the epoch and calculate the
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # Update tqdm progress bar with loss and accuracy
            # progress_bar.set_postfix(loss=loss.item(), accuracy=correct / total * 100)
        
        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100
        
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
class ResNet34(nn.Module):
    def __init__(self, channel_values, num_classes=10):
        super(ResNet34, self).__init__()
        
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
        self.layer2_block4 = BasicBlock(channel_values[1], channel_values[1])
        
        self.layer3_block1 = BasicBlock(channel_values[1], channel_values[2], stride=2)
        self.layer3_block2 = BasicBlock(channel_values[2], channel_values[2])
        self.layer3_block3 = BasicBlock(channel_values[2], channel_values[2])
        self.layer3_block4 = BasicBlock(channel_values[2], channel_values[2])
        self.layer3_block5 = BasicBlock(channel_values[2], channel_values[2])
        self.layer3_block6 = BasicBlock(channel_values[2], channel_values[2])
        
        self.layer4_block1 = BasicBlock(channel_values[2], channel_values[3], stride=2)
        self.layer4_block2 = BasicBlock(channel_values[3], channel_values[3])
        self.layer4_block3 = BasicBlock(channel_values[3], channel_values[3])

        # Fully connected layer
        # self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(channel_values[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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
        x = self.layer2_block4(x)
        
        # Layer 3 (128 -> 256 channels, stride=2)
        x = self.layer3_block1(x)
        x = self.layer3_block2(x)
        x = self.layer3_block3(x)
        x = self.layer3_block4(x)
        x = self.layer3_block5(x)
        x = self.layer3_block6(x)

        # Layer 4 
        x = self.layer4_block1(x)
        x = self.layer4_block2(x)
        x = self.layer4_block3(x)
        
        # Global average pooling and fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

#device configuration
#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
no_epochs = 350
batch_size = 128
learning_rate = 0.002

training_transform = transforms.Compose(
            [
            transforms.RandomCrop(padding=4, size=32),
			transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            Cutout(n_holes = 1, length=16),
			transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
validation_transform = transforms.Compose(
            [
			transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])     
        ])

train_dataset = torchvision.datasets.CIFAR100(
    root='./../datas/cifar100', 
    train=True,
    download=True,
    transform=training_transform,
)

test_dataset = torchvision.datasets.CIFAR100(
    root='./../datas/cifar100',
    train=False,
    download=True,
    transform=validation_transform,
)

# Extract data and labels
cifar100_features = train_dataset.data  # This is already in shape (N, 32, 32, 3)
cifar100_labels = np.array(train_dataset.targets)  # Targets are in a list, convert to numpy array

# Split the CIFAR-10 data into train and test sets using sklearn's train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    cifar100_features, cifar100_labels, test_size=0.2, shuffle=True, random_state=42
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

channel_values = [16, 32, 64, 128]
print(channel_values)
num_classes = 100
model = ResNet34(channel_values, num_classes)
# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()  # Cross entropy loss for classification
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Adam optimizer
# epoch_losses, epoch_accuracies = train_model_function(model, train_loader, criterion, optimizer, num_epochs=no_epochs, device=device)

# current_date = datetime.now().strftime('%Y-%m-%d')
# save_path = f'./../models/resnet34_2025-01-19.pth'
# model.load_state_dict(torch.load(save_path), strict=False)
# torch.save(model.state_dict(), save_path)
# print(f"\nmodel saved at {save_path}")

save_path = f'./../models/resnet34_2025-01-19.pth'
model = ResNet34(channel_values, num_classes).to(device)
model.load_state_dict(torch.load(save_path), strict=False)

predictions, accuracy = inference_model_function(model, test_loader, device=device)
print(f'Inference Accuracy: {accuracy:.2f}%')

# Load 10 images from the MNIST dataset
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
# Predict and display the result for each image in the batch
for batch_idx, (images, targets) in enumerate(test_loader):
	print(f'Batch {batch_idx + 1}')
	for idx in range(len(images)):
		image = images[idx]
		target = targets[idx].item()
		predicted_class = predict_single_image_function(model, image, device='cpu')
		if(target != predicted_class):
			print(f'Wrong Prediction for Image {idx + 1} - True Label: {target}, Predicted Label: {predicted_class}')
		else:
			print(f'Image {idx + 1} - True Label: {target}, Predicted Label: {predicted_class}')
	break