import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import torch.optim as optim
from torchvision import transforms
from FPN_CBAM import FPN
from utils import trainer,tester
from ImageDataset import ImageDataset

## ------------------------------------
## Dataset & Dataloader


transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize images to 224x224 pixels
    transforms.RandomHorizontalFlip(), # Apply horizontal flip for data augmentation
    transforms.ToTensor(), # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet mean and std
])
root_dir = 'E:\BUET Files\Celia MAM Biomedical Signal Processing\RAtCapsNet\data\labelled_images'
full_dataset = ImageDataset(root_dir=root_dir, transform=transform)

# Calculate lengths for training and testing sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


## ------------------------------------
## Check and print of CUDA availability

if torch.cuda.is_available():
    # Get the current CUDA device (GPU)
    gpu_id = torch.cuda.current_device()
    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(gpu_id)
    # Get the capability of the GPU
    gpu_capability = torch.cuda.get_device_capability(gpu_id)
    print(f"CUDA is available. GPU: {gpu_name}")
    print(f"GPU Capability: {gpu_capability}")
else:
    print("CUDA is not available.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## ------------------------------------
## Model

model=FPN(Bottleneck, [2,2,2,2])
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


## Train the model

num_epochs = 10
best_acc = 0.0  # Initialize the best accuracy to 0

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')

    # Train the model for one epoch
    train_loss, train_acc = trainer(model, train_loader, criterion, optimizer, device)

    # Evaluate the model on the test set
    test_loss, test_acc = tester(model, test_loader, criterion, device)

    # Check if the current test accuracy is greater than the best accuracy so far
    if test_acc > best_acc:
        best_acc = test_acc
        # Save the model state dict if the accuracy improves
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Saved Best Model with Accuracy: {best_acc:.4f}')
    else:
        print(f'No improvement in accuracy: {test_acc:.4f}, Best Accuracy: {best_acc:.4f}')


print('... Training complete ...')