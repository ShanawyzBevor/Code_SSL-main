import tensorflow as tf
# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image
from model import VNet
from dataset2 import LAHeart, RandomCrop, RandomNoise, RandomRotFlip, ToTensor

print("hllow world")

# Dice Loss (Only for Supervised)
def dice_loss(score, target):
    target = target.long()
    score = F.softmax(score, dim=1)
    smooth = 1e-5
    target_onehot = torch.zeros_like(score).scatter_(1, target.unsqueeze(1), 1)
    intersect = torch.sum(score * target_onehot)
    y_sum = torch.sum(target_onehot * target_onehot)
    z_sum = torch.sum(score * score)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1 - dice  # loss

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load subset of labeled data based on the percentage
def get_labeled_data_percentage(dataset, label_percentage=1.0):
    total_labeled_data = len(dataset)
    num_labeled_samples = int(total_labeled_data * label_percentage)
    
    # Randomly sample a subset of labeled data
    sampled_labeled_data = torch.utils.data.Subset(dataset, torch.randperm(total_labeled_data)[:num_labeled_samples])
    
    return sampled_labeled_data

# Model
model_a = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device)
model_b = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device)

# DataLoader
batch_size = 4
label_percentage = 0.2  # Change this to control the amount of labeled data (0.5 for 50%, 1.0 for 100%, etc.)

# Apply the function to get the desired labeled data subset
train_transform = transforms.Compose([
    RandomRotFlip(),
    RandomNoise(),
    RandomCrop((112, 112, 80)),
    ToTensor(),
])

# Load training and unlabelled datasets
trainset = LAHeart(split='Training Set', label=True, transform=train_transform)
unlabelled_trainset = LAHeart(split='Training Set', label=False, transform=train_transform)

# Get subset of labeled data based on percentage
trainset_subset = get_labeled_data_percentage(trainset, label_percentage)

# Modify the DataLoader with reduced worker processes (2 instead of 4)
trainloader = torch.utils.data.DataLoader(trainset_subset, batch_size=batch_size, shuffle=True, num_workers=2)
unlabelled_trainloader = torch.utils.data.DataLoader(unlabelled_trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# Optimizer & Scheduler
Max_epoch = 800
learn_rate = 0.0005
optimizer_a = optim.AdamW(model_a.parameters(), lr=learn_rate, weight_decay=1e-4)
optimizer_b = optim.AdamW(model_b.parameters(), lr=learn_rate, weight_decay=1e-4)
scheduler_a = optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=Max_epoch)
scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=Max_epoch)

# TensorBoard
writer = SummaryWriter()

# Create directories for saving images
os.makedirs("./images", exist_ok=True)  # To store images
os.makedirs("./predictions", exist_ok=True)  # To store predictions
os.makedirs("./labels", exist_ok=True)  # To store ground truth labels

# Helper function to normalize images to [0, 255] range for saving
def normalize_and_convert_to_image(slice_data):
    # Normalize to [0, 1] range
    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize to 0-1 range
    
    # Scale to [0, 255] and convert to uint8
    slice_data = np.uint8(slice_data * 255)
    
    # Check min/max values after scaling
    print(f"Normalized Min value: {np.min(slice_data)}, Max value: {np.max(slice_data)}")
    
    return Image.fromarray(slice_data).convert('L')

# Check if the data is being loaded correctly
print(f"Training set size: {len(trainloader.dataset)}")
print(f"Unlabelled set size: {len(unlabelled_trainloader.dataset)}")

# Print the first batch to check data loading
for batch_idx, sample in enumerate(trainloader):
    print(f"Batch {batch_idx}: Image shape: {sample['image'].shape}")
    break  # Just to check the first batch

# Training loop
for epoch in range(Max_epoch):
    print(f'\nEpoch {epoch+1}/{Max_epoch}')
    print('-' * 30)
    model_a.train()
    model_b.train()

    total_sup_loss = 0.0
    total_dice = 0.0

    for batch_idx, sample in tqdm(enumerate(trainloader), total=len(trainloader)):
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
        images = sample["image"].to(device)
        labels = sample["label"].to(device)

        outputs_a = model_a(images)
        outputs_b = model_b(images)

        seg_loss_a = F.cross_entropy(outputs_a, labels)
        seg_loss_b = F.cross_entropy(outputs_b, labels)
        dice_loss_a = dice_loss(outputs_a, labels)

        sup_loss = seg_loss_a + seg_loss_b + dice_loss_a
        sup_loss.backward()
        optimizer_a.step()
        optimizer_b.step()

        total_sup_loss += sup_loss.item()
        total_dice += (1 - dice_loss_a.item())

    avg_sup_loss = total_sup_loss / len(trainloader)
    avg_dice = total_dice / len(trainloader)

    # Print out the metrics at the end of each epoch
    print(f"Epoch {epoch+1}, Labelled Dice: {avg_dice:.4f}, Loss: {avg_sup_loss:.4f}")
    sys.stdout.flush()

    writer.add_scalar("Supervised Loss", avg_sup_loss, epoch)
    writer.add_scalar("Dice Accuracy/Labelled", avg_dice, epoch)

    # CPS for Unlabelled Data
    if epoch >= 100:
        print(f"--- Starting CPS for Unlabelled Data at Epoch {epoch+1} ---")
        total_unsup_loss = 0.0
        total_unsup_dice = 0.0

        for batch_idx, sample in tqdm(enumerate(unlabelled_trainloader), total=len(unlabelled_trainloader)):
            optimizer_a.zero_grad()
            optimizer_b.zero_grad()
            images = sample["image"].to(device)
            labels = sample["label"].to(device)  # pseudo labels or placeholders

            outputs_a = model_a(images)
            outputs_b = model_b(images)

            _, hardlabel_a = torch.max(outputs_a, dim=1)
            _, hardlabel_b = torch.max(outputs_b, dim=1)

            unsup_cps_loss = 0.01 * (F.cross_entropy(outputs_a, hardlabel_b) + F.cross_entropy(outputs_b, hardlabel_a))
            unsup_cps_loss.backward()
            optimizer_a.step()
            optimizer_b.step()

            dice_unsup = dice_loss(outputs_a, labels)
            total_unsup_dice += (1 - dice_unsup.item())
            total_unsup_loss += unsup_cps_loss.item()

        avg_unsup_loss = total_unsup_loss / len(unlabelled_trainloader)
        avg_unsup_dice = total_unsup_dice / len(unlabelled_trainloader)

        # Print out the metrics for unlabelled data
        print(f"Epoch {epoch+1}, Unlabelled Dice: {avg_unsup_dice:.4f}, Unlabelled Loss: {avg_unsup_loss:.4f}")
        sys.stdout.flush()

        writer.add_scalar("Unsupervised CPS Loss", avg_unsup_loss, epoch)
        writer.add_scalar("Dice Accuracy/Unlabelled", avg_unsup_dice, epoch)

    scheduler_a.step()
    scheduler_b.step()

    if (epoch + 1) % 20 == 0:
        print("Saving model checkpoint...")
        torch.save(model_a.state_dict(), f"model_a_epoch_{epoch+1}.pth")
        torch.save(model_b.state_dict(), f"model_b_epoch_{epoch+1}.pth")

writer.flush()
writer.close()
