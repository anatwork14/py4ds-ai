import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load_metadata(metadata_path="dataset_metadata.json"):
    """Loads the metadata JSON generated from Phase 1."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata

def get_transforms(metadata):
    """Sets up PyTorch transforms for train and test splits based on metadata."""
    norm_mean = metadata["normalization"]["mean"]
    norm_std = metadata["normalization"]["std"]
    input_size = tuple(metadata["input_size"])

    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # Modest jitter to add lighting robustness
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    return train_transform, test_transform

def get_dataloaders(metadata, train_transform, test_transform, batch_size=32, valid_files=None):
    """Creates PyTorch DataLoaders for train and test splits.

    Parameters
    ----------
    valid_files : set, optional
        If provided, only images whose path is in this set are included.
        Pass ``set(clean_df['filepath'])`` to use the cleaned dataset.
    """
    dataset_dir = "dataset"
    train_dir = os.path.join(dataset_dir, metadata["splits"]["train"])
    test_dir = os.path.join(dataset_dir, metadata["splits"]["test"])

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)

    if valid_files is not None:
        train_dataset = Subset(train_dataset, [i for i, (p, _) in enumerate(train_dataset.samples) if os.path.abspath(p) in valid_files])
        test_dataset  = Subset(test_dataset,  [i for i, (p, _) in enumerate(test_dataset.samples)  if os.path.abspath(p) in valid_files])

    # shuffle=False is critical to keep extracted features aligned with their labels
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_feature_extractor(device):
    """Initializes a pre-trained ResNet18 model as a feature extractor."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Remove the final classification layer (fc)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval() # Freeze dropout and batchnorm layers
    return model

def batch_extract_features(dataloader, model, device):
    """Helper function to extract features in batches using the feature extractor."""
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model(images)
            # Flatten features from (batch_size, 512, 1, 1) to (batch_size, 512)
            features = features.view(features.size(0), -1)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features_list), np.concatenate(labels_list)