import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split

def load_image_paths_and_labels(image_dir, label_file):
    """
    Args:
        image_dir (str): OSCC image path.
        label_file (str): filename and label path.

    Returns:
        image_paths (list): List of image file paths.
        labels (list): List of corresponding labels.
    """
    image_paths = []
    labels = []

    with open(label_file, 'r') as f:
        for line in f:
            filename, label = line.strip().split(',')
            image_paths.append(os.path.join(image_dir, filename))
            labels.append(int(label))

    return image_paths, labels

class OSCCdataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of file paths to images.
            labels (list): List of corresponding labels.
            transform (callable): Transformations to apply to images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_data_loaders(image_dir, label_file, batch_size=16, test_size=0.2, random_state=42):
    """
    Args:
        image_dir (str): Directory containing OSCC/control images.
        label_file (str): File containing filenames and labels.
        batch_size (int, optional): Batch size.
        test_size (float, optional): Testing size.
        random_state (int, optional): Random seed for splitting data.

    Returns:
        train_loader (DataLoader): Train dataloader.
        test_loader (DataLoader): test dataloader.
    """
    # Load image paths and labels
    image_paths, labels = load_image_paths_and_labels(image_dir, label_file)

    # Split data into training and testing sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state
    )

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = OSCCdataset(train_paths, train_labels, transform=transform)
    test_dataset = OSCCdataset(test_paths, test_labels, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    image_dir = "path_to_image_directory"  # Replace with actual directory path
    label_file = "path_to_label_file.csv"  # Replace with actual label file

    train_loader, test_loader = create_data_loaders(image_dir, label_file)

    print("DataLoaders created:")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
