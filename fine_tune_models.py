import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    Args:
        model: Model to train.
        train_loader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer for training.
        device: Device.
        num_epochs: Number of epochs.

    Returns:
        Trained model.
    """
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    return model

def test_model(model, test_loader, device):
    """
  Args:
        model: Model to test.
        test_loader: Test dataloader.
        device: Device.

    Returns:
        Predictions and true labels.
    """
    model.to(device)
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels

def train_and_test_all_models(train_loader, test_loader, device, num_epochs=10):
    """
    Args:
        train_loader: Train dataloader.
        test_loader: Test dataloader.
        device: Device.
        num_epochs: Number of epochs.

    Returns:
        A dictionary with model names as keys and predictions as values.
    """
    models_dict = {
        "ResNet18": models.resnet18(pretrained=True),
        "VGG16": models.vgg16(pretrained=True),
        "InceptionV3": models.inception_v3(pretrained=True, aux_logits=False),
        "VGG19": models.vgg19(pretrained=True)
    }
    
    num_classes = len(set([label for _, label in train_loader.dataset]))
    results = {}

    for model_name, model in models_dict.items():
        print(f"Training {model_name}...")
        
        # Modify the final layer for the specific number of classes
        if "Inception" in model_name:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        model = train_model(model, train_loader, criterion, optimizer, device, num_epochs)
        
        # Save the trained model
        torch.save(model.state_dict(), f"{model_name}_cancer_model.pth")
        print(f"{model_name} saved.")

        # Test the model
        print(f"Testing {model_name}...")
        predictions, true_labels = test_model(model, test_loader, device)
        results[model_name] = (predictions, true_labels)
    
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100

    # Train and test models
    results = train_and_test_all_models(train_loader, test_loader, device, num_epochs)

    # Save results for voting mechanism
    print("All models trained and tested. Results are ready for voting.")
  
