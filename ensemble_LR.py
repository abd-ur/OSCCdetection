import torch
import numpy as np
from collections import Counter
from torchvision import models

def load_model(model_name, num_classes, device, checkpoint_path):
    """
    Args:
        model_name (str): Model name.
        num_classes (int): Number of classes.
        device (torch.device): Device.
        checkpoint_path (str): Path to the saved model weights.

    Returns:
        model: The modified model with loaded weights.
    """
    # Initialize model
    if model_name == "ResNet18":
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "VGG16":
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == "InceptionV3":
        model = models.inception_v3(pretrained=True, aux_logits=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "VGG19":
        model = models.vgg19(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Load model weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def majority_voting(predictions_list):
    """
    Perform majority voting to determine the final class for each sample.
    Args:
        predictions_list (list): List of predictions from multiple models.

    Returns:
        final_predictions (list): Final predictions after majority voting.
    """
    final_predictions = []
    for preds in zip(*predictions_list):
        # Use Counter to find the most common class across models
        most_common = Counter(preds).most_common(1)[0][0]
        final_predictions.append(most_common)
    return final_predictions

def ensemble_voting(test_loader, model_paths, model_names, num_classes, device):
    """
    Args:
        test_loader: Test dataloader.
        model_paths (list): List of file paths to saved model weights.
        model_names (list): Model names.
        num_classes (int): Number of classes.
        device (torch.device): Device.

    Returns:
        final_predictions (list): Final predictions for the test set.
        true_labels (list): True labels for the test set.
    """
    predictions_list = []

    # Load each model and collect predictions
    for model_name, model_path in zip(model_names, model_paths):
        print(f"Loading {model_name} from {model_path}...")
        model = load_model(model_name, num_classes, device, model_path)

        predictions = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
        
        predictions_list.append(predictions)

    # Perform majority voting
    final_predictions = majority_voting(predictions_list)

    # Extract true labels
    true_labels = []
    for _, labels in test_loader:
        true_labels.extend(labels.numpy())

    return final_predictions, true_labels

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to saved model weights
    model_paths = [
        "ResNet18_cancer_model.pth",
        "VGG16_cancer_model.pth",
        "InceptionV3_cancer_model.pth",
        "VGG19_cancer_model.pth"
    ]
    model_names = ["ResNet18", "VGG16", "InceptionV3", "VGG19"]
    num_classes = 2  # carcinoma or not

    final_predictions, true_labels = ensemble_voting(test_loader, model_paths, model_names, num_classes, device)

    # Evaluate accuracy
    accuracy = np.mean(np.array(final_predictions) == np.array(true_labels))
    print(f"Ensemble Accuracy: {accuracy * 100:.2f}%")
