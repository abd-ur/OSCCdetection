Cancer Image Classification using Pretrained Models with Ensemble Learning
This project utilizes four popular pretrained deep learning models—ResNet18, VGG16, InceptionV3, and VGG19—applied to cancer image classification. The goal is to predict whether an image depicts carcinoma or not. The final predictions are made using an ensemble learning approach, employing majority voting to combine the predictions of each individual model for better accuracy.

Key Steps:
Pretrained Models: We use pretrained versions of ResNet18, VGG16, InceptionV3, and VGG19.
Transfer Learning: The models are fine-tuned for the cancer classification task by modifying their final classification layers to suit the specific number of classes (e.g., binary classification: carcinoma vs. non-carcinoma).
Ensemble Voting: Once the models are trained, predictions from all four models are combined using majority voting to arrive at the final decision for each image.
Accuracy Evaluation: The ensemble model's performance is evaluated on a test dataset, and the results are compared with the true labels.

Dataset
The dataset used in this project consists of labeled Oral Single Cell Carcinoma images both normal and OSCC

Ensemble Voting Workflow:
Model Loading: The pretrained models are loaded from the saved .pth files.
Prediction Collection: For each test image, each model generates a prediction.
Majority Voting: The predictions are aggregated, and the class that appears most frequently is chosen as the final prediction.

If you'd like to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request. Any improvements, bug fixes, or additional model implementations are welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
The pretrained models are provided by PyTorch's torchvision.
Dataset is provided by kaggle (https://www.kaggle.com/datasets/viditgandhi/oral-cancer-dataset)
