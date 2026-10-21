# CNN Feature Extraction and SVM Image Classification

## Overview
This project explores image classification on the CIFAR-10 dataset by combining Convolutional Neural Networks (CNNs) with Support Vector Machines (SVMs).

Three pretrained CNN architectures—ResNet50, MobileNetV2, and EfficientNetB0—are used to extract image features. SVM classifiers are then trained on these features and evaluated on the CIFAR-10 test set.

## Key Features
- CNN-based feature extraction using transfer learning.
- SVM classification using features extracted from pretrained models.
- Performance comparison across three CNN architectures.
- Evaluation using test accuracy and training time.
- Reproducible experiments using a fixed random seed.

## Models
- ResNet50
- MobileNetV2
- EfficientNetB0

## Dataset
CIFAR-10, containing 60,000 color images across 10 classes.

## Best Result
ResNet50 feature extraction combined with an SVM classifier achieved **88.21% test accuracy** in the current experiment.

## Tech Stack
Python, TensorFlow, Keras, Scikit-learn, NumPy, Pandas, Matplotlib, and Streamlit (for the application interface).

## Objective
To compare CNN architectures as feature extractors and evaluate how effectively their learned representations support traditional machine learning classification using SVM.