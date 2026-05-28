---
title: "Machine Learning Fundamentals"
tags: [ml, concepts, basics]
date: 2025-01-15
---

## Supervised Learning

Supervised learning is a type of machine learning where the model is trained on labeled data. The algorithm learns to map input features to output labels based on example input-output pairs.

Key algorithms include:
- Linear Regression: predicts continuous values
- Logistic Regression: binary classification
- Decision Trees: hierarchical decision rules
- Support Vector Machines: finds optimal hyperplanes

## Unsupervised Learning

Unsupervised learning deals with unlabeled data. The algorithm must find patterns and structure in the input data without explicit guidance.

Common techniques:
- K-Means Clustering: partitions data into K clusters
- Principal Component Analysis (PCA): dimensionality reduction
- Autoencoders: neural network-based representation learning

## Transfer Learning

Transfer learning leverages pre-trained models on new but related tasks. Instead of training from scratch, you fine-tune an existing model. This drastically reduces the amount of data and compute required.

The typical workflow:
1. Take a model pre-trained on a large dataset (e.g., ImageNet)
2. Remove the task-specific layers
3. Add new layers for your target task
4. Fine-tune on your smaller dataset

This is especially powerful in computer vision and NLP, where models like [[ResNet]] and [[BERT]] serve as common starting points.
