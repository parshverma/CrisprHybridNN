# CrisprHybridNN
# Hybrid Neural Networks for Predicting Cleavage Frequencies in CRISPR-Cas9 Gene Editing

This repository contains the code for a series of hybrid neural network architectures developed to predict cleavage frequencies in CRISPR-Cas9 gene editing. These models incorporate a combination of convolutional layers (CNN), recurrent layers (LSTM, GRU, and Bi-Directional LSTM), and attention mechanisms to analyze both sequence data (k-mer embeddings) and numeric features (energy levels, CG content). The models aim to improve prediction accuracy by leveraging the strengths of both CNNs and RNNs, coupled with the ability of attention mechanisms to focus on the most relevant parts of the input data.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture Details](#architecture-details)
- [Model Training and Evaluation](#model-training-and-evaluation)


## Project Overview

Predicting cleavage frequencies accurately is critical in CRISPR-Cas9 gene editing to minimize off-target effects and optimize on-target cleavage. The work presented here uses hybrid neural network architectures combining CNNs, LSTMs, Bi-LSTMs, GRUs, and attention mechanisms. The models are trained on CRISPR-related datasets, utilizing both sequence-based features and energy metrics to provide more accurate predictions.

## Architecture Details

This project includes three main architectures:

### 1. **Architecture 1** 
   - **Components**: CNN, LSTM, Bi-Directional LSTM, GRU.
   - **Purpose**: Baseline architecture for evaluating the effectiveness of different RNN types without attention.
   - **Models**: 
      - CNN
      - LSTM
      - Bi-Directional LSTM
      - GRU

### 2. **Architecture 2**
   - **Components**: CNN, LSTM, Bi-Directional LSTM, GRU.
   - **Purpose**: Hybrid models combining CNN for feature extraction from sequences with LSTM, Bi-LSTM, or GRU for temporal sequence modeling.
   - **Models**:
      - CNN + LSTM
      - CNN + Bi-Directional LSTM
      - CNN + GRU

### 3. **Architecture 3**
   - **Components**: CNN, LSTM, Bi-Directional LSTM, GRU, Attention Mechanism.
   - **Purpose**: Extends Architecture 2 by adding attention mechanisms to highlight the most relevant parts of the input sequences.
   - **Models**:
      - CNN + LSTM + Attention
      - CNN + Bi-Directional LSTM + Attention
      - CNN + GRU + Attention

## Model Training and Evaluation

### Data Preprocessing:
- **K-mer Generation**: Sequences are broken down into k-mers (k=2) to capture local sequence features.
- **Tokenization**: The k-mers are tokenized and padded to ensure uniform input length.
- **Feature Extraction**: CG content and energy features are extracted from sequences.

### Model Training:
- **SMOTE-Tomek**: Applied to address class imbalance in the training data.
- **Cross-Validation**: K-Fold cross-validation is used to evaluate model performance and determine the optimal threshold for binary classification.

### Evaluation Metrics:
- **Accuracy**: Overall performance of the model.
- **Precision**: Ability of the model to identify relevant cleavage events.
- **Recall**: Coverage of actual cleavage events by the model.
- **F1 Score**: Harmonic mean of precision and recall, used to select the optimal threshold.
- **AUC**: Area under the ROC curve, indicating the ability of the model to distinguish between classes.
- **Balanced Accuracy**: Takes into account both sensitivity and specificity.
