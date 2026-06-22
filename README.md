# Fraud Detection - End-to-End Machine Learning Pipeline

## Purpose

This repository contains the implementation of an end-to-end fraud detection pipeline. The goal is to predict the probability that an online financial transaction is fraudulent using the `isFraud` binary label from the IEEE-CIS Fraud Detection dataset.

## Project Overview

The pipeline covers the full machine learning workflow:

- Data loading from `train_transaction.csv`
- Data preprocessing including categorical encoding, missing value imputation, and handling class imbalance with SMOTE
- Hyperparameter tuning using Optuna
- Model evaluation using classification metrics
- Experiment tracking with MLflow

A 50% random sample of the dataset is used to reduce memory consumption during training.

## Model and Results

**Model: Random Forest Classifier**

Random Forest was selected as the classification model. Hyperparameters were optimized using Optuna over 10 trials, with AUC-ROC as the objective metric.

Tuned hyperparameters:

- `n_estimators`: search range 50 to 200
- `max_depth`: search range 3 to 10
- `min_samples_split`: search range 2 to 10

Evaluation metrics on the validation set:

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.8776 |
| Precision (Fraud) | 0.24 |
| Recall (Fraud) | 0.60 |
| F1-Score (Fraud) | 0.34 |
| Accuracy | 0.92 |

The confusion matrix and top 15 feature importances are visualized within the notebook.

## How to Navigate

1. Open `Tugas 1.ipynb` in Google Colab.
2. Run the first cell to install required packages (`optuna`, `mlflow`, `imbalanced-learn`).
3. Run the imports cell.
4. In the **Data Loading** section, upload `train_transaction.csv` when the file picker appears.
5. Proceed through the cells in order:
   - **Data Preprocessing** — encoding, imputation, train/val split, SMOTE
   - **Machine Learning Model** — Optuna tuning, Random Forest training, MLflow logging, evaluation report, confusion matrix, feature importance plot
