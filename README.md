# Fraud Detection Model

This project focuses on detecting fraudulent transactions in the well-known credit card fraud dataset. The goal was to build and compare several machine learning models and select the one that performs best on highly imbalanced data.

# Dataset

The dataset used in this project is the Paysim Synthetic Financial Dataset for  Fraud Detection, with more than 6 million transactions,  available on Kaggle:
https://www.kaggle.com/datasets/ealaxi/paysim1

# Overview

The notebook includes the full workflow:

Data loading and preprocessing

Exploratory data analysis

Feature engineering

Handling class imbalance

Training multiple models (Logistic Regression, Random Forest, XGBoost, MultiLayer Perceptron)

Hyperparameter tuning

Evaluation using precision, recall, F1-score, ROC curve and AUC

Because the dataset is extremely imbalanced, additional techniques such as scale_pos_weight, undersampling and SMOTE were tested to improve minority-class performance.

# Best Model

After comparing all models on the validation and test sets, XGBoost delivered the most balanced and reliable results, achieving a high F1-score(0.88) and an AUC of about 0.99.

# Live Demo (Streamlit App)
https://fraudetectionmodels.streamlit.app/
