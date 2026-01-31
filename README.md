# Fraud Detection Model Evaluation

This project focuses on detecting **fraudulent credit card transactions** using the **PaySim synthetic financial dataset**.  
The objective is to **build, compare, and evaluate multiple machine learning models** and identify the most effective approach for **highly imbalanced data**.

A live **Streamlit dashboard** is included to interactively explore model performance.

---

## Live Demo (Streamlit App)
👉 https://fraudetectionmodels.streamlit.app/

---

## Dataset
The dataset used in this project is the **PaySim Synthetic Financial Dataset for Fraud Detection**, containing over **6 million simulated transactions**.

- Source: Kaggle  
  https://www.kaggle.com/datasets/ealaxi/paysim1
- Target variable:
  - `0` → Not Fraud
  - `1` → Fraud

The dataset is **extremely imbalanced**, closely resembling real-world financial transaction data.

---

## Project Overview
The project notebook covers the complete machine learning workflow:

- Data loading and preprocessing  
- Exploratory data analysis (EDA)  
- Feature engineering  
- Handling class imbalance  
- Training multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Multi-Layer Perceptron (MLP)
- Hyperparameter tuning  
- Model evaluation using:
  - Precision, Recall, F1-score
  - ROC Curve & AUC
  - Confusion Matrix

To address class imbalance, multiple techniques were explored, including:
- `scale_pos_weight`
- SMOTE

---

## Best Performing Model
After evaluating all models on validation and test sets, **XGBoost** demonstrated the most **balanced and reliable performance**, achieving:

- **F1-score:** ~0.88  
- **ROC-AUC:** ~0.99  

This model showed the best trade-off between **precision and recall** for fraud detection.

---

## Streamlit Dashboard
The Streamlit application allows users to:

- Load a built-in demo dataset or upload a custom CSV
- Compare multiple fraud detection models
- Visualize performance through:
  - ROC curves
  - Precision–Recall curves
  - Confusion matrices
- Export predictions with fraud probabilities

