### Smart Prediction of Dry Eye Disease using Machine Learning

## 📌 Project Overview
Dry Eye Disease (DED) is a common ocular condition that affects tear film stability and ocular surface health. Early identification is important to prevent discomfort, visual disturbances, and long-term complications. This project applies machine learning techniques to predict Dry Eye Disease using clinical and diagnostic features. Multiple classification models, including Random Forest, Support Vector Machine, XGBoost, and LightGBM, are trained and evaluated. To improve model performance and interpretability, various feature selection techniques such as ANOVA F-test, Random Forest feature importance, Mutual Information, and Permutation Importance are implemented and compared.

## 🎯 Objective
The objective of this project is to develop an accurate and reliable machine learning-based system for predicting Dry Eye Disease using clinical data. The project aims to evaluate and compare multiple classification algorithms, analyze the impact of different feature selection methods on model performance, and identify the most informative features contributing to disease prediction. The final goal is to build an optimized and interpretable prediction model that can assist in early screening and decision support.

## 🗂️ Dataset
- Source: Kaggle
- Description: Dataset contains lifestyle and clinical data
- Format: Excel

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit for Deployment

## 🧠 Machine Learning Approach
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Feature selection using ANOVA F-test, Random Forest Feature Importance, Permutation importance, mutual information
- CalibratedClassifierCV for accurate probability values
- Model training (Random Forest, XGBoost, LightGBM, SVM)
- Model evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC Curve)

## 📊 Results
- comparing the results of four models
- evaluating the model performance

## Demo Link
https://dry-eye-disease-prediction-using-machine-learning.streamlit.app/
