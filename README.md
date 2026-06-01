# IDEA2
<div align="center">

# 💼 GlobaTech Salary Intelligence System

### *A machine learning pipeline that processes global salary survey data to classify income levels (Low / Medium / High), analyse workforce patterns, and identify key drivers of annual income using data cleaning, feature engineering, clustering, and ensemble learning models.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=flat-square&logo=numpy)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=flat-square)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-EDA-4c72b0?style=flat-square)](https://seaborn.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📌 Project Overview

GlobaTech, a multinational HR analytics organisation, aims to understand **global salary distribution patterns** using workforce survey data collected in 2024.

This project builds a complete **end-to-end machine learning pipeline** to:

- Clean and preprocess global salary survey data
- Engineer meaningful workforce features
- Classify income into tiers:
  - Low (≤ 70,000 USD)
  - Medium (70,000–100,000 USD)
  - High (> 100,000 USD)
- Apply clustering techniques to group behavioural patterns
- Train ensemble machine learning models for salary prediction
- Identify key drivers influencing income globally

---

## 📊 Dataset Description

The dataset contains global employee survey responses including:

- Age
- Gender
- Race
- Industry type
- Job role / function
- Experience level
- Education qualification
- Work location
- Currency & compensation data
- Job arrangement (remote/on-site/hybrid)

⚠️ Dataset challenges:
- Missing values
- Mixed categorical formats
- Currency inconsistencies
- Outliers in salary values
- High dimensional categorical variables

---

## ⚙️ Project Workflow

```text
Raw Survey Data
   ↓
Data Cleaning & Column Renaming
   ↓
Categorical Grouping & Feature Engineering
   ↓
Currency Normalisation (USD conversion)
   ↓
Exploratory Data Analysis (EDA)
   ↓
Clustering (Hierarchical)
   ↓
Feature Selection
   ↓
Machine Learning Models
   ↓
Evaluation & Performance Metrics
   ↓
Salary Tier Prediction (Low / Medium / High)
```

---

## 🧠 Data Cleaning & Feature Engineering

### Categorical Grouping Strategy
Industries grouped into macro categories:
- Education & Tech
- Government & Manufacturing
- Healthcare & Media
- Marketing & Advertising
- Other

---

### Race Cleaning Example
```python
if ',' in race:
    return "Multiple Races"
else:
    return race.split(',')[0]
```

---

### Currency Normalisation
All salaries converted into USD and cleaned:
- Removed null values
- Removed invalid entries
- Standardised salary scale

---

## 📊 Exploratory Data Analysis (EDA)

Key insights:
- Salary strongly depends on experience and industry
- Tech and finance roles have highest income distribution
- Remote vs onsite roles show salary variation
- Education improves income probability but is not sole driver

---

## 📌 Clustering Analysis

### Hierarchical Clustering
Used for grouping work modes and job patterns:

```python
AgglomerativeClustering(n_clusters=4)
```

### Label Encoding
```python
LabelEncoder()
```

---

## 💰 Salary Tier Definition

| Tier | Range |
|------|------|
| Low | ≤ 70,000 USD |
| Medium | 70,000–100,000 USD |
| High | > 100,000 USD |

```python
pd.cut(bins=[0, 70000, 100000, float('inf')])
```

---

## 🤖 Machine Learning Models

### Ensemble Strategy
Models used:
- Logistic Regression
- KNN
- Decision Tree

✔ Hard voting classifier improves robustness  
✔ Reduces bias and variance  
✔ Handles nonlinear patterns

---

## 🌲 Random Forest Models

### Classifier
Predicts salary tier (Low / Medium / High)

### Regressor
Identifies key income drivers globally

- 300 trees used for stability
- Feature importance extracted via:
```python
rf.feature_importances_
```

---

## 📊 Model Performance

| Class | Precision | Recall | F1-score |
|------|----------|--------|----------|
| High | 0.50 | 1.00 | 0.67 |
| Low | 0.82 | 0.82 | 0.82 |
| Medium | 0.00 | 0.00 | 0.00 |

- Accuracy: **73%**
- Macro F1: **0.69**

---

## 🔍 Key Findings

- Income strongly depends on experience & industry
- Medium salary class is hardest to predict
- Tech sector dominates high-income distribution
- Geography significantly impacts salary levels
- Model performs best on Low and High classes

---

## 📂 Project Structure

```text
globatech-salary-system/
│
├── salary_analysis.ipynb
├── dataset.csv
├── README.md
├── requirements.txt
│
├── models/
│   ├── random_forest.pkl
│   └── ensemble_model.pkl
│
├── reports/
│   └── salary_report.pdf
│
└── visuals/
    ├── eda.png
    └── feature_importance.png
```

---

## 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/globatech-salary-system.git
cd globatech-salary-system
pip install -r requirements.txt
```

---

## ▶️ Run Project

```bash
n1364759-foai-coursework(1)(1).ipynb
```
## Kaggle Link
https://www.kaggle.com/code/pratham2516/n1364759-foai-coursework
---

## 🔮 Future Improvements

- Handle class imbalance (SMOTE)
- XGBoost / LightGBM models
- Hyperparameter tuning (GridSearchCV)
- SHAP explainability analysis
- Streamlit dashboard deployment
- API for real-time predictions

---

## 📄 License

MIT License

---

## ⭐ Author

GlobaTech Salary Intelligence System  
Machine Learning Portfolio Project
