# Project_ML
# Titanic Disaster: Classification & Regression Analysis

This project performs a comprehensive analysis of the Titanic dataset using both **Classification** (to predict survival) and **Regression** (to predict ticket fares). It includes data preprocessing, model training, and performance comparison using various visualization techniques.

## 🚀 Overview

The goal is to apply multiple machine learning algorithms to the famous Titanic dataset and evaluate their effectiveness. 

- **Classification Task:** Predict whether a passenger survived (1) or not (0).
- **Regression Task:** Predict the continuous value of the ticket `Fare`.

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## 📊 Methodology

### 1. Data Preprocessing
- **Feature Selection:** Removed non-informative columns like `PassengerId`, `Name`, `Ticket`, and `Cabin`.
- **Missing Value Imputation:**
  - Filled missing `Age` with the median.
  - Filled missing `Embarked` with the mode.
- **Encoding:** - `Sex` converted to binary labels.
  - `Embarked` transformed using One-Hot Encoding.

### 2. Models Implemented

#### Classification
- **Logistic Regression:** Baseline linear model for binary classification.
- **Random Forest Classifier:** Ensemble-based decision trees for capturing non-linear patterns.
- **Support Vector Classifier (SVC):** Finding the optimal hyperplane for separation.

#### Regression
- **Linear Regression:** Standard approach for modeling the relationship between features and Fare.
- **Random Forest Regressor:** Handles complex interactions between features.
- **Support Vector Regressor (SVR):** Regression based on support vector machines.

## 📈 Performance Comparison

### Classification Results
| Model | Accuracy |
| :--- | :---: |
| Logistic Regression | 81% |
| Random Forest | 81% |
| SVM | 65% |

### Regression Results
| Model | R² Score | RMSE |
| :--- | :---: | :---: |
| Linear Regression | 0.40 | 30.53 |
| Random Forest | -0.10 | 41.30 |
| SVR | -0.10 | 41.34 |

## 🖼️ Visualizations
The project generates several plots to compare model performance:
1. **Classification Accuracy Comparison**
2. **Regression R² Score Analysis**
3. **Regression Root Mean Squared Error (RMSE) Trends**

## 📂 Project Structure
- `train.csv`: Training dataset.
- `test.csv`: Testing dataset.
- `analysis.py`: Main script for training and evaluation.
- `README.md`: Project documentation.
