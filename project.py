import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Train head:")
print(train.head())
print("\nTrain info:")
print(train.info())
print("\nMissing values in Train:")
print(train.isnull().sum())

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Preprocessing
df = train.copy()

# Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# --- Classification Setup ---
X_clf = df.drop('Survived', axis=1)
y_clf = df['Survived']
X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Models
clf_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC()
}

clf_results = {}
for name, model in clf_models.items():
    model.fit(X_train_c, y_train_c)
    preds = model.predict(X_val_c)
    clf_results[name] = accuracy_score(y_val_c, preds)

# --- Regression Setup ---
# Let's predict 'Fare'
X_reg = df.drop('Fare', axis=1)
y_reg = df['Fare']
X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR()
}

reg_results_r2 = {}
reg_results_rmse = {}
for name, model in reg_models.items():
    model.fit(X_train_r, y_train_r)
    preds = model.predict(X_val_r)
    reg_results_r2[name] = r2_score(y_val_r, preds)
    reg_results_rmse[name] = np.sqrt(mean_squared_error(y_val_r, preds))

# --- Visualization ---

# Plot 1: Classification Accuracy
plt.figure(figsize=(10, 5))
sns.barplot(x=list(clf_results.keys()), y=list(clf_results.values()), palette='viridis')
plt.title('Classification Model Comparison (Accuracy)')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(clf_results.values()):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.savefig('clf_comparison.png')

# Plot 2: Regression R2 Score
plt.figure(figsize=(10, 5))
sns.barplot(x=list(reg_results_r2.keys()), y=list(reg_results_r2.values()), palette='magma')
plt.title('Regression Model Comparison (R2 Score)')
plt.ylabel('R2 Score')
for i, v in enumerate(reg_results_r2.values()):
    plt.text(i, v + 0.02 if v > 0 else v - 0.05, f'{v:.2f}', ha='center')
plt.savefig('reg_comparison_r2.png')

# Plot 3: Regression RMSE
plt.figure(figsize=(10, 5))
sns.barplot(x=list(reg_results_rmse.keys()), y=list(reg_results_rmse.values()), palette='coolwarm')
plt.title('Regression Model Comparison (RMSE)')
plt.ylabel('RMSE (Lower is better)')
for i, v in enumerate(reg_results_rmse.values()):
    plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
plt.savefig('reg_comparison_rmse.png')

print("Classification Accuracy:", clf_results)
print("Regression R2 Scores:", reg_results_r2)
print("Regression RMSE:", reg_results_rmse)


