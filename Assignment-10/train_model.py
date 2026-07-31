"""
train_model.py
----------------
Task 1: Data Understanding and Preprocessing
Task 2: Model Development

Loads the heart disease dataset, explores it, preprocesses it,
trains a classification model, evaluates it, and saves it with joblib.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---------------------------------------------------------
# Task 1: Data Understanding and Preprocessing
# ---------------------------------------------------------

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")

# 2. Display the first five records
print("First 5 records:")
print(df.head())

# 3. Identify numerical features and target variable
target_col = "target"
numerical_features = [col for col in df.columns if col != target_col]
print("\nNumerical features:", numerical_features)
print("Target variable:", target_col)

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# ---------------------------------------------------------
# Task 2: Model Development
# ---------------------------------------------------------

# Build a classification model (Random Forest)
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy Score: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model using Joblib
joblib.dump(model, "model.pkl")
# Also save the exact feature order the model expects, for the API
joblib.dump(numerical_features, "feature_names.pkl")

print("\nModel saved as model.pkl")
print("Feature order saved as feature_names.pkl")
