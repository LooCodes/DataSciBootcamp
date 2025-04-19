import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#  Load data
data = pd.read_csv("employee.csv")

#  Drop unnecessary columns
data = data.drop(columns=['id', 'timestamp', 'country'])

# Handle missing numeric values
data['hours_per_week'].fillna(data['hours_per_week'].median(), inplace=True)
data['telecommute_days_per_week'].fillna(data['telecommute_days_per_week'].median(), inplace=True)

# Drop rows with missing categorical values
data = data.dropna()

# Target and features
y = data['salary']
X = data.drop(columns=['salary'])

# Binary variable encoding
binary_cols = ['is_manager', 'certifications']
for col in binary_cols:
    X[col] = X[col].replace({'Yes': 1, 'No': 0})

# One-hot encode categorical features
cat_cols = [col for col in X.columns if X[col].dtype == 'object' and col not in binary_cols]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numeric features
num_cols = ['job_years', 'hours_per_week', 'telecommute_days_per_week']
scaler = StandardScaler()
scaler.fit(X_train[num_cols])               # Fit on train
X_train[num_cols] = scaler.transform(X_train[num_cols])  # Transform train
X_test[num_cols] = scaler.transform(X_test[num_cols])    # Transform test

# Train model
reg = LinearRegression()
reg.fit(X_train, y_train)

#Predict and eval
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_pred, y_test) / np.mean(y_test)
print("Normalized Mean Squared Error on Test Set:", mse)