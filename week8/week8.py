from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assume df is your dataset and target_col is the binary target
X = df.drop('target_col', axis=1)
y = df['target_col']

# One-hot encode and scale
X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Fit model
model = LogisticRegression()
model.fit(X_train, y_train)




y_probs = model.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.5, 0.7]
for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    print(f"Threshold: {t}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}\n")

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# PART 2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertias = []
silhouettes = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot
plt.plot(k_values, inertias, marker='o')
plt.title('Inertia vs k')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()

plt.plot(k_values, silhouettes, marker='x')
plt.title('Silhouette Score vs k')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.show()

# 2. What if You Don’t Scale?
# Without scaling, features with larger ranges dominate distance calculations.
# This leads to poor clustering (inertia might still decrease, but silhouette scores likely drop).

# 3. Is There a Right k?
# Not strictly. Look for an elbow in the inertia plot and peak in silhouette score.
# The "right" k depends on domain knowledge and the stability of cluster assignments.

