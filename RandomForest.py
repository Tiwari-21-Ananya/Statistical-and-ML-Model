
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
dataset = pd.read_csv("C:/Users/a/Desktop/AIML/AIML DATASET FINAL.csv", header=0)
print("First 5 rows of the dataset:")
print(dataset.head())

# Extract features and labels
X = dataset.iloc[:, 1:].values  # features
y = dataset.iloc[:, 0].values   # labels
feature_names = dataset.columns[1:]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0, stratify=y)

# Train Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot 1: Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Plot 2: Feature Importances
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Print sorted feature importances
print("\nFeature Importances:")
for idx in indices:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}")

# Plot feature importances
plt.figure(figsize=(8, 6))
plt.bar(range(X.shape[1]), importances[indices], color='skyblue')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
plt.title("Feature Importance - Random Forest")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()