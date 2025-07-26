import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.inspection import permutation_importance

# Load dataset
dataset = pd.read_csv("C:/Users/a/Desktop/AIML/AIML DATASET FINAL.csv", header=0)
print(dataset.head())

# Extract features and labels
X = dataset.iloc[:, 1:].values  # features
y = dataset.iloc[:, 0].values   # labels


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0, stratify=y)

# Train SVM classifier
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot 1: Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


result = permutation_importance(classifier, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy')

# Get importance values
importances = result.importances_mean
std = result.importances_std
indices = np.argsort(importances)[::-1]  # Sort descending
print("\nFeature Ranking:")


for idx in indices:
    print(f"Feature {idx}: Importance = {importances[idx]:.4f}")