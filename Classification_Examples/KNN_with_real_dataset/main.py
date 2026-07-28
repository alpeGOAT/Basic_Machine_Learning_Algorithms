import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import parse_version
import scipy.stats
import numpy as np

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 350)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 100)

df = pd.read_csv('data_banknote_authentication.txt')

print("\nFirst 5 Rows of the Dataset")
print(df.head())

print("\nLast 5 Rows of the Dataset")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nDataset Description")
print(df.describe())

print("\nMissing Values in the Dataset")
print(df.isnull().sum())
print("There are no missing values in the Dataset\n")

X = df[["variance", "skewness", "curtosis", "entropy"]]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

print("Accuracy Score")
print(accuracy_score(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))

k_range = range(1, 21)
cv_scores = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

plt.figure(figsize=(8,5))
plt.plot(k_range, cv_scores, marker='o')
plt.title("KNN Accuracy vs K ")
plt.xlabel("# of Neighbors (k)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("Accuracy_of_each_k_value.png")
plt.show()

best_k = k_range[np.argmax(cv_scores)]
print(f"Best k from cross-validation: {best_k}")
print("\n")

best_knn =KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred = best_knn.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])

display.plot(cmap="Blues")
plt.title(f"Confusion Matrix (k={best_k})")
plt.grid(False)
plt.savefig("Confusion_Matrix.png")
plt.show()

print("Final Model Accuracy Score")
print(accuracy_score(y_test, y_pred))
print("\nFinal Model Classification Report")
print(classification_report(y_test, y_pred))








