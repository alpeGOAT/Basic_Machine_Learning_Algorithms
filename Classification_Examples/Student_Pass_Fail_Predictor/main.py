import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # it uses if and else questions to understand the result
from sklearn.metrics import accuracy_score

data = {
    "name": ["Ali", "Ayşe", "Mehmet", "Zeynep", "Can", "Elif", "Burak", "Deniz", "Ece", "Mert"],
    "study_hours": [5, 2, 8, 1, 4, 6, 3, 7, 2, 9],
    "attendance": [80, 50, 90, 40, 70, 85, 60, 95, 45, 98],
    "midterm": [60, 40, 75, 30, 55, 65, 45, 80, 35, 90],
    "final": [70, 45, 85, 35, 60, 75, 50, 88, 40, 95],
    "result": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
print(df.head()) # prints first 5 students grades
print("\n")

print(df.iloc[-1]) # prints the last students grades
print("\n")

x = df[["study_hours","attendance","midterm","final"]]
y = df["result"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Predictions: ", y_pred)
print("Real Values: ",list(y_test))
print("Accuracy: ", accuracy)