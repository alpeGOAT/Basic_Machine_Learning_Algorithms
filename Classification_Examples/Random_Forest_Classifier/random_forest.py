import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 350)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 100)

titanic = pd.read_csv('titanic.csv')

print("\nFirst 5 Rows of the Titanic Dataset")
print(titanic.head())

print("\nLast 5 Rows of the Titanic Dataset")
print(titanic.tail())

print("\nTitanic Dataset Shape")
print(titanic.shape)

print("\nTitanic Dataset Information")
print(titanic.info())

print("\nTitanic Dataset Description")
print(titanic.describe())

print("\nTitanic Dataset Missing Values")
print(titanic.isnull().sum())

titanic = titanic.dropna(subset=['Survived'])

X = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic['Survived']

X['Sex'] = X['Sex'].map({'female':0, 'male':1})
X['Age'] = X['Age'].fillna(X['Age'].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\nAccuracy Score")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report")
print(classification_rep)

sample = X_test.iloc[0:1]
prediction = rf_classifier.predict(sample)

sample_dict = sample.iloc[0].to_dict()
print(f"\nSample Passenger : {sample_dict}")
print(f"Predicted Survival: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")











