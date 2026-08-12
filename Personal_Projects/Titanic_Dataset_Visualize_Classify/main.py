import pandas as pd
from matplotlib import pyplot as plt
from narwhals.typing import NonNestedDType
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 350)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 100)

titanic = pd.read_csv('Titanic-Dataset.csv')

print("\nFirst 5 Rows:")
print(titanic.head())

print("\nLast 5 Rows:")
print(titanic.tail())

print("\nShape")
print(titanic.shape)

print("\nAll Columns:")
print(titanic.columns)

print("\nInformation")
titanic.info()

print("\nDescription")
print(titanic.describe())

print("\nMissing Values")
print(titanic.isnull().sum())

# ********************************************************************

titanic['Age'] = titanic['Age'].round().astype('Int64')
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean().round())

titanic = titanic.dropna(subset=['Survived'])
titanic['Sex'] = titanic['Sex'].map({'female':0, 'male':1})
titanic['Cabin'] = titanic['Cabin'].fillna('Unknown')

print("\nCleaned Dataset")
print(titanic.head(25))

titanic.to_csv('titanic_cleaned.csv', index=False)

# ********************************************************************

X = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification report:", classification_report)

# ********************************************************************

sample = X_test.iloc[0:1]
prediction = rf_classifier.predict(sample)

sample_dict = sample.iloc[0].to_dict()
print(f"\nSample Passenger : {sample_dict}")
print(f"Predicted Survival: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")

sample2 = X_test.iloc[67:]
prediction2 = rf_classifier.predict(sample2)

sample_dict2 = sample2.iloc[67].to_dict()
print(f"\nSample Passenger 2 : {sample_dict2}")
print(f"Predicted Survival: {'Survived' if prediction2[0] == 1 else 'Did Not Survive'}")

# ********************************************************************

embarked_location_places = titanic['Embarked'].unique()
print("\n", embarked_location_places, "\n")

titanic_for_visualization = titanic.copy()
titanic_for_visualization['Family_Size']= pd.NA
titanic_for_visualization['Family_Size'] = titanic_for_visualization['SibSp'] + titanic_for_visualization['Parch']

plt.figure(figsize = (10,6))
plt.title('Survivor Count based on Family Size')
sns.countplot(data=titanic_for_visualization, x='Family_Size', hue='Survived')
plt.savefig('Survival Count based on Family_Size.png')
plt.show()

plt.figure(figsize = (10,6))
plt.title('Survivor Count Based on the Gender (0-> Female, 1-> Male)')
sns.countplot(data=titanic_for_visualization, x='Sex', hue='Survived')
plt.savefig('Survival_Count_Based_on_the_Gender.png')
plt.show()

plt.figure(figsize = (10,6))
plt.title('Survivor Count based on the Embarked Location')
sns.countplot(data=titanic_for_visualization, x='Embarked', hue='Survived')
plt.savefig('Survival_Count_based_on_the_Embarked_Location')
plt.show()

plt.figure(figsize = (10,6))
plt.title("Survivor Count based on the Passenger Class")
sns.countplot(data=titanic_for_visualization, x='Pclass', hue='Survived')
plt.savefig('Survival_Count_based_on_the_Passenger_Class.png')
plt.show()

# ********************************************************************

passengers_older_than_50 = titanic_for_visualization[titanic_for_visualization['Age'] > 50]

plt.title("Passengers Age Distribution")
plt.scatter(titanic_for_visualization['PassengerId'], titanic_for_visualization['Age'], c='royalblue', cmap='viridis', alpha=0.6, label='Passengers')
plt.scatter(passengers_older_than_50['PassengerId'], passengers_older_than_50['Age'], c='red', cmap='viridis', alpha=0.6, label='Passengers older than 50')
plt.xlabel("PassengerId")
plt.ylabel("Age")
plt.grid(True)
plt.legend()
plt.savefig('Passengers_Age_Distribution.png')
plt.show()

# ********************************************************************

each_embarked_percentage = (
    titanic_for_visualization
    .groupby('Embarked')['Survived']
    .mean() * 100
)

print(each_embarked_percentage, "\n")

each_passenger_class_percentage = (
    titanic_for_visualization.groupby('Pclass')['Survived'].mean() * 100
)

print(each_passenger_class_percentage, "\n")

each_genders_survival_percentage = (
    titanic_for_visualization.groupby('Sex')['Survived'].mean() * 100
)

print(each_genders_survival_percentage, "\n")

# ********************************************************************






