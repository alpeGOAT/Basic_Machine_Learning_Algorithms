import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 350)
pd.set_option('display.max_colwidth', 100)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation','relationship', 'race',
                'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income']

census_data = pd.read_csv(url, names=column_names)

print("\nFirst 5 rows of the Dataset")
print(census_data.head())

print("\nLast 5 rows of the Dataset")
print(census_data.tail())

print("\nDataset Shape")
print(census_data.shape)

print("\nDataset Information")
print(census_data.info())

print("\nDataset Description")
print(census_data.describe())

print("\nMissing values in the Dataset")
print(census_data.isnull().sum())
print("\n")

le = LabelEncoder()

categorical_features = ['workclass', 'education', 'marital-status','occupation', 'relationship',
                        'race', 'sex','native-country', 'income']

for feature in categorical_features:
    census_data[feature] = le.fit_transform(census_data[feature])

census_data[['age', 'fnlwgt', 'education-num',
             'capital-gain','capital-loss', 'hours-per-week']] = census_data[['age', 'fnlwgt', 'education-num',
             'capital-gain','capital-loss', 'hours-per-week']].apply(lambda x: (x-x.min()) / (x.max() - x.min()))

print(census_data.head())
print("\n")

X = census_data.drop('income', axis=1)
y = census_data['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

print("\nClassification Report")
print(classification_report(y_test, y_pred))

