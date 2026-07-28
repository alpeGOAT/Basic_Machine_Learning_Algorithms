import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 300)

df = pd.read_csv('bank-full.csv', sep=";")

print("\nFirst 5 Rows of the Bank Dataset")
print(df.head())

print("\nLast 5 Rows of the Bank Dataset")
print(df.tail())

print("\nBank Dataset Description")
print(df.describe())

print("\nBank Dataset Information")
print(df.info())

print("\nBank Dataset Shape Information")
print(df.shape)

print("\nMissing Values in the Bank Dataset")
print(df.isnull().sum())
print("There are no missing values in the Bank Dataset\n")

df_for_regression = df.drop('y',axis=1)
df_for_regression['default'] = df_for_regression['default'].map({"yes":1,"no":0})
df_for_regression['housing'] = df_for_regression['housing'].map({"yes":1,"no":0})
df_for_regression['loan'] = df_for_regression['loan'].map({"yes":1,"no":0})

print("Bank Dataset after converting values into binary\n")
print(df_for_regression.head())

categorial_columns = ["job", "marital", "education", "contact", "month", "poutcome"]
numerical_columns =  ["age", "default", "balance", "housing", "loan", "day", "duration",
                      "campaign", "pdays", "previous"]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_data = encoder.fit_transform(df_for_regression[categorial_columns])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorial_columns))

y = df["y"].map({"yes": 1, "no": 0})

numerical_df = df_for_regression[numerical_columns].reset_index(drop=True)
encoded_df = encoded_df.reset_index(drop=True)

X = pd.concat([numerical_df, encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23, stratify=y)

X_train = X_train.copy()
X_test = X_test.copy()

scaler = StandardScaler()

X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

model = LogisticRegression(max_iter=20000, class_weight="balanced", random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred) * 100
print(f"\nLogistic Regression model accuracy: {acc:.2f}%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))








