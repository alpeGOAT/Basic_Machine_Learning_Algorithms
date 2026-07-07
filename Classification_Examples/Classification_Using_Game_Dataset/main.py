import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer # preparation method
from sklearn.pipeline import Pipeline # connects multiple steps together
from sklearn.impute import SimpleImputer # fixes missing data
from sklearn.ensemble import RandomForestClassifier # for classification
from sklearn.preprocessing import OneHotEncoder # converts text categories into numbers so machine understand

df = pd.read_csv('games_dataset.txt')

print(df)
print("\n")

print("Missing Values:")
print(df.isnull().sum())
print("\n")

x = df.drop("success", axis = 1) # axis = 1 for columns, 0 for rows
y = df["success"]

numeric_features = ["price", "user_rating", "critic_score", "avg_playtime", "review_count"]
categorical_features = ["genre","platform"]

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
# for numeric data, if data is missing, it gives mean of the other datas

categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                          ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
# if there are missing data, give the most frequently use categorial feature
# if the model sees a new category, don't crash

preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                               ("cat", categorical_transformer, categorical_features)
])
# combines numerical cleaning and categorial cleaning

model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("classifier", RandomForestClassifier(random_state=0))
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nPredictions: ", y_pred)
print("\nAccuracy: ", accuracy)
print("\nReal Values: ", list(y_test))
print("\nConfusion Matrix: ",confusion_matrix(y_test, y_pred))
print("\nClassification Report: ",classification_report(y_test, y_pred))



