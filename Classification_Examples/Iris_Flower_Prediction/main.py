from sklearn import datasets
from sklearn.model_selection import train_test_split # this is used to train the model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()

X = iris.data # Sepal length, sepal width, petal length, petal width
Y = iris.target # 0 for Setosa, 1 for Versicolor, 2 for Virginica

print("Feature Names:\n", iris.feature_names)
print("Target Names:\n", iris.target_names)

print("X Shape:\n", X.shape)
print("Y Shape:\n", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
# Used the ½30 of data to testing

print("X_train Shape:\n", X_train.shape)
print("X_test Shape:\n", X_test.shape)
print("Y_train Shape:\n", Y_train.shape)
print("Y_test Shape:\n",Y_test.shape)

model = KNeighborsClassifier(3)
model.fit(X_train, Y_train) # training the models
y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)

print("Predictions:\n", y_pred)
print("Actual Values:\n", Y_test)
print("Accuracy:\n", accuracy)







