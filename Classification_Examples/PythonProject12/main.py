import mysql.connector # to connect MySQL datasets to PyCharm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd= "alperen31",
    port= 3306,
    database= "printerworld"
)

my_cursor = mydb.cursor()

query = "SELECT * FROM filament"
my_cursor.execute(query)

rows = my_cursor.fetchall()

for row in rows: # printing filament table from MySQL
    print(row)

data = pd.read_sql(query, mydb)
print(data.head())
print("\n")
print(data.info())
print("\n")

encoder = LabelEncoder()
data["filament_type"] = encoder.fit_transform(data["filament_type"]) #convert filament type text to number

print(data.head())
print("\n")

X = data[["filament_type","nozzle_temp","bed_temp","cooling_fan","layer_height"]]
Y = data["success"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 42)
model = DecisionTreeClassifier(random_state = 42) # used decision tree because it's a classification project, also uses same training row everytime

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Prediction: ", y_pred)
print("Accuracy: ", accuracy)
print("Real Values",list(y_test))





