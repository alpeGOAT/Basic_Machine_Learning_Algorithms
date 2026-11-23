import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 120)
pd.set_option('display.width', 350)
pd.set_option('display.max_colwidth', 120)

airbnb = pd.read_csv('Aemf1.csv')

print("\nFirst 5 rows of the Airbnb Dataset")
print(airbnb.head())

print("\nLast 5 rows of the Airbnb Dataset")
print(airbnb.tail())

print("\nAirbnb Dataset Shape")
print(airbnb.shape)

print("\nAirbnb Dataset Information")
airbnb.info()

print("\nAirbnb Dataset Description")
print(airbnb.describe())

print("\nMissing values in the Airbnb Dataset")
print(airbnb.isnull().sum())

if airbnb.isnull().sum().sum() > 0:
    print("\nThere are missing values in the Airbnb Dataset")
elif airbnb.isnull().sum().sum() == 0:
    print("\nThere are no missing values in the Airbnb Dataset")

print("\nSum of Duplicated Rows in the Airbnb Dataset: ", airbnb.duplicated().sum())

if airbnb.duplicated().sum() > 0:
    print("\nThere are duplicated rows in the Airbnb Dataset")
elif airbnb.duplicated().sum() == 0:
    print("\nThere are no duplicated rows in the Airbnb Dataset")

each_city_distribution = airbnb['City'].value_counts().sort_index()
print("\nCity Distribution of Airbnb Dataset")
print(each_city_distribution)

plt.figure(figsize=(8.4, 6))
plt.bar(each_city_distribution.index, each_city_distribution.values, color='royalblue', alpha=0.6)
plt.xlabel('City', fontsize=12, labelpad=10)
plt.ylabel('Value Count',labelpad=11,fontsize=12)
plt.grid(True, alpha=0.5, axis='y')
plt.title('City Distribution of Airbnb Dataset', fontsize=14)
plt.savefig('city_distribution_airbnb.jpg')
plt.show()

average_prices = airbnb['Price'].mean().round(2)
print("\nAverage Price of Airbnb Dataset (All Cities): ", average_prices)

average_prices_each_city = airbnb.groupby('City')['Price'].mean().round(2)
print("\nAverage Price for each City")
print(average_prices_each_city)

plt.figure(figsize=(8.8, 7.5))
plt.bar(average_prices_each_city.index, average_prices_each_city.values, color='royalblue', alpha=0.6)
plt.xlabel('City', fontsize=12)
plt.xticks(rotation=45,fontsize=10)
plt.ylabel('Average Price',labelpad=11,fontsize=12)
plt.grid(True, alpha=0.5, axis='y')
plt.title("Average Price for each City", fontsize=14)
plt.savefig('avg_price_each_city.jpg')
plt.tight_layout()
plt.show()

day_distribution = airbnb['Day'].value_counts().sort_index()
print("\nDay Distribution of Airbnb Dataset")
print(day_distribution)

plt.bar(day_distribution.index, day_distribution.values, color='royalblue', alpha=0.6)
plt.xlabel('Day Type', fontsize=11)
plt.ylabel('Value Count', fontsize=11)
plt.ylim(20800, 20900)
plt.yticks(range(20800, 20901, 10))
plt.grid(True, alpha=0.5, axis='y')
plt.title('Day Distribution of Airbnb Dataset', fontsize=14)
plt.savefig('day_type_distribution.jpg')
plt.show()

room_type_distribution = airbnb['Room Type'].value_counts().sort_index()
print("\nRoom Distribution of Airbnb Dataset")
print(room_type_distribution)

plt.bar(room_type_distribution.index, room_type_distribution.values, color='royalblue', alpha=0.6)
plt.xlabel('Room Type')
plt.ylabel('Value Count')
plt.yticks([0, 1000, 5000, 10000, 15000, 20000, 25000, 30000])
plt.grid(True, alpha=0.5, axis='y')
plt.title('Room Distribution of Airbnb Dataset', fontsize=14)
plt.savefig('room_type_distribution.jpg')
plt.show()

average_person_capacity = airbnb['Person Capacity'].mean().round(2)
print("\nAverage Person Capacity of Airbnb Dataset: ", average_person_capacity)

avg_capacity_each_city = airbnb.groupby('City')['Person Capacity'].mean().round(2)
print("\nAverage Person Capacity for each City")
print(avg_capacity_each_city)

plt.figure(figsize=(8.8, 7.5))
plt.bar(avg_capacity_each_city.index, avg_capacity_each_city.values, color='royalblue', alpha=0.6)
plt.xlabel('City', labelpad=15, fontsize=12)
plt.ylabel('Average Person Capacity', labelpad=15, fontsize=12)
plt.grid(True, alpha=0.5, axis='y')
plt.title('Average Person Capacity of each City')
plt.savefig('avg_person_capacity_each_city.jpg')
plt.show()

airbnb['Superhost'] = airbnb['Superhost'].astype('int')

host_type_distribution = airbnb['Superhost'].value_counts().sort_index()
print("\nHost Type Distribution of Airbnb Dataset")
print(host_type_distribution)

plt.bar(host_type_distribution.index, host_type_distribution.values, color='royalblue', alpha=0.6)
plt.xlabel('Superhost Type',labelpad=-3)
plt.xticks([0,1])
plt.ylabel('Value Count')
plt.yticks(fontsize=9)
plt.grid(True, alpha=0.5, axis='y')
plt.title('Host Type Distribution of Airbnb Dataset', fontsize=12)
plt.savefig('host_type_distribution.jpg')
plt.show()

room_type_distribution_business_or_not = airbnb['Business'].value_counts().sort_index()
print("\nRoom Type Distribution of Airbnb Dataset (Business or not)")
print(room_type_distribution_business_or_not)

plt.bar(room_type_distribution_business_or_not.index, room_type_distribution_business_or_not.values, color='royalblue', alpha=0.6)
plt.xlabel('Business or not')
plt.xticks([0,1])
plt.ylabel('Value Count')
plt.grid(True, alpha=0.5, axis='y')
plt.title('Room Type Distribution of Airbnb Dataset (Business or not)')
plt.savefig('room_type_distribution_business_or_not.jpg')
plt.show()

multiple_rooms_or_not = airbnb['Multiple Rooms'].value_counts().sort_index()
print("\nDoes Airbnb has multiple rooms?")
print(multiple_rooms_or_not)

plt.bar(multiple_rooms_or_not.index, multiple_rooms_or_not.values, color='royalblue', alpha=0.6)
plt.xlabel('Room Amount (Multiple or not)')
plt.xticks([0,1], ['No','Yes'])
plt.ylabel('Value Count')
plt.title('Does Airbnb has multiple rooms?')
plt.grid(True, alpha=0.5, axis='y')
plt.savefig('multiple_rooms_or_not.jpg')
plt.show()

average_price_by_room_type = airbnb.groupby('Room Type')['Price'].mean().round(2)
print("\nAverage Price based on Room Type")
print(average_price_by_room_type)

plt.bar(average_price_by_room_type.index, average_price_by_room_type.values, color='royalblue', alpha=0.6)
plt.xlabel('Room Type')
plt.ylabel('Average Price',labelpad=7)
plt.title('Average Price based on Room Type')
plt.grid(True, alpha=0.5, axis='y')
plt.savefig('average_price_by_room_type.jpg')
plt.show()

average_price_by_capacity = airbnb.groupby('Person Capacity')['Price'].mean().round(2)
print("\nAverage Price based on Person Capacity")
print(average_price_by_capacity)

plt.bar(average_price_by_capacity.index, average_price_by_capacity.values, color='royalblue', alpha=0.6)
plt.xlabel('Person Capacity')
plt.ylabel('Average Price')
plt.title('Average Price based on Person Capacity')
plt.grid(True, alpha=0.5, axis='y')
plt.savefig('average_price_by_capacity.jpg')
plt.show()

average_cleanliness_rating = airbnb['Cleanliness Rating'].mean().round(2)
print("\nAverage Cleanliness Rating: ", average_cleanliness_rating)

average_cleanliness_rating_by_city = airbnb.groupby('City')['Cleanliness Rating'].mean().round(2)
print("\nAverage Cleanliness Rate each city")
print(average_cleanliness_rating_by_city)

plt.figure(figsize=(8.5,6))
plt.bar(average_cleanliness_rating_by_city.index, average_cleanliness_rating_by_city.values, color='royalblue', alpha=0.6)
plt.xlabel('City')
plt.ylabel('Average Cleanliness Rating')
plt.grid(True, alpha=0.5, axis='y')
plt.title('Average Cleanliness Rating by City')
plt.savefig('average_cleanliness_rating.jpg')
plt.show()

average_guest_satisfaction = airbnb['Guest Satisfaction'].mean().round(2)
print("\nAverage Guest Satisfaction: ", average_guest_satisfaction)

average_guest_satisfaction_by_city = airbnb.groupby('City')['Guest Satisfaction'].mean().round(2)
print("\nAverage Guest Satisfaction each city")
print(average_guest_satisfaction_by_city)

plt.figure(figsize=(8.5,6))
plt.bar(average_guest_satisfaction_by_city.index, average_guest_satisfaction_by_city.values, color='royalblue', alpha=0.6)
plt.xlabel('City')
plt.ylabel('Average Guest Satisfaction',labelpad=15)
plt.grid(True, alpha=0.5, axis='y')
plt.title('Average Guest Satisfaction by City')
plt.savefig('average_guest_satisfaction_by_city.jpg')
plt.show()

most_expensive_price = airbnb['Price'].max().round()
print("\nMost Expensive Airbnb Price: ", most_expensive_price)

most_expensive_by_city = airbnb.groupby('City')['Price'].max().round()
print("\nMost Expensive Airbnb each city")
print(most_expensive_by_city)

plt.figure(figsize=(8.5,6))
plt.bar(most_expensive_by_city.index, most_expensive_by_city.values, color='royalblue', alpha=0.6)
plt.xlabel('City', labelpad=9)
plt.ylabel('Most Expensive Airbnb Price',labelpad=9)
plt.grid(True, alpha=0.5, axis='y')
plt.title('Most Expensive Airbnb by City')
plt.savefig('most_expensive_by_city.jpg')
plt.show()

average_guest_satisfaction_by_host_type = airbnb.groupby('Superhost')['Guest Satisfaction'].mean().round(2)
print("\nAverage  Guest Satisfaction depending on Host Type")
print(average_guest_satisfaction_by_host_type)

plt.bar(average_guest_satisfaction_by_host_type.index, average_guest_satisfaction_by_host_type.values, color='royalblue', alpha=0.6)
plt.xlabel('Host Type')
plt.xticks([0,1], ['Normal Host','Super Host'])
plt.ylabel('Average  Guest  Satisfaction  Rate')
plt.grid(True, alpha=0.5, axis='y')
plt.title('Average Guest Satisfaction Rate by Host Type')
plt.savefig('average_guest_satisfaction_by_host_type.jpg')
plt.show()

average_price_by_host_type = airbnb.groupby('Superhost')['Price'].mean().round(2)
print("\nAverage Price based on Host Type")
print(average_price_by_host_type)

plt.bar(average_price_by_host_type.index, average_price_by_host_type.values, color='royalblue', alpha=0.6)
plt.xlabel('Host Type')
plt.xticks([0,1], ['Normal Host','Super Host'])
plt.ylabel('Average Price',labelpad=9)
plt.grid(True, alpha=0.5, axis='y')
plt.title('Average Price based on Host Type')
plt.savefig('average_price_by_host_type.jpg')
plt.show()

average_cleanliness_rate_by_host_type = airbnb.groupby('Superhost')['Cleanliness Rating'].mean().round(2)
print("\nAverage Cleanliness  Rate based on Host Type")
print(average_cleanliness_rate_by_host_type)

host_percentages = airbnb['Superhost'].value_counts(normalize=True).mul(100).round(2).to_frame(name='Host Percentage')
print("\nSuperhost Distribution Percentages")
print(host_percentages)

plt.scatter(airbnb['City Center (km)'], airbnb['Price'], color='red', alpha=0.3)
plt.xlabel('City Center (km)')
plt.ylabel('Price')
plt.title('Price Depending on how far bnb from the City center (km)')
plt.grid(True, alpha=0.5)
plt.savefig('price_distribution_depending_on_city_center.jpg')
plt.show()

super_host_percentage_city = airbnb.groupby('City')['Superhost'].mean().mul(100).round(2)
print("\nSuperhost Percentage by City")
print(super_host_percentage_city)

plt.figure(figsize=(8.5,6))
plt.bar(super_host_percentage_city.index, super_host_percentage_city.values, color='royalblue', alpha=0.6)
plt.xlabel('City', labelpad=15, fontsize=12)
plt.ylabel('Superhost  Percentage',labelpad=15,fontsize=12)
plt.grid(True, alpha=0.5, axis='y')
plt.title('Super host Percentage by City')
plt.savefig('super_host_percentage_by_city.jpg')
plt.show()

plt.scatter(airbnb['Restraunt Index'], airbnb['Price'], color='red', alpha=0.3)
plt.xlabel('Restaurant Index')
plt.ylabel('Price')
plt.grid(True, alpha=0.5)
plt.title('Price Depending on Restaurant Index')
plt.savefig('price_distribution_depending_on_restaurant_index.jpg')
plt.show()
plt.show()

plt.scatter(airbnb['Attraction Index'], airbnb['Price'], color='red', alpha=0.3)
plt.xlabel('Attraction Index')
plt.ylabel('Price')
plt.grid(True, alpha=0.5)
plt.title('Price Depending on Attraction Index')
plt.savefig('price_distribution_depending_on_attraction_index.jpg')
plt.show()

numerical_columns = airbnb.select_dtypes(include='number')
correlations = numerical_columns.corr()

print("\nCorrelation Matrix")
print(correlations.round(2))

plt.figure(figsize=(14, 10))
sns.heatmap(correlations, annot=True ,fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Airbnb Dataset')
plt.tight_layout()
plt.savefig('correlation_matrix.jpg')
plt.show()

plt.boxplot(airbnb['Price'])
plt.ylabel('Price')
plt.title('Price Boxplot of Airbnb Dataset')
plt.grid(True, alpha=0.5, axis='y')
plt.savefig('price_boxplot.jpg')
plt.show()

features = ['City','Price','Day','Room Type','Person Capacity', 'Multiple Rooms',
            'Business', 'Cleanliness Rating', 'Guest Satisfaction', 'Bedrooms',
            'City Center (km)','Metro Distance (km)',
            'Normalised Attraction Index', 'Normalised Restraunt Index']

airbnb['Superhost'] = airbnb['Superhost'].astype('int')

X = airbnb[features]
X = pd.get_dummies(X, columns=['City', 'Day', 'Room Type'], drop_first=True)
y = airbnb['Superhost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=35)
random_forest_classifier.fit(X_train, y_train)
y_pred = random_forest_classifier.predict(X_test)

print("\nAccuracy: ", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

features_without_ratings = ['City','Price','Day','Room Type','Person Capacity', 'Multiple Rooms',
            'Business', 'Bedrooms',
            'City Center (km)','Metro Distance (km)',
            'Normalised Attraction Index', 'Normalised Restraunt Index']

X1 = airbnb[features_without_ratings]
X1 = pd.get_dummies(X1, columns=['City', 'Day', 'Room Type'], drop_first=True)
y1 = airbnb['Superhost']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=35)

random_forest_classifier1 = RandomForestClassifier(n_estimators=100, random_state=35)
random_forest_classifier1.fit(X1_train, y1_train)
y1_pred = random_forest_classifier1.predict(X1_test)

print("\nAccuracy (without ratings): ", accuracy_score(y1_test, y1_pred))

print("\nClassification Report (without ratings)")
print(classification_report(y1_test, y1_pred))

print("\nConfusion Matrix (without ratings)")
print(confusion_matrix(y1_test, y1_pred))

features_for_regression = ['City','Price','Day','Room Type','Person Capacity', 'Multiple Rooms',
            'Business', 'Cleanliness Rating', 'Guest Satisfaction', 'Bedrooms',
            'City Center (km)','Metro Distance (km)',
            'Normalised Attraction Index', 'Normalised Restraunt Index']

X_reg = airbnb[features_for_regression]
X_reg = pd.get_dummies(X_reg, columns=['City', 'Day', 'Room Type'], drop_first=True)
y_reg = airbnb['Price']

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=35)

linear_regression = LinearRegression()
linear_regression.fit(X_reg_train, y_reg_train)
y_reg_pred = linear_regression.predict(X_reg_test)

print("\nAccuracy: ", accuracy_score(y_reg_test, y_reg_pred))

print("\nClassification Report")
print(classification_report(y_reg_test, y_reg_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_reg_test, y_reg_pred))









