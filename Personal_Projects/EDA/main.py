import pandas as pd
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 60)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 100)

cafe_dataset = pd.read_csv('dirty_cafe_sales.csv')

print("\nCafe Sales Dataset First 5 rows:")
print(cafe_dataset.head())

print("\nCafe Sales Dataset Last 5 rows:")
print(cafe_dataset.tail())

print("\nCafe Sales Dataset Shape")
print(cafe_dataset.shape)

print("\nCafe Sales Dataset Columns")
print(cafe_dataset.columns)

print("\nCafe Sales Dataset Information")
cafe_dataset.info()

print("\nCafe Sales Dataset Description")
print(cafe_dataset.describe())

print("\nMissing Values in Cafe Sales Dataset")
print(cafe_dataset.isnull().sum())

cafe_dataset.drop_duplicates()

cafe_dataset = cafe_dataset[cafe_dataset['Quantity'] != 'ERROR']
cafe_dataset = cafe_dataset[cafe_dataset['Quantity'] != 'UNKNOWN']
cafe_dataset['Quantity'] = cafe_dataset['Quantity'].round().astype('Int64')
cafe_dataset['Quantity'] = cafe_dataset['Quantity'].fillna(cafe_dataset['Quantity'].mean().round())

print("\nCafe Sales Dataset all quantity values")
print(cafe_dataset['Quantity'].value_counts())

cafe_dataset = cafe_dataset[cafe_dataset['Payment Method'] != 'ERROR']
cafe_dataset = cafe_dataset[cafe_dataset['Payment Method'] != 'UNKNOWN']
cafe_dataset.dropna(subset=['Payment Method'], inplace=True)

print("\nAll Payment methods and how much they are used")
print(cafe_dataset['Payment Method'].value_counts())

cafe_dataset = cafe_dataset[cafe_dataset['Item'] != 'ERROR']
cafe_dataset = cafe_dataset[cafe_dataset['Item'] != 'UNKNOWN']
cafe_dataset.dropna(subset=['Item'], inplace=True)

print("\nAll Item types and their counts")
print(cafe_dataset['Item'].value_counts())

cafe_dataset = cafe_dataset[cafe_dataset['Location'] != 'ERROR']
cafe_dataset = cafe_dataset[cafe_dataset['Location'] != 'UNKNOWN']
cafe_dataset.dropna(subset=['Location'], inplace=True)

print("\nWhere each user make payment")
print(cafe_dataset['Location'].value_counts())

cafe_dataset = cafe_dataset[cafe_dataset['Price Per Unit'] != 'ERROR']
cafe_dataset = cafe_dataset[cafe_dataset['Price Per Unit'] != 'UNKNOWN']
cafe_dataset.dropna(subset=['Price Per Unit'], inplace=True)

print("\nEach item and their cost")
unique_values = cafe_dataset[['Item', 'Price Per Unit']].drop_duplicates().sort_values(by=['Price Per Unit'], ascending=True)
print(unique_values)

cafe_dataset = cafe_dataset[cafe_dataset['Total Spent'] != 'ERROR']
cafe_dataset = cafe_dataset[cafe_dataset['Total Spent'] != 'UNKNOWN']
cafe_dataset.dropna(subset=['Total Spent'], inplace=True)

cafe_dataset = cafe_dataset[cafe_dataset['Transaction Date'] != 'ERROR']
cafe_dataset = cafe_dataset[cafe_dataset['Transaction Date'] != 'UNKNOWN']
cafe_dataset.dropna(subset=['Transaction Date'], inplace=True)

print("\nFirst 40 Rows after filtering the Dataset")
print(cafe_dataset.head(40))

cafe_dataset.to_csv('cleaned_cafe_sales',index=False)

plt.title('Value Count of each Quantity')
plt.xlabel('Quantity')
plt.ylabel('Count')
plt.bar(cafe_dataset['Quantity'].unique(), cafe_dataset['Quantity'].value_counts(), color='royalblue')
plt.savefig('Value_Count_of_each_Quantity.png')
plt.show()

plt.title('Value Count of each Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.bar(cafe_dataset['Payment Method'].unique(), cafe_dataset['Payment Method'].value_counts(), color='royalblue')
plt.savefig('Value_Count_of_each_Payment_Method.png')
plt.show()

plt.title('Value Count of each Item')
plt.xlabel('Item')
plt.ylabel('Count')
plt.xticks(fontsize=8)
plt.bar(cafe_dataset['Item'].unique(), cafe_dataset['Item'].value_counts(), color='royalblue')
plt.savefig('Value_Count_of_each_Item.png')
plt.show()

plt.title('Value Count of each Location')
plt.xlabel('Location')
plt.ylabel('Count')
plt.bar(cafe_dataset['Location'].unique(), cafe_dataset['Location'].value_counts(), color='royalblue',log=True)
plt.savefig('Value_Count_of_each_Location.png')
plt.show()

plt.title('Each Items and their Price Per Unit')
plt.xlabel('Item')
plt.xticks(fontsize=8)
plt.ylabel('Price Per Unit')
plt.bar(unique_values['Item'], unique_values['Price Per Unit'], color='royalblue', width=0.4)
plt.savefig('Each_Items_Cost.jpg')
plt.show()