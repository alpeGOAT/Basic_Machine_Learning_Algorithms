import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 8)

df = pd.read_excel('Online Retail.xlsx')
print("First 5 rows of dataset")
print(df.head())
print("\n")

print("Dataset Information")
print(df.info())
print("\n")

print("Dataset Description")
print(df.describe())
print("\n")

print("Missing Values in the Dataset")
print(df.isnull().sum())
print("\n")

# Data Preparation
df = df.dropna(subset=['Description'])
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df = df[~df['InvoiceNo'].astype(str).str.startswith('c')]
df['Description'] = df['Description'].str.strip()

transactions = df.groupby('InvoiceNo')['Description'].apply(list).tolist()
transaction_Encoder = TransactionEncoder()
transaction_Encoder.fit(transactions)
encoded_Transactions = transaction_Encoder.transform(transactions)

df_pd = pd.DataFrame(encoded_Transactions, columns=transaction_Encoder.columns_)

frequent_item_sets = fpgrowth(df_pd, min_support=0.01, use_colnames=True)
frequent_item_sets = frequent_item_sets.sort_values(by=['support'], ascending=False)

print("Frequent Items Sets")
print(frequent_item_sets.head())
print("\n")

rules = association_rules(frequent_item_sets, metric='lift', min_threshold=1.0)
print(rules[['antecedents', 'consequents','support','lift','confidence']].head(15))
print("\n")
