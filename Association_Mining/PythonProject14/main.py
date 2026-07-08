import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules # importing apriori algorithm
from mlxtend.preprocessing import  TransactionEncoder # apriori needs the data in true / false format
import matplotlib.pyplot as plt

df = pd.read_csv('Groceries_dataset.csv')
print(df.head())
print("\n")

basket = df.groupby(['Member_number','Date'])['itemDescription'].apply(list).reset_index()
transactions = basket['itemDescription'].tolist()
print(transactions)
# Group items purchased together by same customer on same date

print("\n")

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array,columns=te.columns_)
# converts transaction list into True / False format

frequent_item_sets = apriori(df_encoded, min_support=0.01, use_colnames=True) # itemsets must appear in %1 of transactions
print("Total Frequent Itemsets: ",frequent_item_sets.shape[0])

rules = association_rules(frequent_item_sets, metric='confidence', min_threshold=0.1)
# Confidence means chance of Y is purchased when X is purchased

rules = rules[
    (rules['antecedents'].apply(lambda x: len(x) >= 1)) & # Go through every antecedent column
    (rules['consequents'].apply(lambda x: len(x) >= 1)) # Go through every consequent column
]
# Antecedents are the left side and Consequent is right side
# Lets say that if customer buys bread, he might also buy milk, {bread} --> {milk}
# Here, Bread is antecedent and Milk is consequent


print("Association Rules: ",rules.shape[0])
print(rules[['antecedents','consequents','support','confidence','lift']].head(5))
print("\n")

top_items = df['itemDescription'].value_counts().head(10)
#value_counts() returns a Series containing the counts of unique values in descending order
top_items.plot(kind='bar', title='Top 10 Items')
plt.xlabel('Item')
plt.ylabel('Count') # plt library allows us to visualize a graph
plt.show()







