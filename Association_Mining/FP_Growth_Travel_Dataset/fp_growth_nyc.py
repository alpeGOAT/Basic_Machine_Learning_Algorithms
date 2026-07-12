import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

pd.set_option('display.max_columns', None) # maximum column limit
pd.set_option("display.max_colwidth", None)
pd.set_option('display.width', 1000)

df = pd.read_csv('dataset_TSMC2014_NYC.csv')
print("Dataset successfully loaded.")
print("Dataset shape:", df.shape)

print("\n---First 5 Rows---")
print(df.head())

print("\n---Dataset Information---")
print(df.info())

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# drop if dataset has duplicate values
df = df.drop_duplicates()

# drop if venueCategory has 'NaN'
df["venueCategory"] = df["venueCategory"].replace(["NaN", "nan", "NAN", ""], pd.NA)
df = df.dropna(subset=["venueCategory"])

# Not considering non-touristic locations
non_touristic_categories = [
    "Medical Center",
    "Subway",
    "Office",
    "Student Center",
    "Bus Station",
    "Home (private)",
    "NaN"
]

df = df[~df["venueCategory"].isin(non_touristic_categories)]

df = df[~df["venueCategory"].str.contains("Restaurant",na=False)]
# drop where venueCategory include Restaurant

print("\n---Dataset after removing non-touristic categories:---")
print(df.head(5))
print("\n)")

df = df[["userId", "venueCategory"]] # only necessary columns

# group userId by venueCategory
transactions = df.groupby("userId")["venueCategory"].apply(lambda x: list(dict.fromkeys(x))).tolist()
print(transactions[:5])
print("\n")

transaction_encoder = TransactionEncoder()
transaction_encoder.fit(transactions)
encoded_transactions = transaction_encoder.transform(transactions)
# show list in True / False format

df_pd = pd.DataFrame(encoded_transactions, columns=transaction_encoder.columns_)
print("Dataset after TransactionEncoder:")
print(df_pd.head())
print("\n")

frequent_itemSets = fpgrowth(df_pd, min_support=0.05, use_colnames=True, max_len=3)
# min_support means that items must appear in at least %5 of transactions
# maximum 3 combinations allowed

frequent_itemSets = frequent_itemSets.sort_values(by=['support'], ascending=False)
# sort item sets by their support descending

print("---Frequent Item Sets---")
print(frequent_itemSets.head())
print("\n")

rules = association_rules(frequent_itemSets, metric='lift', min_threshold=1.2)
# lift must be at least 1 or higher

rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
# remove frozenset

print("Item Combinations using FP-Growth")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15))
print("\n")

