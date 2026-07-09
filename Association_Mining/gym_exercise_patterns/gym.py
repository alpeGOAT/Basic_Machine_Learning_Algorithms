import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv('weightlifting_721_workouts.csv')
df = df.drop(["Notes", "Workout Notes"],axis=1)

print("First 5 Exercises")
print(df.head())
print("\n")

print("Dataset Information")
print(df.info())
print("\n")

df = df.dropna(subset=["Date", "Workout Name", "Exercise Name"]) # if exercise name empty drop row

print("Cleaned Dataset Information")
print(df.head())
print("\n")

df["workout_id"] = df["Date"].astype(str) + " - " + df["Workout Name"].astype(str)
basket = df.groupby("workout_id")["Exercise Name"].apply(lambda x: list(set(x))).reset_index()
# apply function used here to remove duplication

print("Basket Data Information")
print(basket.head())
print("\n")

transactions = basket["Exercise Name"].tolist()

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df_encoded = pd.DataFrame(te_array,columns=te.columns_)
frequent_items = apriori(df_encoded, min_support=0.02, use_colnames=True, max_len=2)
# max_len = 2 means only find 2 combinations

print("Dataset Information with True / False")
print(df_encoded.iloc[:5,:5]) # prints first 5 rows and 5 columns
print("\n")

print("Frequent Items Information")
print(frequent_items.head())
print("\n")

print("Total Frequent Items: ")
print(frequent_items.shape[0])
print("\n")

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.2)
rules = rules.sort_values(by="lift", ascending=False)

rules["antecedents_text"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
rules["consequents_text"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

rules["rule"] = rules["antecedents_text"] + "-->" + rules["consequents_text"]
rules_clean = rules[["rule", "support", "confidence", "lift"]]

print("Top 40 Exercise Combinations:")
print(rules_clean.head(40).to_string(index=False))
print("\n")

# ------------------------------------------------------------------------------------------------ #

def get_exercise_recommendation(exercise_name, rules_df, top_n=5):
    matched_rules = rules_df[rules_df["antecedents"].apply(lambda x: exercise_name in x)]

    # Remove recommendations that are basically the same exercise
    matched_rules = matched_rules[~matched_rules["consequents_text"].str.lower().str.contains(exercise_name.lower(), regex=False)]

    if matched_rules.empty:
        print("No Matched Exercise for ",exercise_name)
        return

    else:
        matched_rules = matched_rules.sort_values(by=["lift","confidence"], ascending=False)

    print("\n")

    print(f"Recommended Exercises Based On {exercise_name}")
    print("--" * 20)
    print(matched_rules[["antecedents_text", "consequents_text","support","lift"]].head(top_n).to_string(index=False))
    print("--" * 20)
    print("\n")

exc_name = input("Enter Exercise Name: ")
get_exercise_recommendation(exc_name, rules, 5)















