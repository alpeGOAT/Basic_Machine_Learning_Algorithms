import pandas as pd
import numpy as np

data = {"salary": [3000, 3200, 3100, 3050, 3300, 10000, 3150, 2900]}

df = pd.DataFrame(data)

mean = df["salary"].mean() # mean of all values given
std_dev = df["salary"].std() # standard deviation of all values given

lower_limit = mean - 2 * std_dev
upper_limit = mean + 2 * std_dev

# if the given values are not between these limits, they are outlier
outlier = df[(df["salary"] < lower_limit) | (df["salary"] > upper_limit)]

print("Mean:", mean)
print("Standard Deviation:", std_dev)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)
print("\n")

print("Outliers: ")
print(outlier) # only 10000 is outlier in the dataset

