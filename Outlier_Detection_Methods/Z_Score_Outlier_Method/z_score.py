from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import zscore # scipy library helps finding z_score of each value

data = [5, 2, 4.5, 4, 3, 2, 6, 20, 9, 2.5, 3.5, 4.75, 6.5, 2.5, 8, 1]
df = pd.DataFrame(data, columns=['Value'])
print(df.head())

print("-----------------------------")

df['Z-score'] = zscore(df['Value']) # z_score = (current_value - mean) / standard deviation
print(df)
print("\n")


print("Outlier Values:")
outliers = df[df['Z-score'].abs() > 3] # if |z_score| bigger than 3, that value is an outlier
print(outliers)

plt.figure(figsize=(10,6))

plt.scatter(df['Value'].index, df['Z-score'], label='Data Points') # creates data points
plt.scatter(outliers['Value'].index, outliers['Z-score'], color='red', label='Outliers')
# creates outliers but with red color

plt.xlabel('Index Value')
plt.ylabel('Z-score')
plt.legend()
plt.grid(True)
plt.show()



























