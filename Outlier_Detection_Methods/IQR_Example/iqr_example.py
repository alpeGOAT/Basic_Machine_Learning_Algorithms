import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = [6, 2, 3, 4, 5, 1, 50]
sort_data = np.sort(data)
print(sort_data)
print("\n")
# Here, we sort data and print it make sure it's ready to use IQR Method

Q1 = np.percentile(data,25, method="midpoint")
Q2 = np.percentile(data,50, method="midpoint")
Q3 = np.percentile(data,75, method="midpoint")

print("Q1 25 Percentile of the give data is: ",Q1)
print("Q2 50 Percentile of the give data is: ",Q2)
print("Q3 75 Percentile of the give data is: ",Q3)

IQR = Q3 - Q1 # Interquartile Range
print("Interquartile range is: ",IQR)

# if values are not between limit range, it is an outlier
low_limit = Q1 - 1.5*IQR
high_limit = Q3 + 1.5*IQR
print("Low and high limit is: ",low_limit, ",", high_limit)
print("\n")

outlier = []
for x in data:
    if ((x < low_limit) or (x > high_limit)):
        outlier.append(x)

print("Outliers are: ",outlier)

sns.boxplot(data) # create a graph using seaborn library
plt.show()


