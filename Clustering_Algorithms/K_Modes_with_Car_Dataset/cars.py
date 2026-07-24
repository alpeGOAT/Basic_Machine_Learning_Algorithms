from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import pandas as pd

column_names = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "class"
]

df = pd.read_csv('car.data', names=column_names)

print("First 5 rows of the car dataset:")
print(df.head())

print("\nLast 5 rows of the car dataset:")
print(df.tail())

print("\nDataset information:")
print(df.info())

print("\nDataset shape: ", df.shape)

print("\nDataset description:")
print(df.describe())

print("\nMissing values in the dataset:")
print(df.isnull().sum())

df_categorical = df[["buying", "maint", "doors", "persons", "lug_boot", "safety"]]
print("\nCategorical dataset\n")
print(df_categorical.head())
print("\n")

k = 4 # of clusters

modes = KModes(n_clusters=k, init='random', n_init=5, verbose=1, random_state=42)
cluster_labels = modes.fit_predict(df_categorical)

df["cluster"] = cluster_labels
print("\nDataset after adding cluster labels")
print(df.head(20))

print("\nHow many row each cluster has?")
print(df.value_counts(subset=['cluster']))

cluster_percentages = df["cluster"].value_counts(normalize=True).sort_index().mul(100).round(2)
cluster_counts = df["cluster"].value_counts().sort_index()

cluster_summary = pd.DataFrame({"row_count":cluster_counts, "cluster_percentage":cluster_percentages})
print("\nCluster summary:")
print(cluster_summary)

cluster_modes = pd.DataFrame(modes.cluster_centroids_, columns = df_categorical.columns)
cluster_modes.index.name ="cluster"

print("\nMode of each cluster:")
print(cluster_modes)

print("\nK-Modes cost:")
print(modes.cost_)

cluster_class_counts = pd.crosstab(df["cluster"], df["class"])
print("\nClass distribution inside each cluster:")
print(cluster_class_counts)
print("\n")

cluster_class_counts["class_sum"] = cluster_class_counts.sum(axis=1)
print(cluster_class_counts)

cluster_class_percentages = pd.crosstab(df["cluster"], df["class"], normalize="index").mul(100).round(2)
print("\nEach class types percentage:")
print(cluster_class_percentages)

dominant_percentages = cluster_class_percentages.max(axis=1)
dominant_classes = cluster_class_percentages.idxmax(axis=1)
cluster_class_summary = pd.DataFrame({"dominant_class": dominant_classes, "dominant_percentage": dominant_percentages})

print("\nCluster Class summary:")
print(cluster_class_summary)
print("\nIt seems that unacc is the most frequent class in each cluster\n")

cost = []
k_values = range(2,11)

for k in k_values:
    model = KModes(n_clusters=k, init="random", n_init=5, verbose=1, random_state=42)
    model.fit_predict(df_categorical)
    cost.append(model.cost_)

plt.plot(k_values, cost, marker="o")
plt.title("K-Modes Elbow Method")
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Curve')
plt.savefig('elbow_curve.png')
plt.show()
# by looking at elbow method, we can say 4 is the ideal amount of clusters


