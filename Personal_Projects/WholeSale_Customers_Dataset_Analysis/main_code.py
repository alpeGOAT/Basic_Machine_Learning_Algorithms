import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 350)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_colwidth', 40)

customers = pd.read_csv('Wholesale customers data.csv')

print("\nFirst 5 Rows of the Dataset")
print(customers.head())

print("\nLast 5 Rows of the Dataset")
print(customers.tail())

print("\nDataset Information")
customers.info()

print("\nDataset Shape")
print(customers.shape)

print("\nDataset Description")
print(customers.describe())

print("\nAre there any missing values in the Dataset?")
print(customers.isnull().sum())

if customers.isnull().sum().sum() > 0:
    print("\nThere are missing values in the Dataset")
elif customers.isnull().sum().sum() == 0:
    print("\nThere are no missing values in the Dataset")

customers = customers.drop_duplicates()

# Channel Distribution (Horeca or Retail)
channel_distribution = customers['Channel'].value_counts().sort_index()
print("\nChannel Distribution")
print(channel_distribution, "\n")

plt.bar(channel_distribution.index, channel_distribution.values, color='red', alpha=0.5)
plt.xlabel('Channel',labelpad=5)
plt.xticks([1,2], ['Horeca', 'Retail'])
plt.ylabel('Count')
plt.grid(True, alpha=0.5)
plt.title('Channel Distribution (Horeca or Retail)', fontweight='bold')
plt.savefig('channel_distribution.png')
plt.show()

channel_distribution_percentages = (customers['Channel'].value_counts(normalize=True).mul(100).round(2).reset_index())
channel_distribution_percentages.columns = ['Channel', 'Channel Percentage']
print(channel_distribution_percentages, "\n")

# Region Distribution (Lisbon, Oporto, Other Region)
region_distribution = customers['Region'].value_counts().sort_index()
print("\nRegion Distribution")
print(region_distribution, "\n")

plt.bar(region_distribution.index, region_distribution.values, color='red', alpha=0.5)
plt.xlabel('Region')
plt.xticks([1,2,3], ['Lisbon', 'Oporto', 'Other Region'])
plt.ylabel('Count')
plt.grid(True, alpha=0.5)
plt.title('Region Distribution (Lisbon, Oporto, Other Region)', fontweight='bold')
plt.savefig('region_distribution.png')
plt.show()

region_distribution_percentages = (customers['Region'].value_counts(normalize=True).mul(100).round(2).sort_index())
region_distribution_percentages.columns = ['Region', 'Region Percentage']
print(region_distribution_percentages, "\n")

avg_grocery_spendings = customers['Grocery'].mean().round(2)
print("\nAverage Grocery Spendings")
print(avg_grocery_spendings)

avg_milk_purchases = customers['Milk'].mean().round(2)
print("\nAverage Milk Purchases")
print(avg_milk_purchases)

avg_fresh_purchases = customers['Fresh'].mean().round(2)
print("\nAverage Fresh Purchases")
print(avg_fresh_purchases)

avg_delicatessen_purchases = customers['Delicatessen'].mean().round(2)
print("\nAverage Delicatessen Purchases")
print(avg_delicatessen_purchases)

avg_frozen_purchases = customers['Frozen'].mean().round(2)
print("\nAverage Frozen Purchases")
print(avg_frozen_purchases)

avg_detergents_purchases = customers['Detergents_Paper'].mean().round(2)
print("\nAverage Detergents/Paper Purchases")
print(avg_frozen_purchases)

print("\nMost Expensive Grocery Spending:", customers['Grocery'].max())
print("\nMost Expensive Milk Spending:", customers['Milk'].max())
print("\nMost Expensive Fresh Spending:", customers['Fresh'].max())
print("\nMost Expensive Delicatessen Spending:", customers['Delicatessen'].max())
print("\nMost Expensive Frozen Spending:", customers['Frozen'].max())
print("\nMost Expensive Detergents/Paper Spending:", customers['Detergents_Paper'].max())

columns = ['Grocery', 'Milk', 'Fresh', 'Delicatessen', 'Frozen', 'Detergents_Paper']

max_values = customers[columns].max()

plt.bar(max_values.index, max_values, color='red', alpha=0.5, width=0.6)
plt.xlabel('Numerical Columns')
plt.xticks(fontsize=8)
plt.ylabel('Max Label',labelpad=-3)
plt.grid(True, alpha=0.5)
plt.title('Each Columns Maximum Value', fontweight='bold')
plt.savefig('Each Columns Maximum Value')
plt.tight_layout()
plt.show()

avg_values = customers[columns].mean().round(2)
plt.bar(avg_values.index, avg_values, color='red', alpha=0.5, width=0.6)
plt.xticks(fontsize=8)
plt.xlabel('Numerical Columns')
plt.ylabel('Average Values',labelpad=-3)
plt.title('Each Columns Average Value', fontweight='bold')
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('Each Columns Average Value.png')
plt.show()

print("\nEach Channels Average Spending")
channel_means = customers.groupby('Channel')[columns].mean().round(2)
print(channel_means, "\n")

channel_means_plot = channel_means.reset_index().melt(
    id_vars='Channel',
    var_name='Product',
    value_name='Average Spending'
)

print(channel_means_plot, "\n")

sns.barplot(data=channel_means_plot, x='Product',y='Average Spending', hue='Channel')
plt.xlabel('Product')
plt.ylabel('Average Spending')
plt.tight_layout()
plt.title('Each Channel Average Spending', fontweight='bold')
plt.savefig('Each Channel Average Spending.png')
plt.tight_layout()
plt.show()

numerical_features = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicatessen']

X = customers[numerical_features]
y = customers['Channel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=26, stratify=y)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred), "\n")

X1 = customers[numerical_features]
y1 = customers['Region']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=26, stratify=y1)

rfc1 = RandomForestClassifier(n_estimators=100, random_state=42)
rfc1.fit(X1_train, y1_train)
y1_pred = rfc1.predict(X1_test)

print("Accuracy:", accuracy_score(y1_test, y1_pred))

print("\nClassification Report:")
print(classification_report(y1_test, y1_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y1_test, y1_pred), "\n")

clustering_features = [
    'Fresh',
    'Milk',
    'Grocery',
    'Frozen',
    'Detergents_Paper',
    'Delicatessen'
]

X_cluster = customers[clustering_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

inertias = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o', color='red', alpha=0.5)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True, alpha=0.5)
plt.savefig('elbow_method.jpg')
plt.show()

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)

    print("Cluster amount:", k)
    print("Silhouette score:", score) # k=3 is suitable for clustering
    print("--" * 20)

plt.plot(range(2, 11), silhouette_scores, marker="o", color='red', alpha=0.5)
plt.title("Silhouette Curve")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette score")
plt.xticks(range(2, 11))
plt.grid(True,alpha=0.5)
plt.savefig("silhouette_curve.png")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

customers['Cluster'] = kmeans.fit_predict(X_scaled)
print(customers['Cluster'].value_counts().sort_index(), "\n")
print(customers.groupby('Cluster')[numerical_features].mean().round(2), "\n")

cluster_percentages = customers['Cluster'].value_counts(normalize=True).mul(100).round(2).sort_index().reset_index()
cluster_percentages.columns = ['Cluster', 'Cluster_Percentage']
print(cluster_percentages, "\n")

cluster_counts = customers['Cluster'].value_counts().sort_index()

plt.bar(cluster_counts.index, cluster_counts.values, color='red', alpha=0.5)
plt.xlabel('Clusters')
plt.xticks([0,1,2])
plt.ylabel('Value Counts', labelpad=13)
plt.grid(True, alpha=0.5)
plt.title('Cluster Distribution (bar)')
plt.savefig('cluster_distribution_bar.png')
plt.show()

cluster_means = (customers.groupby('Cluster')[numerical_features].mean().round(2))
print(cluster_means, "\n")

cluster_means_plot = cluster_means.reset_index().melt(id_vars='Cluster', var_name='Product', value_name='Average Spending')
print(cluster_means_plot, "\n")

plt.figure(figsize=(12, 12))
sns.barplot(data=cluster_means_plot, x='Product', y='Average Spending', hue='Cluster')
plt.xlabel('Product',fontsize=18,labelpad=15)
plt.ylabel('Average Spending',fontsize=18,labelpad=15)
plt.title('Average Spending by Cluster', fontweight='bold',fontsize=18,y=1.05)
plt.grid(True, alpha=0.5)
plt.savefig('avg_spending_by_cluster.jpg')
plt.show()

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_scaled)

plt.scatter(x_pca[:,0], x_pca[:,1], c=customers['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('K-Means Customer Clusters', fontweight='bold')
plt.grid(True, alpha=0.5)
plt.savefig('k_means_customer_cluster.png')
plt.show()
