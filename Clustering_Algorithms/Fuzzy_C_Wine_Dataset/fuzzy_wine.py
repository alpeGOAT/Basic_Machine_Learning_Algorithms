from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

wine = load_wine()

x = wine.data
y = wine.target

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

# Fuzzy-C Parameters
n_clusters = 3
m = 1.7 #
error = 1e-5
maxiter = 2000

data = x_pca.T # convert samples x features to features x samples
# (178, 2) --> (2, 178)

cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(data, c=n_clusters, maxiter=maxiter, m=m,
                                               error=error, init=None)

hard_clusters = np.argmax(u,axis=0)

fig, ax = plt.subplots(figsize=(8,6))

for i in range(n_clusters):
    ax.scatter(data[0], data[1], c=u[i], cmap='coolwarm', alpha=0.5, label=f'Fuzzy Cluster: {i+1}')

markers = ['o','s','^']
colors = ['b','g','orange'] # blue, green, orange for clusters 1,2,3

for i in range(n_clusters):
    cluster_points = data[:, hard_clusters == i]
    ax.scatter(cluster_points[0], cluster_points[1], c=colors[i], marker=markers[i],
               edgecolors='k', s=80, label=f'Hard Cluster: {i+1}')

ax.scatter(cntr[:,0], cntr[:,1], c='red', marker='x', s=200, label='Cluster Centers')

ax.set_title("Fuzzy C-Means with Wine Dataset")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
plt.show()








