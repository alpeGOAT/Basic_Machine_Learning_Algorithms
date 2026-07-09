import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, centers=3, random_state=42, cluster_std=[1.0, 1.5, 0.8])
# 500 data points, 3 centroids, and spread of each cluster

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42) # create 3 components
# covariance_type='full' means that cluster can have any shape
gmm.fit(X) # train the given data
labels = gmm.predict(X) # cluster labels assigned to each data
# GMM is a soft clustering algorithm, but predict will give the most high percentile cluster

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=50, edgecolors='k')
# s=50 meant size of the points

plt.scatter(
    gmm.means_[:, 0],
    gmm.means_[:, 1],
    c='red',
    marker='X',
    label='Centers'
)

# plotting the dataset
plt.title('Gaussian Mixture Model')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.legend()
plt.show()



