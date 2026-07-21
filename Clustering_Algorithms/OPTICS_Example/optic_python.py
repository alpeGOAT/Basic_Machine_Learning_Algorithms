
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, Y = make_blobs(n_samples=300, centers=3, random_state=42)

clustering = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
clustering.fit(X)

labels = clustering.labels_

plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow', edgecolor='k')
plt.title("OPTICS Clustering Using Synthetic Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("OPTICS Clustering Using Synthetic Data.png")
plt.show()
