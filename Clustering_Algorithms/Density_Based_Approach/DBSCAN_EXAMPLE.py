import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import adjusted_rand_score

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.50, random_state=0)
# cluster_std controls how spread out each cluster is

db = DBSCAN(eps=0.3, min_samples=10).fit(X) # eps is the radius of each core point
# to define core point, point must have at least 10 neighbors

core_samples_mask = np.zeros_like(db.labels_, dtype=bool) # creates an array where all values are false
core_samples_mask[db.core_sample_indices_] = True # changes core points to True
labels = db.labels_ # labels are cluster numbers, if -1 then it's a noise

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# gives number of clusters, if there is a -1 in the labels, subtract them

unique_labels = set(labels) # Here, we prepare labels and colors
colors = ['y','b','g','r']
print(colors)

for k, col in zip(unique_labels, colors):
    if k == -1: # if the point is noise paint it black
        col = 'k'

    class_member_mask = (labels == k) # True / False mask for current cluster
    xy = X[class_member_mask & core_samples_mask] # Select points which currently in cluster and core point
    plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    # xy[:,0], xy[:,1] means x and y axis
    # uses circle markers and edge colors are black

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:,0], xy[:,1], 'x', markerfacecolor=col, markeredgecolor='k', markersize=6)
    # points where current cluster is not core point, could be border points

plt.title(f'Number of clusters: {n_clusters}')
plt.show()

sc = metrics.silhouette_score(X, labels)
print(f'Silhouette Coefficient: {sc}')
ari = metrics.adjusted_rand_score(y, labels)
print(f'Adjusted Rand Index: {ari}')



