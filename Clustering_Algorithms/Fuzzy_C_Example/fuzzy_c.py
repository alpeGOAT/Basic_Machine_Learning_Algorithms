from matplotlib import pyplot as plt
import numpy as np
import skfuzzy as fuzz

np.random.seed(0) # seed number given so result will be same each time code run
center = 0.5
spread = 0.1

data = center + spread * np.random.randn(2,100) # 100 data points in 2D space
data = np.clip(data,0,1) # data point are between 0 and 1

# Fuzzy-C Parameters
n_clusters = 3
m = 1.7 # higher this value, cluster memberships will be softer
error = 1e-5 # stopping condition, if the value change is smaller than this value, algorithm stops running.
maxiter = 2000 # maximum number of iterations (how many times code can run)

cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(data, c=n_clusters, maxiter=maxiter, m=m,
                                               error=error, init=None)
# cntr is the final cluster centers
# u is a matrix indicates degree of belonging of point to each cluster

hard_clusters = np.argmax(u,axis=0) # we taken the highest value from each point's degree to define one hard cluster
                                    #for each data point

print("Cluster Centers: ")
print(cntr)
print("\n")

print("Fuzzy Membership Matrix: (first 5 data points)")
print(u[:,:5])
print("\n")

print("Hard Clusters of first 5 data points: ")
print(hard_clusters[:5])
print("\n")

fig, ax = plt.subplots(figsize=(8,6))

for i in range(n_clusters):
    ax.scatter(data[0], data[1], c=u[i], cmap='coolwarm', alpha=0.5, label=f'Fuzzy Cluster: {i+1}') #alpha = transparency rate
# c = u[i] means that points colored depend on how strongly they belong to each cluster
# For example if data point has the highest degree on cluster 2, it means that cluster 2 will be most colored

markers = ['o','s','^']
colors = ['b','g','orange'] # blue, green, orange for clusters 1,2,3

for i in range(n_clusters):
    cluster_points = data[:, hard_clusters == i]
    ax.scatter(cluster_points[0], cluster_points[1], c=colors[i], marker=markers[i],
               edgecolors='k', s=80, label=f'Hard Cluster: {i+1}')

ax.scatter(cntr[:,0], cntr[:,1], c='red', marker='x', s=200, label='Cluster Centers')

ax.set_title("Fuzzy C-Means")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend(loc='upper left') # legend will be in the upper left
plt.show()










