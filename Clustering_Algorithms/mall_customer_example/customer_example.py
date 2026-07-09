import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('Mall_Customers.csv')
print(df.head())
print("\n")

print("Dataset Information")
print(df.info())
print("\n")

X_df = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# scale before KMeans
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

k = 5 # number of centroids
clusters = {}
np.random.seed(23)

for i in range(k):
    center = 4 * (np.random.random(X_scaled.shape[1])) - 2 # between -2 and 2
    points = []
    cluster = {'center': center, 'points': []}
    clusters[i] = cluster

centers_array = np.array([clusters[i]['center'] for i in range(k)])
# array to keep all centroids

# First graph without centroids
fig = plt.figure()
plt.grid(True)
plt.scatter(X_scaled[:,0], X_scaled[:,1])
plt.show()

# Graphic with centroids
plt.scatter(X_scaled[:,0], X_scaled[:,1])
plt.grid(True)
plt.scatter(centers_array[:,0], centers_array[:,1], marker='x', color='red')
plt.show()

def distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

def assign_clusters(X, clusters):
    for idx in range(X.shape[0]): # go through every row
        dist = [] # dist between data point and cluster
        curr_X = X[idx] # current index

        for i in range(k): # go through every cluster
            dis = distance(curr_X, clusters[i]['center']) # calculate distance between cluster and current point
            dist.append(dis) # update the dist list

        curr_cluster = np.argmin(dist) # choose the minimum distance
        clusters[curr_cluster]['points'].append(curr_X) # data point assigned to closest cluster

    return clusters

def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points']) # puts cluster points into array using NumPy
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center

        clusters[i]['points'] = []

    return clusters

def predict_clusters(X, clusters):
    pred = []
    for idx in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[idx], clusters[j]['center']))

        pred.append(np.argmin(dist))

    return pred

for i in range(20):
    clusters = assign_clusters(X_scaled, clusters)
    clusters = update_clusters(X_scaled, clusters)

pred = predict_clusters(X_scaled, clusters)

plt.scatter(X_scaled[:,0], X_scaled[:,1])
plt.grid(False)

for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker= '*', c='red')

plt.show()












