import numpy as np # for distance calculation
from sklearn.datasets import make_blobs # to create a dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

X,y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=23)
# creates dataset with 500 entries, 2D environment, and 3 clusters

fig = plt.figure(0)
plt.grid(True) # adds an grid to plot
plt.scatter(X[:,0],X[:,1]) # scatter plot featuring x and y axis
plt.show()

scaler = StandardScaler()
X = scaler.fit_transform(X)
# scaling takes the mean and std deviation of data and converts into standard scale value.

clusters = {}
k = 3 # number of centroids
np.random.seed(23) # defined seed number means that we will always choose the same centroids

for i in range(k):
    center = 2*(2*np.random.random((X.shape[1])) - 1) # centroids are between positions -2 and 2
    points = []
    cluster = {'center': center, 'points': []}
    clusters[i] = cluster
# for loop used to create k amount of random centroids

print("\n")
print(clusters) # prints the location of each centroid (cluster)

plt.scatter(X[:,0],X[:,1])
plt.grid(True)


for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='*', c='red')
plt.show()
# Here, we plot the data points and centroids
# Centroids are in '*' shape and red color

def distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))
# finding the distance between data points using euclidean formula

# these function created to assign clusters
def assign_clusters(X, clusters):
    for idx in range(X.shape[0]): # checks every row
        dist = [] # distance between current point and each centroid available
        curr_X  = X[idx] # current data point

        for i in range(k): # checks every cluster
            dis = distance(curr_X, clusters[i]['center']) # distance between current data point and center[i]
            dist.append(dis) # then, update the distance to list named dist[]

        curr_cluster = np.argmin(dist) # finds the index of smallest dist[i]
        clusters[curr_cluster]['points'].append(curr_X)
        # data point will be assigned to closest cluster

    return clusters

# updates the location of centroids by taking mean of the points in its surroundings
def update_clusters(X, clusters):
    for i in range(k): # go through each centroid
        points = np.array(clusters[i]['points']) # turns the cluster points to NumPy array so mean can found
        if points.shape[0] > 0: # checks if cluster has point or not
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center # new centroid defined

            clusters[i]['points'] = [] # clear point list

    return clusters

# this function used to predict the centroid for each data point
def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]): # check every row
        dist = [] # list for distance between the current data point and centroid
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(np.argmin(dist)) # appends the minimum distance between centroid and data point to prediction list

    return pred

clusters = assign_clusters(X, clusters)
clusters = update_clusters(X, clusters)
pred = pred_cluster(X, clusters)

plt.scatter(X[:,0],X[:,1], c=pred)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker= '^', c='red')

plt.show()

