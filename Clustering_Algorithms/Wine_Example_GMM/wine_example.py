from sklearn.datasets import load_wine # wine dataset from scikit-learn
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # PCA used to lower the amount of features which is 13 to just 2 PCA features

wine = load_wine()

x = wine.data
y = wine.target

print("Feature Names: ", wine.feature_names)
print("Target Names: ", wine.target_names)
print("Data Shape: ", x.shape)
print("Target Shape: ",y.shape)
print("\n")

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
# We need to scale data before any important operation

pca = PCA(n_components=2) # we still use 13 features but its converted to only 2 features since we plot dataset in 2D Shape
x_pca = pca.fit_transform(x_scaled)
print(x_pca.shape)
print("\n")

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(x_pca) # train the Wine Dataset
labels = gmm.predict(x_pca) # predict the cluster label

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=labels, cmap='viridis', s=50, edgecolors='k')

plt.scatter(
    gmm.means_[:, 0],
    gmm.means_[:, 1],
    c='red',
    marker='X',
    label='GMM_Means',
    s=100
)

plt.title("Gaussian Mixture Model With Wine Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()




