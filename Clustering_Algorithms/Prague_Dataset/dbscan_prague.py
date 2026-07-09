import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import geopandas as gpd
import contextily as cx

df = pd.read_csv('listings.csv') # Airbnb Prague Dataset

print("First 5 Rows of Dataset")
print(df.head())
print("\n")

print("Dataset Information")
print(df.info())
print("\n")

print("Missing Values")
print(df.isnull().sum())
print("\n")

df = df[["name", "neighbourhood_cleansed", "latitude", "longitude", "room_type", "price"]]

print("Dataset with Selected Columns")
print(df.head())
print("\n")

print("Missing Values After Selecting Columns")
print(df.isnull().sum())
print("\n")

df = df.dropna(subset=['latitude', 'longitude'])
# delete rows where latitude and longitude empty

df["price"] = (df["price"].astype(str).str.replace("$","", regex=False))
df["price"] = (df["price"].astype(str).str.replace(",","", regex=False))
# changing how price look 2,500 -> 2500 || 3,400$ --> 3400

df["price"] = pd.to_numeric(df["price"], errors='coerce')
# if price can not converted to number, give NaN (missing value)

print("Dataset After Cleaning")
print(df.head())
print("\n")

print("Missing Values After Cleaning")
print(df.isnull().sum())
print("\n")

coordinates = df[["latitude", "longitude"]].to_numpy() # creates a numPy array from latitude, longitude rows

print("Coordinate Date:")
print(coordinates[:5])
print("\n")

print("Coordinate Shape:")
print(coordinates.shape)
print("\n")

# Convert coordinates to radians for haversine distance
# Haversine distance is distance between two locations by looking at their longitude and latitude
coordinates_radians = np.radians(coordinates)
# radians = degrees × π / 180, we convert the coordinates values to radians so Haversine Distance can be applied

print("Coordinates in  Radians")
print(coordinates_radians[:5])
print("\n")

earth_radius_km = 6371.01
eps_km = 0.5
min_samples = 14

dbscan = DBSCAN(eps=eps_km / earth_radius_km, min_samples=min_samples, metric = 'haversine')

labels = dbscan.fit_predict(coordinates_radians) # creates cluster label for every listing
df["cluster"] = labels # add a new row named cluster labels
# if label = -1, it means that it's a noise

print("Cluster Labels")
print(df["cluster"].value_counts())
print("\n")

num_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of Clusters: ", num_of_clusters)
print("\n")

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs='EPSG:4326')
# We convert normal dataframe to GeoDataFrame

print("GeoDataFrame")
print(gdf.head())
print("\n")

gdf_map = gdf.to_crs(epsg=3857) # convert GeoDataFrame to real map background

fig, ax = plt.subplots(figsize=(10,8))

gdf_map[gdf_map["cluster"] != -1].plot( # details for clusters
    ax=ax,
    column="cluster",
    cmap="tab20", # give different color for each cluster
    markersize=6,
    marker="o",
    alpha=0.8, # transparency rate, 1-> non transparent, 0.3-> very transparent
    legend=True
)

gdf_map[gdf_map["cluster"] == -1].plot( # details for noises
    ax=ax,
    color='black',
    markersize=4,
    marker="x",
    alpha=0.8,
    label='noise'
)

cx.add_basemap( # real map background
    ax,
    source=cx.providers.OpenStreetMap.Mapnik,
    crs = gdf_map.crs,
    zoom = 12
)

cluster_price = df.groupby("cluster")["price"].mean()
print("Average Price Per Cluster")
print(cluster_price)
print("\n")

ax.set_title("DBSCAN Clustering Using Prague Airbnb Listings")

ax.legend()
ax.grid(True)
ax.set_axis_off()
plt.savefig("dbscan_prague_airbnb_map.png", dpi=300, bbox_inches="tight")
plt.show()











