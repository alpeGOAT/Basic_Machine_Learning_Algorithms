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

# Proportionally reduce plotted data points (stratified by cluster) for clarity
sample_frac = 0.15
gdf_sampled = gdf_map.groupby("cluster", group_keys=False).sample(frac=sample_frac, random_state=42)
print(f"Plotted Points (Proportional {int(sample_frac * 100)}% Sample): {len(gdf_sampled)} / {len(gdf_map)}\n")

fig, ax = plt.subplots(figsize=(12, 10))

gdf_sampled[gdf_sampled["cluster"] != -1].plot( # details for clusters
    ax=ax,
    column="cluster",
    cmap="tab20", # give different color for each cluster
    markersize=10,
    marker="o",
    alpha=0.35, # transparency rate, 1-> non transparent, 0.2-> very transparent
    legend=True,
    legend_kwds={"label": "DBSCAN Cluster ID (0–12)"},
    zorder=2
)

gdf_sampled[gdf_sampled["cluster"] == -1].plot( # details for noises
    ax=ax,
    color='black',
    markersize=8,
    marker="x",
    alpha=0.45,
    label='Noise',
    zorder=2
)

# Zoom into Prague city center (convert WGS84 bounds to EPSG:3857)
bounds_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy([14.378, 14.465], [50.056, 50.104]),
    crs="EPSG:4326"
).to_crs(epsg=3857)
ax.set_xlim(bounds_gdf.geometry.x.iloc[0], bounds_gdf.geometry.x.iloc[1])
ax.set_ylim(bounds_gdf.geometry.y.iloc[0], bounds_gdf.geometry.y.iloc[1])

cx.add_basemap( # real map background
    ax,
    source=cx.providers.Esri.WorldStreetMap,
    crs=gdf_map.crs,
    zoom=14,
    attribution="Tiles © Esri",
    attribution_size=7
)

# Explain the colorbar (cluster IDs 0-12) in the bottom-left corner
colorbar_info = (
    "Colorbar (0–12): DBSCAN Cluster IDs\n"
    "• Each color represents a density cluster\n"
    "  (eps = 0.5 km, min_samples = 14)\n"
    "• Cluster 0 (blue): Main city-center cluster\n"
    "• Clusters 1–12: Outer neighborhood clusters"
)
ax.text(
    0.015, 0.04,
    colorbar_info,
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="bottom",
    horizontalalignment="left",
    color="#1a1a1a",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#555555", alpha=0.9, linewidth=0.8),
    zorder=7
)

# Highlight major Prague tourist landmarks
landmarks = [
    ("Prague Castle", 50.0909, 14.4005, (-15, 16)),
    ("Petřín Tower", 50.0835, 14.3951, (-65, -18)),
    ("Charles Bridge", 50.0865, 14.4114, (-80, -20)),
    ("Old Town Square", 50.0875, 14.4213, (12, 16)),
    ("Wenceslas Square", 50.0813, 14.4275, (14, -14)),
    ("Dancing House", 50.0755, 14.4141, (-82, -16)),
    ("Vyšehrad", 50.0644, 14.4180, (12, -12)),
    ("Letná Park", 50.0956, 14.4158, (12, 12)),
    ("Žižkov TV Tower", 50.0811, 14.4511, (-45, 16)),
]

landmarks_gdf = gpd.GeoDataFrame(
    {"name": [lm[0] for lm in landmarks], "offset": [lm[3] for lm in landmarks]},
    geometry=gpd.points_from_xy([lm[2] for lm in landmarks], [lm[1] for lm in landmarks]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

landmarks_gdf.plot(
    ax=ax,
    color="#d62728",
    edgecolor="white",
    linewidth=0.9,
    markersize=140,
    marker="*",
    label="Tourist Landmarks",
    zorder=5
)

for _, row in landmarks_gdf.iterrows():
    ax.annotate(
        row["name"],
        xy=(row.geometry.x, row.geometry.y),
        xytext=row["offset"],
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color="#1a1a1a",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d62728", alpha=0.88, linewidth=0.8),
        arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.9),
        zorder=6
    )

# Highlight specific Airbnb listing (ID: 1199826022731689115) as "My Room"
my_room_gdf = gpd.GeoDataFrame(
    {"name": ["My Room"]},
    geometry=gpd.points_from_xy([14.454452], [50.079816]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

my_room_gdf.plot(
    ax=ax,
    color="#2ca02c",
    edgecolor="white",
    linewidth=0.9,
    markersize=150,
    marker="*",
    label="My Room",
    zorder=8
)

my_room_pt = my_room_gdf.geometry.iloc[0]
ax.annotate(
    "My Room",
    xy=(my_room_pt.x, my_room_pt.y),
    xytext=(10, -16),
    textcoords="offset points",
    fontsize=7,
    fontweight="bold",
    color="#1a1a1a",
    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#2ca02c", alpha=0.92, linewidth=0.8),
    arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=0.8),
    zorder=9
)

cluster_price = df.groupby("cluster")["price"].mean()
print("Average Price Per Cluster")
print(cluster_price)
print("\n")

ax.set_title("Prague Airbnb Locations\n(City Center, 15% Proportional Sample)", fontsize=13, fontweight="bold", pad=10)

ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True)
ax.set_axis_off()
plt.savefig("dbscan_prague_airbnb_map.png", dpi=300, bbox_inches="tight")
plt.show()











