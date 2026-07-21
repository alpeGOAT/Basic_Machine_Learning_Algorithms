import pandas as pd
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx
import numpy as np
from shapely.geometry import box

df = pd.read_csv('olist_geolocation_dataset.csv')

pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', 300)
pd.set_option('display.width', 250)

print("\nNumber of rows in the dataset: ", df.shape[0])
print("Number of columns in the dataset: ", df.shape[1])

print("\nFirst 5 rows of the dataset: ")
print(df.head())

print("\nLast 5 rows of the dataset: ")
print(df.tail())

print("\nMissing values in the dataset: ")
print(df.isnull().sum())

print("\nDataset Information: ")
print(df.info())

print("\nDataset Description: ")
print(df.describe())

df = df.drop_duplicates()
df = df.dropna(subset=['geolocation_lat', 'geolocation_lng'])
df = df[df['geolocation_lat'].between(-90,90) & df['geolocation_lng'].between(-180,180)]

print("\nNumber of unique postal-code prefixes: ")
print(df['geolocation_zip_code_prefix'].nunique())

print("\nNumber of repeated postal-code prefixes: ")
print(df.duplicated(subset=['geolocation_zip_code_prefix']).sum())
# Large difference between number of unique postal-code prefixes and Number of repeated postal-code prefixes
# means that many prefixes has multiple geographical records

# são paulo becomes sao paulo / make sure city names are same
df['geolocation_city'] = df['geolocation_city'].str.strip()
df['geolocation_city'] = df['geolocation_city'].str.lower().str.normalize("NFKD")
df['geolocation_city'] = df['geolocation_city'].str.encode('ascii', errors='ignore').str.decode("utf-8")

df['geolocation_state'] = (df['geolocation_state'].str.upper().str.normalize("NFKD").
                           str.encode('ascii', errors='ignore').str.decode("utf-8"))

print("\nRows where city name is just initials")
print(df[df['geolocation_city'] == "sp"])
print("\n")

df.loc[(df['geolocation_city'] == "sp") & (df["geolocation_state"] == "SP"), "geolocation_city"] = "sao paulo"
print(df[df['geolocation_city'] == "sp"]) # outputs empty dataframe

print("\nFirst 25 rows of the Dataset after converting city and state names")
print(df.head(25))

cities_per_zip_prefix = df.groupby("geolocation_zip_code_prefix")['geolocation_city'].nunique()
states_per_zip_prefix = df.groupby("geolocation_zip_code_prefix")['geolocation_state'].nunique()

zip_with_multiple_cities = cities_per_zip_prefix[cities_per_zip_prefix > 1]
zip_with_multiple_states = states_per_zip_prefix[states_per_zip_prefix > 1]

print("\nPostal code prefixes that have multiple cities: ")
print(zip_with_multiple_cities)
print("\nPostal code prefixes that have multiple states: ")
print(zip_with_multiple_states)

print("\nNumber of Postal code prefixes that have multiple cities ")
print(len(zip_with_multiple_cities))
print("\nNumber of postal code prefixes that have multiple states: ")
print(len(zip_with_multiple_states))

most_frequent_city = df['geolocation_city'].value_counts().idxmax()
print("\nMost frequently shown city in the dataset")
print(most_frequent_city)

most_frequent_state = df['geolocation_state'].value_counts().idxmax()
print("\nMost frequently shown state in the dataset")
print(most_frequent_state)

df_aggregated = (df.groupby('geolocation_zip_code_prefix').agg(geolocation_lat = ("geolocation_lat", "median"),
                                                               geolocation_lng = ("geolocation_lng", "median"),
                                                               geolocation_city = ("geolocation_city", lambda values: values.mode().iloc[0]),
                                                               geolocation_state = ("geolocation_state", lambda values: values.mode().iloc[0])
                                                               ).reset_index()
)

print("\nFirst 10 rows of the aggregated dataset: ")
print(df_aggregated.head(10))
print("\nAggregated dataset shape: ")
print(df_aggregated.shape)

# Number of unique postal code prefixes must be same to number of rows
print("\nNumber of unique postal code prefixes: ")
print(df_aggregated['geolocation_zip_code_prefix'].nunique())

# Amount of duplicated postal code prefixes should be 0
print("\nNumber of duplicated postal-code prefixes:")
print(df_aggregated.duplicated(subset=['geolocation_zip_code_prefix']).duplicated().sum())

print("\nNumber of duplicated coordinate pairs:")
print(df_aggregated.duplicated(subset=['geolocation_lat', 'geolocation_lng']).sum())

print("\nAre there any missing values in the aggregated dataset?")
print(df_aggregated.isnull().sum())

print("\nLatitude range:")
print(df_aggregated["geolocation_lat"].min(), "/", df_aggregated["geolocation_lat"].max())

print("\nLongitude range:")
print(df_aggregated["geolocation_lng"].min(), "/", df_aggregated["geolocation_lng"].max())

print("\nFirst and last 5 rows of the aggregated dataset: ")
print(df_aggregated.head(5).tail(5))

plt.scatter(df_aggregated["geolocation_lng"], df_aggregated["geolocation_lat"], marker='o', c='red',
            edgecolors='black')
plt.title("O Markers Must Resemble the Shape of Brazil")
plt.savefig("brazil1.png")
plt.show()

coordinates = df_aggregated[["geolocation_lat", "geolocation_lng"]].to_numpy()

print("\nCoordinate Data:")
print(coordinates[:5])

print("\nCoordinate Shape:")
print(coordinates.shape)

coordinates_radians = np.radians(coordinates)

print("\nCoordinates in  Radians")
print(coordinates_radians[:5])

min_sample = 20
min_cluster_size = 50
earth_radius = 6371.01

optic = OPTICS(
    min_samples=min_sample,
    min_cluster_size=min_cluster_size,
    metric="haversine",
    xi=0.05,
    cluster_method="xi"
)

labels = optic.fit_predict(coordinates_radians)
df_aggregated["cluster"] = labels

print("\nCluster label counts:")
print(df_aggregated["cluster"].value_counts().sort_index())

unique_labels = np.unique(labels)

number_of_clusters = len(unique_labels) - ( 1 if -1 in unique_labels else 0)
number_of_noise_points = np.sum(labels == -1)
noise_percentage = (number_of_noise_points / len(labels)) * 100

print("\nNumber of clusters: ", number_of_clusters)
print("Number of noise points: ", number_of_noise_points)
print("Noise percentage: ", noise_percentage)

optic_ordering = optic.ordering_
ordered_reachability = (optic.reachability_[optic_ordering])
reachability_km = (ordered_reachability * earth_radius)

finite_values = np.isfinite(reachability_km)
ordering_positions = np.arange(len(optic_ordering))

plt.scatter(optic_ordering[finite_values], reachability_km[finite_values], edgecolors='black',marker='o',
            c='royalblue',alpha=0.7)
plt.title("Reachability Plot")
plt.xlabel("Points in OPTICS Ordering")
plt.ylabel("Reachability Distance (km)")
plt.grid(True)
plt.savefig("reachability_plot.png")
plt.show()

gdf = gpd.GeoDataFrame(df_aggregated, geometry=gpd.points_from_xy(df_aggregated["geolocation_lng"],
                                                                  df_aggregated["geolocation_lat"]), crs='EPSG:4326')

gdf_map = gdf.to_crs(epsg=3857) # convert GeoDataFrame to real map background

fig, ax = plt.subplots(figsize=(12, 10))

gdf_map[gdf_map["cluster"] != -1].plot( # details for clusters
    ax=ax,
    column="cluster",
    cmap="tab20",
    edgecolor="none",
    marker="o",
    markersize=7,
    alpha=0.8,
    legend=False
)

gdf_map[gdf_map["cluster"] == -1].plot( # details for noises
    ax=ax,
    color='black',
    markersize=4,
    marker="x",
    alpha=0.5,
    label='noise'
)

brazil_area = gpd.GeoSeries(
    [box(-74, -34, -34, 6)],
    crs="EPSG:4326"
).to_crs(epsg=3857)

x_min, y_min, x_max, y_max = brazil_area.total_bounds

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

cx.add_basemap( # real map background
    ax,
    source=cx.providers.OpenStreetMap.Mapnik,
    crs = gdf_map.crs,
    zoom = 5
)

ax.set_title("Geographical Distribution of OPTICS Clusters in Brazil")
ax.grid(True)
plt.savefig("brazil.png")
plt.show()




