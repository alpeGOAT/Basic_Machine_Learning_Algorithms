# Music_Listener_Habits
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', 100)

df = pd.read_csv('music_streaming_habits_2026.csv')

print("First 5 rows of the Music Dataset")
print(df.head())
print("\n")

print("Music Dataset Information")
print(df.info())
print("\n")

print("Music Dataset Description")
print(df.describe())
print("\n")

# ------------------------------------------------------------------------------- #

df_for_analysis = df.copy()
df_for_analysis = df_for_analysis[["listener_id", "daily_listening_minutes"]]

users_with_most_daily_listening_minutes = df_for_analysis.sort_values(by="daily_listening_minutes", ascending=False)
print("20 Users with most daily listening minutes")
print(users_with_most_daily_listening_minutes.head(20))
print("\n")

plt.scatter(users_with_most_daily_listening_minutes['listener_id'].head(20),
         users_with_most_daily_listening_minutes['daily_listening_minutes'].head(20), color='red',alpha=0.7)

plt.title("Top 20 Users with most daily listening minutes",y=1.02)
plt.xlabel("Listener ID")
plt.ylabel("Daily listening minutes", labelpad=10, fontsize=12)
plt.grid(True)
plt.savefig("top_users_with_daily_listening_minutes.png")
plt.show()

df_for_analysis1 = df.copy()
df_for_analysis1 = df_for_analysis1[["listener_id", "age"]].head(250)

users_older_than_45 = df_for_analysis1[df_for_analysis1['age'] > 45]

plt.scatter(df_for_analysis1['listener_id'].head(250), df_for_analysis1['age'].head(250), color='cyan',
            alpha=0.6, edgecolors='black')

plt.scatter(users_older_than_45['listener_id'], users_older_than_45['age'],
            color='red',alpha=0.7,
            edgecolors='black',label="Users older than 45")

plt.title("User Age Distribution (First 250 Users)")
plt.xlabel("Listener ID")
plt.ylabel("Age")
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig("first_250_users_and_users_older_than_45.png")
plt.show()

df_behaviour = df.copy()

df_behaviour = df_behaviour[["daily_listening_minutes", "playlists_count",
    "skip_rate_pct" ,"discover_weekly_user", "uses_offline_mode", "podcasts_too"]]

# makem sure every column is numerical
df_behaviour[["discover_weekly_user", "uses_offline_mode", "podcasts_too"]] = df_behaviour[["discover_weekly_user", "uses_offline_mode",
                                                                                            "podcasts_too"]].astype(int)

print("Dataset to use for Behavioural Analysis")
print(df_behaviour.head())
print("\n")

scaler = StandardScaler()
scaled_behaviour = scaler.fit_transform(df_behaviour)

scaled_behaviour = pd.DataFrame(
    scaled_behaviour,
    columns=df_behaviour.columns,
    index=df_behaviour.index
)

print("Dataset Description after converting into Data Frame")
print(scaled_behaviour.describe())
print("\n")

inertia_values = []

for k in range(2,11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    kmeans.fit(scaled_behaviour)
    inertia_values.append(kmeans.inertia_)

# Elbow method used to find ideal number of clusters for clustering
plt.plot(range(2, 11), inertia_values, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(range(2, 11))
plt.grid(True)
plt.savefig("elbow_method.png")
plt.show()
# number of clusters might be 4 after looking at graph

# Let's add Silhouette Score to make sure about cluster amount
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(scaled_behaviour)
    score = silhouette_score(scaled_behaviour, cluster_labels)
    silhouette_scores.append(score)

    print("Cluster amount:", k)
    print("Silhouette score:", score)
    print("--" * 20)

plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.title("Silhouette Curve")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette score")
plt.xticks(range(2, 11))
plt.grid(True)
plt.savefig("silhouette_curve.png")
plt.show()

# Elbow method taken into consideration
no_of_clusters = 4

k_means = KMeans(n_clusters=no_of_clusters, random_state=42, n_init=10)
k_means.fit(scaled_behaviour)

scaled_behaviour['kmeans_4'] = k_means.labels_

print("Cluster Labels of first 5 users")
print(scaled_behaviour['kmeans_4'].head())
print("\n")

df["cluster_label"] =pd.NA

df['cluster_label'] = scaled_behaviour['kmeans_4']
print("Number of users in each cluster")
print(df["cluster_label"].value_counts().sort_index())
print("\n")

behaviour_columns = ["daily_listening_minutes", "playlists_count", "skip_rate_pct",
                    "discover_weekly_user", "uses_offline_mode", "podcasts_too"
]

cluster_profiles = df.groupby("cluster_label")[behaviour_columns].mean()
print("Behavioural Profiles of Clusters")
print(cluster_profiles.round(2))
print("\n")

cluster_profiles_display = cluster_profiles.copy()
boolean_columns = ["discover_weekly_user", "uses_offline_mode", "podcasts_too"]

cluster_profiles_display[boolean_columns] = (cluster_profiles_display[boolean_columns] * 100)
print("Behavioural Profiles of Clusters -Boolean columns as percentages-")
print(cluster_profiles_display.round(2))
print("\n")

cluster_sizes = df["cluster_label"].value_counts().sort_index()
cluster_percentages = df["cluster_label"].value_counts(normalize=True).sort_index().mul(100).round(2)

cluster_size_summary = pd.DataFrame({'user_count':cluster_sizes, 'user_percentage':cluster_percentages})
print("Cluster Size Summary")
print(cluster_size_summary)
print("\n")

plt.bar(cluster_size_summary.index, cluster_size_summary['user_percentage'],color=["royalblue", "orange",
                                                                                   "green", "red"],edgecolor='black')

for cluster, percentage in cluster_percentages.items():
    plt.text(
        cluster,
        percentage + 0.3,
        f"{percentage:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.title("Percentage of Users in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Users (%)")
plt.grid(True)
plt.savefig("cluster_percentages.png")
plt.show()

genre_distribution = pd.crosstab(df["cluster_label"], df["top_genre"])
print("Genre Distribution")
print(genre_distribution)
print("\n")

genre_percentage = pd.crosstab(df["cluster_label"], df["top_genre"], normalize=True).mul(100)
dominant_genres = genre_percentage.idxmax(axis=1)
dominant_genres_amount_of_clusters = dominant_genres.value_counts()

print("Dominant Genre of Each Cluster")
print(dominant_genres)
print(dominant_genres_amount_of_clusters)
print("\n")

plt.bar(dominant_genres.index, [1] * len(dominant_genres), color=["cyan", "blue", "green", "orange"],
        facecolor="lightblue")

for cluster in dominant_genres.index:
    genre = dominant_genres.loc[cluster]
    plt.text(
        cluster,
        0.5,
        genre,
        ha="center",
        va="center",
        fontweight="bold"
    )

plt.title("Top Genre of Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Genre")
plt.xticks(dominant_genres.index)
plt.savefig("top_genres_each_cluster.png")
plt.show()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_behaviour)

pca_input = scaled_behaviour.drop(
    columns=["kmeans_4"],
    errors="ignore"
)

pca_df = pd.DataFrame(pca_result,columns=["PCA1", "PCA2"], index=df.index)
pca_df["cluster_label"] = cluster_labels

colors = ["red", "blue", "green", "orange"]

for cluster, color in zip(
    sorted(pca_df["cluster_label"].unique()),
    colors
):
    cluster_data = pca_df[pca_df["cluster_label"] == cluster]

    plt.scatter(cluster_data["PCA1"], cluster_data["PCA2"], marker='x',
                color=color, alpha=0.6, label=f"Cluster:{cluster}")

plt.title("K Means Clustering")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.grid(True)
plt.legend(loc='upper left')
plt.savefig("pca_clusters.png")
plt.show()