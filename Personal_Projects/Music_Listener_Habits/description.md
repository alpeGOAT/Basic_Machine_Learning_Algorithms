
# Music Listener Habits: K-Means Clustering

This project analyzes a music-streaming habits dataset and groups listeners into behavioral segments using **K-Means clustering**. It combines exploratory data analysis, feature preprocessing, cluster evaluation, profile interpretation, genre analysis, and PCA-based visualization.

## Project Objective

The main goal is to identify groups of listeners with similar streaming behavior. The analysis considers how long users listen each day, how many playlists they have, how frequently they skip tracks, and whether they use features such as Discover Weekly, offline mode, and podcasts.

The resulting clusters can help describe different listener personas and may support personalized recommendations, marketing strategies, and product-feature decisions.

## Dataset

The program reads the following CSV file:

```text
music_streaming_habits_2026.csv
```

Important columns used in the project include:

| Column | Description |
| --- | --- |
| `listener_id` | Unique identifier for each listener |
| `age` | Age of the listener |
| `daily_listening_minutes` | Average listening time per day |
| `playlists_count` | Number of playlists owned by the listener |
| `skip_rate_pct` | Percentage of tracks skipped |
| `discover_weekly_user` | Whether the listener uses Discover Weekly |
| `uses_offline_mode` | Whether the listener uses offline mode |
| `podcasts_too` | Whether the listener also listens to podcasts |
| `top_genre` | Listener's most frequently played genre |

## Technologies Used

- Python
- pandas
- Matplotlib
- scikit-learn
  - `StandardScaler`
  - `KMeans`
  - `silhouette_score`
  - `PCA`

## Program Workflow

### 1. Dataset inspection

The dataset is loaded with pandas. The program prints:

- The first five rows
- Dataset structure and data types
- Descriptive statistics for numerical columns

These checks provide an initial understanding of the data before analysis begins.

### 2. Daily listening-time analysis

Listeners are sorted by `daily_listening_minutes`, and the 20 users with the highest daily listening time are displayed and plotted.

Output:

```text
top_users_with_daily_listening_minutes.png
```

### 3. Age analysis

The program visualizes the ages of the first 250 listeners. Users older than 45 are highlighted in red so that they can easily be distinguished from the rest of the sample.

Output:

```text
first_250_users_and_users_older_than_45.png
```

### 4. Behavioral feature selection

The following features are selected for clustering:

```python
behaviour_columns = [
    "daily_listening_minutes",
    "playlists_count",
    "skip_rate_pct",
    "discover_weekly_user",
    "uses_offline_mode",
    "podcasts_too"
]
```

The three Boolean columns are converted into numerical values:

- `True` becomes `1`
- `False` becomes `0`

### 5. Feature scaling

The selected features have different units and ranges. For example, daily listening time is measured in minutes, while the Boolean features contain only 0 and 1.

`StandardScaler` standardizes each feature so that it has approximately:

- Mean = 0
- Standard deviation = 1

Scaling prevents features with larger numerical values from having an unfair influence on K-Means clustering.

### 6. Choosing the number of clusters

The program evaluates cluster counts from 2 through 10 using two methods.

#### Elbow method

The inertia value is calculated for each value of `k`. Inertia represents the total squared distance between data points and their assigned cluster centers. The point at which the improvement begins to slow down is considered a possible elbow.

Output:

```text
elbow_method.png
```

#### Silhouette score

The Silhouette Score evaluates how similar each listener is to their own cluster compared with other clusters. Its value ranges from -1 to 1, and a higher score normally indicates better-separated and more cohesive clusters.

Output:

```text
silhouette_curve.png
```

After examining these results, the program uses four clusters.

### 7. K-Means clustering

The final model is created with:

```python
KMeans(n_clusters=4, random_state=42, n_init=10)
```

- `n_clusters=4` creates four listener groups.
- `random_state=42` makes the result reproducible.
- `n_init=10` runs K-Means with ten different centroid initializations and retains the best result.

Each listener receives a `cluster_label` from 0 to 3.

### 8. Cluster profiling

The mean value of every behavioral feature is calculated for each cluster. This makes it possible to compare the typical behavior of the four listener groups.

For easier interpretation, the mean values of the Boolean columns are multiplied by 100 and presented as percentages. For example, an offline-mode mean of `0.72` means that approximately 72% of the listeners in that cluster use offline mode.

The exact meaning of a cluster should be determined from its final profile rather than from its numeric label. Cluster numbers are identifiers and do not indicate ranking or quality.

### 9. Cluster-size analysis

The program calculates both the number and percentage of users assigned to each cluster. A bar chart displays the percentage represented by each group.

Output:

```text
cluster_percentages.png
```

### 10. Dominant-genre analysis

A cross-tabulation compares `cluster_label` with `top_genre`. The program then identifies the most common genre within each cluster.

When calculating within-cluster percentages, row normalization should be used:

```python
genre_percentage = pd.crosstab(
    df["cluster_label"],
    df["top_genre"],
    normalize="index"
).mul(100)
```

This ensures that the genre percentages for each cluster add up to 100%.

Output:

```text
top_genres_each_cluster.png
```

### 11. PCA visualization

The behavioral dataset contains six features, so it cannot be displayed directly in a two-dimensional graph. Principal Component Analysis reduces the standardized features to two principal components while preserving as much variation as possible.

The cluster label must not be included as an input feature for PCA. The labels from the final four-cluster model are added only after dimensionality reduction:

```python
pca_input = scaled_behaviour.drop(columns=["kmeans_4"], errors="ignore")

pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_input)

pca_df = pd.DataFrame(
    pca_result,
    columns=["PCA1", "PCA2"],
    index=df.index
)

pca_df["cluster_label"] = k_means.labels_
```

Each cluster is shown with a different color in the final scatter plot.

Output:

```text
pca_clusters.png
```

## Generated Visualizations

Running the program creates the following image files:

| File | Purpose |
| --- | --- |
| `top_users_with_daily_listening_minutes.png` | Shows the 20 listeners with the highest daily listening time |
| `first_250_users_and_users_older_than_45.png` | Shows the age distribution and highlights users older than 45 |
| `elbow_method.png` | Helps evaluate the number of clusters using inertia |
| `silhouette_curve.png` | Compares Silhouette Scores for different cluster counts |
| `cluster_percentages.png` | Shows the percentage of users in each cluster |
| `top_genres_each_cluster.png` | Shows the dominant genre associated with each cluster |
| `pca_clusters.png` | Displays the final clusters in two-dimensional PCA space |

## Installation

Install the required libraries with:

```bash
pip install pandas matplotlib scikit-learn
```

## How to Run

1. Place the Python program and `music_streaming_habits_2026.csv` in the same directory.
2. Open a terminal in that directory.
3. Run the program:

```bash
python music_example.py
```

Replace `music_example.py` with the actual name of the Python file if it is different.

The numerical results will be printed in the console, and the generated graphs will be saved in the project directory.

## Interpretation Notes

- K-Means finds patterns based only on the selected behavioral features.
- Cluster labels such as 0, 1, 2, and 3 have no predefined meaning.
- PCA is used for visualization and does not perfectly represent all six original dimensions.
- A visual overlap in the PCA plot does not necessarily mean that the original high-dimensional clusters are identical.
- The Elbow Method and Silhouette Score should be considered together when choosing the final number of clusters.
- The clusters describe associations in this dataset; they do not prove causal relationships.

## Conclusion

This project demonstrates a complete unsupervised machine-learning workflow for music-listener segmentation. It prepares numerical and Boolean behavioral data, evaluates possible cluster counts, trains a K-Means model, examines cluster sizes and behavioral profiles, relates clusters to preferred genres, and visualizes the final result using PCA.

The resulting listener groups provide a foundation for defining behavioral personas and can be extended into recommendation, engagement, or customer-retention analyses.
