# DBSCAN Clustering of Prague Airbnb Listings
This real-world project groups Airbnb listings in Prague according to latitude and longitude. The dataset is cleaned by selecting useful columns, removing missing coordinates, and converting price values into numerical form. 
Coordinates are converted to radians, and the Haversine distance is used with DBSCAN because the data represents real locations on Earth. 
GeoPandas and Contextily are used to display the clusters and noise points on a real Prague map. The project also calculates the average listing price for each cluster.
