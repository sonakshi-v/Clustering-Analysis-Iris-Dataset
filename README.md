# Clustering-Analysis-Iris-Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3)
kmeans_labels = kmeans.fit_predict(X_normalized)


dbscan = DBSCAN(eps=0.3, min_samples=5)  
dbscan_labels = dbscan.fit_predict(X_normalized)

agglomerative = AgglomerativeClustering(n_clusters=3)
agglomerative_labels = agglomerative.fit_predict(X_normalized)

metrics = {
    'K-Means': {
        'Silhouette Score': silhouette_score(X_normalized, kmeans_labels),
        'Davies-Bouldin Index': davies_bouldin_score(X_normalized, kmeans_labels),
        'Adjusted Rand Index': adjusted_rand_score(y, kmeans_labels)
    },
    'DBSCAN': {
        'Silhouette Score': silhouette_score(X_normalized, dbscan_labels),
        'Davies-Bouldin Index': davies_bouldin_score(X_normalized, dbscan_labels),
        'Adjusted Rand Index': adjusted_rand_score(y, dbscan_labels)
    },
    'Agglomerative': {
        'Silhouette Score': silhouette_score(X_normalized, agglomerative_labels),
        'Davies-Bouldin Index': davies_bouldin_score(X_normalized, agglomerative_labels),
        'Adjusted Rand Index': adjusted_rand_score(y, agglomerative_labels)
    }
}

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=kmeans_labels)
plt.title('K-Means Clustering')

plt.subplot(1, 3, 2)
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=dbscan_labels)
plt.title('DBSCAN Clustering')

plt.subplot(1, 3, 3)
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=agglomerative_labels)
plt.title('Agglomerative Clustering')
plt.show()
