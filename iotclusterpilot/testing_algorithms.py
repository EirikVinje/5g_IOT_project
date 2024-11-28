import numpy as np
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.cluster import KMeans, OPTICS, SpectralClustering, AgglomerativeClustering, DBSCAN, MeanShift, AffinityPropagation, Birch, estimate_bandwidth
from sklearn.mixture import GaussianMixture
import hdbscan
import matplotlib.pyplot as plt
from kmeans import load_signal_data


def normalize(X):
    X_maxs = np.max(X, axis=0)
    X_mins = np.min(X, axis=0)

    X = (X - X_mins) / (X_maxs - X_mins)

    return X

def test_optics(X):
    clustering_optics = OPTICS(min_samples=50,max_eps=np.inf,metric='euclidean',cluster_method='xi',xi=0.001, n_jobs=-1)

    clustering_optics.fit(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clustering_optics.labels_, s=50, cmap='viridis',marker = 'o')
    plt.title('Clusters identified by OPTICS-DBSCAN')
    plt.show()

def test_hdbscan(X):
    clustering_hdbscan = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)

    labels = clustering_hdbscan.fit_predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis',marker = 'o')
    plt.title('Clusters identified by HDBSCAN')
    plt.show()

def test_spectral(X):
    clustering_spectral = SpectralClustering(n_clusters=5, assign_labels='discretize', random_state=42)

    clustering_spectral.fit(X)

    # Visualize the results
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clustering_spectral.labels_, s=50, cmap='viridis')
    plt.title('Clusters identified by Spectral Clustering')
    plt.show()

def test_gmm(X):
    clustering_gmm = GaussianMixture(n_components=5, random_state=42)

    labels_gmm = clustering_gmm.fit_predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, s=50, cmap='viridis')
    plt.title('Clusters identified by Gaussian Mixture Models')
    plt.show()

def test_agglo(X):
    clustering_agglo = AgglomerativeClustering(n_clusters=6)

    clustering_agglo.fit(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clustering_agglo.labels_, s=50, cmap='viridis')
    plt.title('Clusters identified by Agglomerative Clustering')
    plt.show()

def test_meanshift(X):
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

    clustering_mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    labels_mean_shift = clustering_mean_shift.fit_predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_mean_shift, s=50, cmap='viridis')
    plt.title('Clusters identified by Mean Shift Clustering')
    plt.show()

def test_affinity(X):
    clustering_affinity = AffinityPropagation(random_state=42)

    labels_affinity = clustering_affinity.fit_predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_affinity, s=50, cmap='viridis', marker='x')
    plt.title('Clusters identified by Affinity Propagation')
    plt.show()

def test_birch(X):
    clustering_birch = Birch(n_clusters=6)

    labels_birch = clustering_birch.fit_predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_birch, s=50, cmap='viridis')
    plt.title('Clusters identified by Birch Clustering')
    plt.show()


if __name__=='__main__':
    X, origdata = load_signal_data()
    print(X)

    radius=10
    n_clusters=6

    all_features_ordered = [
                "Longitude", 
                "Latitude", 
                "Signal Strength (dBm)", 
                "Data Throughput (Mbps)", 
                "Network Type"
                ]
    features_to_normalize = ["Longitude", "Latitude", "Signal Strength (dBm)", "Data Throughput (Mbps)"]
    feature_to_scale = ['Longitude','Latitude','Network Type', 'Signal Strength (dBm)', 'Data Throughput (Mbps)']
    feature_scales=[10,10,1,1,1]


    max_longitude = X[:, 0].max()
    min_longitude = X[:, 0].min()
    max_latitude = X[:, 1].max()
    min_latitude = X[:, 1].min()

    feature_to_normalize_idx = [all_features_ordered.index(feature) for feature in features_to_normalize]

    X[:, feature_to_normalize_idx] = normalize(X[:, feature_to_normalize_idx])

    feature_to_scale_idx = [all_features_ordered.index(feature) for feature in feature_to_scale]

    X[:, feature_to_scale_idx] *= feature_scales

    test_optics(X)
    
