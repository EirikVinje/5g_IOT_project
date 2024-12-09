import argparse
import hdbscan
import pickle
import math

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans, OPTICS, SpectralClustering, AgglomerativeClustering, DBSCAN, MeanShift, AffinityPropagation, Birch, estimate_bandwidth
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.mixture import GaussianMixture
from kmeans import load_signal_data

def normalize(X):
    X_maxs = np.max(X, axis=0)
    X_mins = np.min(X, axis=0)

    X = (X - X_mins) / (X_maxs - X_mins)

    return X

def haversine_distance(lon1, lat1, lon2, lat2):

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        
        return c * r

def filter_clusters_by_radius(X, labels, max_radius, cluster_centers=None):
    max_radius = max_radius*10
    filtered_labels = labels.copy()
    
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    cluster_centers_new = [] if cluster_centers is None else None

    for cluster in unique_clusters:
        cluster_mask = labels == cluster
        cluster_points = X[cluster_mask]
        
        if cluster_centers is not None:
            cluster_center = cluster_centers[cluster]
        else:
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers_new.append(cluster_center)
        
        cluster_center_long = (cluster_center[0] * (max_longitude - min_longitude)) + min_longitude
        cluster_center_lat = (cluster_center[1] * (max_latitude - min_latitude)) + min_latitude
        for idx, point in enumerate(cluster_points):
            x_longitude = (point[0] * (max_longitude - min_longitude)) + min_longitude
            x_latitude = (point[1] * (max_latitude - min_latitude)) + min_latitude

            distance = haversine_distance(cluster_center_long, cluster_center_lat, x_longitude, x_latitude)

            print(x_longitude, x_latitude, cluster_center[0], cluster_center[1], distance)
            
            if distance > max_radius:
                filtered_labels[np.where(cluster_mask)[0][idx]] = -1
    
    if cluster_centers is not None:
        return filtered_labels
    else:
        return filtered_labels, cluster_centers_new


def test_kmeans(origdata, X, max_radius=10):
    clustering_kmeans = KMeans(n_clusters=6, random_state=42)

    clustering_kmeans.fit(X)
    cluster_centers = clustering_kmeans.cluster_centers_

    filtered_labels = filter_clusters_by_radius(X, clustering_kmeans.labels_, max_radius)

    plt.figure(figsize=(9, 9))
    plt.scatter(X[:, 0], X[:, 1], c=filtered_labels, s=30, cmap='viridis')
    # remove x and y axis numbers
    plt.xticks([])
    plt.yticks([])
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=250, linewidths=3, label='Centers')
    plt.title('Clusters identified by KMeans (max_radius = {})'.format(max_radius), fontsize=30)
    plt.show()

def test_optics(origdata, X, max_radius=10):
    clustering_optics = OPTICS(min_samples=50,max_eps=np.inf,metric='euclidean',cluster_method='xi',xi=0.001, n_jobs=-1)

    clustering_optics.fit(X)
    filtered_labels, cluster_centers = filter_clusters_by_radius(X, clustering_optics.labels_, max_radius)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=filtered_labels, s=50, cmap='viridis',marker = 'o')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title('Clusters identified by OPTICS (max radius = {})'.format(max_radius))
    plt.show()

def test_hdbscan(origdata, X, max_radius=10):
    clustering_hdbscan = hdbscan.HDBSCAN(min_cluster_size=10, 
                                         min_samples=10, 
                                         metric='euclidean', 
                                         cluster_selection_method='eom', 
                                         gen_min_span_tree=True)

    labels = clustering_hdbscan.fit_predict(X)
    filtered_labels, cluster_centers = filter_clusters_by_radius(X, labels, max_radius)


    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=filtered_labels, s=50, cmap='viridis',marker = 'o')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title('Clusters identified by HDBSCAN (max radius = {})'.format(max_radius))
    plt.show()

def test_spectral(origdata, X, max_radius=10):
    clustering_spectral = SpectralClustering(n_clusters=7, assign_labels='discretize', random_state=42)

    clustering_spectral.fit(X)
    filtered_labels, cluster_centers = filter_clusters_by_radius(X, clustering_spectral.labels_, max_radius)
    
    # Visualize the results
    plt.figure(figsize=(8, 6))
    plt.scatter(origdata[:, 0], origdata[:, 1], c=filtered_labels, s=50, cmap='viridis')
    #plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title('Clusters identified by Spectral Clustering (max radius = {})'.format(max_radius))
    plt.show()

def test_gmm(X):
    clustering_gmm = GaussianMixture(n_components=5, random_state=42)

    labels_gmm = clustering_gmm.fit_predict(X)
    cluster_centers = clustering_gmm.cluster_centers_

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, s=50, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title('Clusters identified by Gaussian Mixture Models')
    plt.show()

def test_agglo(origdata, X, max_radius=10):
    clustering_agglo = AgglomerativeClustering(n_clusters=6)

    clustering_agglo.fit(X)
    filtered_labels, cluster_centers = filter_clusters_by_radius(X, clustering_agglo.labels_, max_radius)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=filtered_labels, s=50, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title('Clusters identified by Agglomerative Clustering (max radius = {})'.format(max_radius))
    plt.show()

def test_meanshift(origdata, X, max_radius=10):
    bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=X.shape[0])

    clustering_mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    labels_mean_shift = clustering_mean_shift.fit_predict(X)
    cluster_centers = clustering_mean_shift.cluster_centers_
    filtered_labels = filter_clusters_by_radius(X, labels_mean_shift, max_radius, cluster_centers)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=filtered_labels, s=50, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title('Clusters identified by Mean Shift Clustering (max radius = {})'.format(max_radius))
    plt.show()

def test_affinity(X):
    clustering_affinity = AffinityPropagation(random_state=42, )

    labels_affinity = clustering_affinity.fit_predict(X)
    cluster_centers = clustering_affinity.cluster_centers_

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_affinity, s=50, cmap='viridis', marker='o')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title('Clusters identified by Affinity Propagation')
    plt.show()

def test_birch(origdata, X, max_radius=10):
    clustering_birch = Birch(n_clusters=6)

    labels_birch = clustering_birch.fit_predict(X)
    filtered_labels, cluster_centers = filter_clusters_by_radius(X, labels_birch, max_radius)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=filtered_labels, s=50, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title('Clusters identified by Birch Clustering (max radius = {})'.format(max_radius))
    plt.show()


if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--algo", type=str, default='optics', help='Algorithm to test, options: optics, hdbscan, spectral, gmm, agglo, meanshift, affinity, birch')
    argparser.add_argument("--radius", type=int, default=10, help='Radius for filtering clusters')
    args = argparser.parse_args()

    X, origdata = load_signal_data()
    origdata = X.copy()

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
    # load X_pca from pickle
    #X_pca = pickle.load(open('data/X_pca.pkl', 'rb'))
    #X_reduced = pickle.load(open('data/X_reduced.pkl', 'rb'))

    func = globals()['test_'+args.algo]
    func(origdata, X, max_radius=args.radius)
