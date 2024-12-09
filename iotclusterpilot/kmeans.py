import datetime
import math
import os

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_signal_data():
    
    cwd = os.getcwd()
    df_raw = pd.read_csv(f"{cwd}/data/signal_metrics.csv")

    # feature encoding
    df_raw["Network Type"] = df_raw["Network Type"].map({"3G": 0, "4G": 1, "5G": 2, "LTE": 3})
    # only keep those with 5G
    df_raw = df_raw[df_raw["Network Type"] == 2].reset_index(drop=True).reset_index(drop=False)
    # choose area
    # area = "Fraser Road"
    #origdata = df_raw[(df_raw["Locality"] == area)].reset_index(drop=True).reset_index(drop=False)
    
    # keep only those rows with the 10 most popular localities
    origdata = df_raw[df_raw["Locality"].isin(df_raw["Locality"].value_counts().head(10).index)].reset_index(drop=True).reset_index(drop=False)
    
    print(origdata["Locality"].value_counts())
    print('number of rows:', origdata.shape[0])
    
    origdata = origdata.rename(columns={'index': 'node'})

    features = ["Longitude",
                "Latitude", 
                "Network Type", 
                "Signal Strength (dBm)", 
                "Data Throughput (Mbps)"]

    origdata = origdata[features]
    clusterdata = origdata.copy()

    feature_order = ["Longitude",
                     "Latitude", 
                     "Signal Strength (dBm)", 
                     "Data Throughput (Mbps)",
                     "Network Type"]

    clusterdata = clusterdata[feature_order]

    clusterdata = clusterdata.to_numpy()

    return clusterdata, origdata


class CustomKMeans:
    def __init__(
            self,
            include_features : list=["Longitude", "Latitude", "Signal Strength (dBm)", "Data Throughput (Mbps)", "Network Type"],
            features_to_scale : list=["Longitude", "Latitude", "Network Type"],
            feature_scales : list=[10, 10, 4],
            n_clusters : int=6,
            max_iter : int=100, 
            radius : int=5,
            tol : float=0.0001,
            algorithm : str="lloyd",
    ):

        for feature in features_to_scale:
            if feature not in include_features:
                raise ValueError(f"feature to scale : {feature} : is not in include_features")

        if len(feature_scales) != len(features_to_scale):
            raise ValueError("feature_scales and features_to_scale must have same length")

        self.model = KMeans(n_clusters=n_clusters, 
                            max_iter=max_iter, 
                            random_state=42,
                            tol=tol,
                            algorithm=algorithm)
        
        
        self.radius = radius
        self.centroids = None

        self.include_features = include_features
        
        self.feature_to_scale = features_to_scale
        self.feature_scales = feature_scales
        
        self.features_to_normalize = ["Longitude", "Latitude", "Signal Strength (dBm)", "Data Throughput (Mbps)"]

        self.all_features_ordered = [
            "Longitude", 
            "Latitude", 
            "Signal Strength (dBm)", 
            "Data Throughput (Mbps)", 
            "Network Type"
            ]

    
    def haversine_distance(self, lon1, lat1, lon2, lat2):

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        
        return c * r


    def fit(self, X):
        
        self.max_longitude = X[:, 0].max()
        self.min_longitude = X[:, 0].min()
        self.max_latitude = X[:, 1].max()
        self.min_latitude = X[:, 1].min()

        feature_to_normalize_idx = [self.all_features_ordered.index(feature) for feature in self.features_to_normalize]

        X[:, feature_to_normalize_idx] = self.normalize(X[:, feature_to_normalize_idx])
        
        feature_to_scale_idx = [self.all_features_ordered.index(feature) for feature in self.feature_to_scale]

        X[:, feature_to_scale_idx] *= self.feature_scales
        
        features_to_include_idx = [self.all_features_ordered.index(feature) for feature in self.include_features]

        X = X[:, features_to_include_idx]
        
        self.model.fit(X)
        
        self.centroids = self.model.cluster_centers_
        
        self.longlat_centroids = self.centroids[:, [0,1]]
        
        self.longlat_centroids /= self.feature_scales[:2]

        # denormalize centroids for plotting
        self.longlat_centroids[:, 0] = (self.longlat_centroids[:, 0] * (self.max_longitude - self.min_longitude)) + self.min_longitude
        self.longlat_centroids[:, 1] = (self.longlat_centroids[:, 1] * (self.max_latitude - self.min_latitude)) + self.min_latitude
    

    def predict_one(self, x):
                        
        assert self.centroids is not None, "Please call fit before predict."
        
        pred = self.model.predict(x)[0]
        x[:, [0, 1]] /= self.feature_scales[:2]

        # denormalize longitude and latitude features
        x_longitude = (x[0][0] * (self.max_longitude - self.min_longitude)) + self.min_longitude
        x_latitude = (x[0][1] * (self.max_latitude - self.min_latitude)) + self.min_latitude
        
        # get centroid of predicted cluster
        centroid = self.longlat_centroids[pred, :]

        # calculate haversine distance between centroid and datapoint
        distance = self.haversine_distance(centroid[0], centroid[1], x_longitude, x_latitude)        
        
        if distance <= self.radius:
            return pred
        else:
            return -1


    def normalize(self, X):
        
        X_maxs = np.max(X, axis=0)
        X_mins = np.min(X, axis=0)

        X = (X - X_mins) / (X_maxs - X_mins)

        return X


if __name__ == "__main__":

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run this file from root of repository.")

    clusterdata, origdata = load_signal_data()
    radius=10
    n_clusters=6
    feature_scales=[10,10,1,1,1]
    features_to_scale=['Longitude','Latitude','Network Type', 'Signal Strength (dBm)', 'Data Throughput (Mbps)']
    
    kmeans = CustomKMeans(features_to_scale=features_to_scale, 
                          feature_scales=feature_scales, 
                          n_clusters=n_clusters, 
                          max_iter=100,
                          radius=radius)
    
    kmeans.fit(clusterdata)

    labels = np.array([kmeans.predict_one(x.reshape(1,-1)) for x in clusterdata])

    is_outlier = np.where(labels == -1)[0].shape[0]
    
    coverage_ratio = (len(labels) - is_outlier) / len(labels)
    print(f"Coverage ratio: {coverage_ratio}")
    
    plt.scatter(origdata["Longitude"], origdata["Latitude"], c=labels, cmap='viridis')
    plt.scatter(kmeans.longlat_centroids[:, 0], kmeans.longlat_centroids[:, 1], c='red', s=50)
    plt.title(f"Coverage ratio: {round(coverage_ratio,3)}")

    #plt.savefig(f"./plots/kmeans_{datetime.datetime.now()}.png")
    plt.savefig(f"./plots/kmeans_radius{radius}.png")
