import datetime
import math

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_data():

    df_raw = pd.read_csv("/home/eirik/data/signal_metrics.csv")

    # feature encoding
    df_raw["Network Type"] = df_raw["Network Type"].map({"3G": 0, "4G": 1, "5G": 2, "LTE": 3})

    # choose area
    area = "Fraser Road"
    origdata = df_raw[df_raw["Locality"] == area].reset_index(drop=True).reset_index(drop=False)

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
    def __init__(self, 
                 n_clusters : int=6, 
                 max_iter : int=100, 
                 radius : int=5, 
                 longlat_scale : int=5,
                 feature_plus : list=[0,1],
                 feature_scale : list=[0,1,2,3]):

        
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        self.radius = radius
        self.longlat_scale = longlat_scale
        self.feature_plus = feature_plus
        self.feature_scale = feature_scale
        self.centroids = None

    
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

        X = self.normalize(X)
        
        X[:, self.feature_plus] *= self.longlat_scale
        self.model.fit(X)
        
        self.centroids = self.model.cluster_centers_
        
        self.longlat_centroids = self.centroids[:, self.feature_plus]
        
        self.longlat_centroids /= self.longlat_scale

        # denormalize longitude and latitude features
        self.longlat_centroids[:, 0] = (self.longlat_centroids[:, 0] * (self.max_longitude - self.min_longitude)) + self.min_longitude
        self.longlat_centroids[:, 1] = (self.longlat_centroids[:, 1] * (self.max_latitude - self.min_latitude)) + self.min_latitude
    

    def predict_one(self, x):
                        
        assert self.centroids is not None, "Please call fit before predict."
        
        pred = self.model.predict(x)[0]
        x[:, self.feature_plus] /= self.longlat_scale

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
        
        cols = self.feature_scale
        
        X_maxs = np.max(X[:, cols], axis=0)
        X_mins = np.min(X[:, cols], axis=0)

        X[:, cols] = (X[:, cols] - X_mins) / (X_maxs - X_mins)

        return X


def radius_to_size(radius):
    return np.pi * (radius ** 2)


if __name__ == "__main__":

    clusterdata, origdata = load_data()
    
    kmeans = CustomKMeans(longlat_scale=20, 
                          radius=5)
    
    kmeans.fit(clusterdata)

    labels = [kmeans.predict_one(x.reshape(1,-1)) for x in clusterdata]
    
    plt.scatter(origdata["Longitude"], origdata["Latitude"], c=labels, cmap='viridis')
    plt.scatter(kmeans.longlat_centroids[:, 0], kmeans.longlat_centroids[:, 1], c='red', s=radius_to_size(40), alpha=0.3)

    plt.savefig(f"./plots/kmeans_{datetime.datetime.now()}.png")
