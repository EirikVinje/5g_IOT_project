import os
import math
import datetime
import argparse
import tikzplotly
import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from kmeans import load_signal_data

class CustomKMeans:
    def __init__(
            self,
            include_features : list=["Longitude", "Latitude", "Signal Strength (dBm)", "Data Throughput (Mbps)", "Network Type"],
            features_to_normalize : list=["Longitude", "Latitude", "Signal Strength (dBm)", "Data Throughput (Mbps)"],
            features_to_scale : list=["Longitude", "Latitude", "Network Type"],
            all_features_ordered : list=["Longitude", "Latitude", "Signal Strength (dBm)", "Data Throughput (Mbps)", "Network Type"],
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
        
        self.features_to_normalize = features_to_normalize

        self.all_features_ordered = all_features_ordered

    
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

def radius_to_size(radius):
    return np.pi * (radius ** 2)

def plot_single(clusterdata, origdata, radius, n_clusters, plot=True):
    clusterdata, origdata = load_signal_data()

    include_features = ["Longitude", "Latitude", "Signal Strength (dBm)", "Data Throughput (Mbps)", "Network Type"]
    features_to_scale=['Longitude','Latitude', 'Signal Strength (dBm)', 'Data Throughput (Mbps)']

    all_features_ordered = ["Longitude", "Latitude", "Signal Strength (dBm)", "Data Throughput (Mbps)", "Network Type"]
    feature_scales=[10,10,1,1]
    
    kmeans = CustomKMeans(include_features=include_features,
                          features_to_scale=features_to_scale, 
                          feature_scales=feature_scales, 
                          n_clusters=n_clusters, 
                          max_iter=100,
                          radius=radius)
    
    kmeans.fit(clusterdata)

    labels = np.array([kmeans.predict_one(x.reshape(1,-1)) for x in clusterdata])

    is_outlier = np.where(labels == -1)[0].shape[0]
    
    coverage_ratio = (len(labels) - is_outlier) / len(labels)

    if plot:
        outliers = labels==-1
        inliers = labels!=-1

        plt.figure(figsize=(9, 9))  # Made figure slightly taller to accommodate bottom text

        #plt.scatter(origdata["Longitude"], origdata["Latitude"], c=labels, cmap='viridis')

        plt.scatter(origdata["Longitude"][inliers], origdata["Latitude"][inliers], c=labels[inliers], cmap='viridis', s=15, alpha=0.6)

        plt.scatter(origdata["Longitude"][outliers], origdata["Latitude"][outliers], color='red', s=15, alpha=0.6, label='Outliers')


        for center in kmeans.longlat_centroids:
            #print(center)
            #circle1 = plt.Circle((center[0], center[1]), radius/(120*math.cos(math.radians(center[1]))), color='red', alpha=0.2)
            circle = matplotlib.patches.Ellipse(xy=(center[0], center[1]), width=radius/(111.320*math.cos(math.radians(center[1])))*2, height=2*(radius/110.574), color='grey', alpha=0.2)
            
            plt.gca().add_patch(circle)
            #plt.gca().add_patch(circle1)
            #print(circle)

        plt.scatter(kmeans.longlat_centroids[:, 0], kmeans.longlat_centroids[:, 1], c='blue', s=20, marker="x")  # Centroid markers

        plt.title(f"Custom K-Means clustering (Coverage ratio: {round(coverage_ratio, 3)})", fontsize=20)
        plt.xlabel("Longitude", fontsize=15, labelpad=10)
        plt.ylabel("Latitude", fontsize=15, labelpad=10)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        '''# Annotation for feature parameters
        annotation_text = (
            f"include_features={include_features}\n"
            f"feature_to_scale={features_to_scale}\n"
            f"feature_scale={feature_scales}\n"
            f"n_clusters={n_clusters}\n"
            f"radius={radius}\n"
        )

        plt.subplots_adjust(bottom=0.3)
        ax = plt.gca()
        ax.text(0.5, -0.2, annotation_text, transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))'''

        plot_path = f"./plots/kmeans_radius{radius}_n-clusters{n_clusters}_features{len(include_features)}.png"
        plt.savefig(plot_path)
    return coverage_ratio

def run_experiment(variant="radius"):
    clusterdata, origdata = load_signal_data()
    if variant == "radius":
        # change from plt to plotly go
        fig = go.Figure()
        for n_clusters in range(3, 9):
            coverage_ratios = []
            for radius in range(1, 11):
                coverage_ratio = plot_single(clusterdata, origdata, radius, n_clusters, plot=False)
                coverage_ratios.append((radius, n_clusters, coverage_ratio))


            #plt.plot([x[0] for x in coverage_ratios], [x[2] for x in coverage_ratios], label=f"n_clusters={n_clusters}")
            fig.add_trace(go.Scatter(x=[x[0] for x in coverage_ratios], y=[x[2] for x in coverage_ratios], mode='lines', name=f"n_clusters={n_clusters}"))
        
        ''' plt.xlabel("Radius")
        plt.ylabel("Coverage ratio")
        plt.title("Adjusting radius for different number of clusters")
        plt.legend()
        plt.savefig("./plots/adjusting_radius.png")'''
        fig.update_layout(title="Adjusting radius for different number of clusters", xaxis_title="Radius", yaxis_title="Coverage ratio")
        fig.write_image("./plots/adjusting_radius.png", width=800, height=600)
        tikzplotly.save("adjusting_radius.tex", fig)

    elif variant == "n_clusters":
        # change from plt to plotly go
        fig = go.Figure()
        for radius in range(5, 11):
            coverage_ratios = []
            for n_clusters in range(1, 11):
                coverage_ratio = plot_single(clusterdata, origdata, radius, n_clusters, plot=False)
                coverage_ratios.append((radius, n_clusters, coverage_ratio))


            #plt.plot([x[0] for x in coverage_ratios], [x[2] for x in coverage_ratios], label=f"n_clusters={n_clusters}")
            fig.add_trace(go.Scatter(x=[x[1] for x in coverage_ratios], y=[x[2] for x in coverage_ratios], mode='lines', name=f"radius={radius}"))
        
        ''' plt.xlabel("Radius")
        plt.ylabel("Coverage ratio")
        plt.title("Adjusting radius for different number of clusters")
        plt.legend()
        plt.savefig("./plots/adjusting_radius.png")'''
        fig.update_layout(title="Adjusting n_clusters for different number of radiuses", xaxis_title="n_clusters", yaxis_title="Coverage ratio")
        fig.write_image("./plots/adjusting_n_clusters.png", width=800, height=600)
        tikzplotly.save("adjusting_n_clusters.tex", fig)

    return None

if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--radius", type=int, default=6)
    argparser.add_argument("--n_clusters", type=int, default=6)
    argparser.add_argument("--variant", type=str, default="radius")

    args = argparser.parse_args()

    clusterdata, origdata = load_signal_data()
    cov_ratio = plot_single(clusterdata, origdata, args.radius, args.n_clusters)
    print(f"Coverage ratio: {cov_ratio}")

    run_experiment(variant=args.variant)
    