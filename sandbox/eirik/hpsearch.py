import datetime
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna

from kmeans import CustomKMeans, load_signal_data, radius_to_size



def objective(trial):

    include_features = [
        "Longitude",
        "Latitude", 
        "Signal Strength (dBm)", 
        "Data Throughput (Mbps)",
        "Network Type",
        ]

    if trial.number == 0:
        print()
        print("#"*100)
        print("using features :", include_features)
        print("#"*100)
        print()
        time.sleep(2)

    feature_normalize = [0,1,2,3]
    
    feature_scale_longlat = trial.suggest_int("feature_scale_longlat", 5, 20)
    feature_scale_signal = trial.suggest_int("feature_scale_signal", 1, 5)
    feature_scale_data = trial.suggest_int("feature_scale_data", 1, 5)
    feature_scale_network = trial.suggest_int("feature_scale_network", 1, 5) 
    
    
    feature_plus = [0,1,2,3,4]
    
    feature_scale = [
        feature_scale_longlat, 
        feature_scale_longlat,
        feature_scale_signal,
        feature_scale_data, 
        feature_scale_network
        ]
    
    trial.set_user_attr("feature_scale", feature_scale)

    clusterdata, _ = load_signal_data(include_features)
    n_clusters = trial.suggest_int("n_clusters", 6, 8)
    max_iter = trial.suggest_int("max_iter", 100, 500)
    radius = trial.suggest_int("radius", 3, 7)

    kmeans = CustomKMeans(n_clusters=n_clusters, 
                          max_iter=max_iter, 
                          radius=radius,
                          feature_scale=feature_scale,
                          feature_plus=feature_plus,
                          feature_normalize=feature_normalize)
    

    kmeans.fit(clusterdata)

    labels = np.array([kmeans.predict_one(x.reshape(1,-1)) for x in clusterdata])

    is_outlier = np.where(labels == -1)[0].shape[0]

    opt_value = (is_outlier) * (radius)

    return opt_value



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    best_params = study.best_trials[0].params
    best_feature_scale = study.best_trials[0].user_attrs["feature_scale"]

    clusterdata, origdata = load_signal_data()
    
    kmeans = CustomKMeans(n_clusters=best_params["n_clusters"], 
                          max_iter=best_params["max_iter"], 
                          radius=best_params["radius"],
                          feature_scale=best_feature_scale,
                          feature_plus = [0,1,2,3,4],
                          feature_normalize = [0,1,2,3])
    
    kmeans.fit(clusterdata)

    labels = [kmeans.predict_one(x.reshape(1,-1)) for x in clusterdata]
    
    # plt.scatter(origdata["Longitude"], origdata["Latitude"], c=labels, cmap='viridis')
    # plt.scatter(kmeans.longlat_centroids[:, 0], kmeans.longlat_centroids[:, 1], c='red', s=50)

    # plt.title(f"n_clusters={best_params['n_clusters']}, max_iter={best_params['max_iter']}, radius={best_params['radius']}, feature_scale={best_feature_scale}")    

    # Create a figure with a specific size
    plt.figure(figsize=(10, 6))

    # Create the scatter plots
    plt.scatter(origdata["Longitude"], origdata["Latitude"], c=labels, cmap='viridis')
    plt.scatter(kmeans.longlat_centroids[:, 0], kmeans.longlat_centroids[:, 1], c='red', s=50)

    # Create the annotation text
    annotation_text = (f"n_clusters={best_params['n_clusters']}\n"
                    f"max_iter={best_params['max_iter']}\n"
                    f"radius={best_params['radius']}\n"
                    f"feature_scale={best_feature_scale}")

    # Adjust the subplot parameters to make room for the annotation
    plt.subplots_adjust(right=0.60)

    # Get the current axes
    ax = plt.gca()

    # Add text in axes coordinates (0 to 1)
    ax.text(1.05, 0.5, annotation_text,
            transform=ax.transAxes,  # This makes it use axis coordinates
            verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.savefig(f"./plots/hpsearch_{datetime.datetime.now()}.png")