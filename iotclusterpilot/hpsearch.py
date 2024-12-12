import datetime
import argparse
import logging
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from plot_results import CustomKMeans
from kmeans import load_signal_data

logger = logging.getLogger('cluster hpsearch')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def set_feature_scales(features_to_scale : list, trial : optuna.Trial):

    feature_scales = []
    
    for feature in features_to_scale:
        
        if feature in ['Longitude','Latitude']:
            feature_scales.append(trial.suggest_int("longitude_latitude_scale", 1, 10))

        elif feature in ["Network Type", "Signal Strength (dBm)", "Data Throughput (Mbps)"]:
            feature_scales.append(trial.suggest_int(f"{feature}_scale", 1, 10))

        else:
            raise ValueError(f"feature : {feature} : not in include_features or features_to_scale")

    return feature_scales


def objective(trial):

    clusterdata, _ = load_signal_data(remove_features=["Network Type"])
    
    include_features = [
        "Longitude",
        "Latitude",
        "Signal Strength (dBm)", 
        "Data Throughput (Mbps)",
        ]
    
    features_to_scale = [
        "Longitude",
        "Latitude",
        # "Network Type",
        "Signal Strength (dBm)",
        "Data Throughput (Mbps)" 
        ]
    
    feature_scales = set_feature_scales(features_to_scale, trial)
    
    trial.set_user_attr("feature_to_scale", features_to_scale)
    trial.set_user_attr("include_features", include_features)
    trial.set_user_attr("feature_scale", feature_scales)
    n_clusters = trial.suggest_int("n_clusters", 6, 8)
    max_iter = trial.suggest_int("max_iter", 100, 500)
    radius = trial.suggest_float("radius", 2.0, 10.0)
    
    algorithm = trial.suggest_categorical("algorithm", ["lloyd", "elkan"])
    tol = trial.suggest_float("tol", 0.00001, 0.01, log=True)

    kmeans = CustomKMeans(n_clusters=n_clusters, 
                          max_iter=max_iter, 
                          radius=radius,
                          include_features=include_features,
                          features_to_normalize=include_features,
                          features_to_scale=features_to_scale,
                          all_features_ordered=include_features,
                          feature_scales=feature_scales,
                          algorithm=algorithm,
                          tol=tol)
    
    kmeans.fit(clusterdata)

    labels = np.array([kmeans.predict_one(x.reshape(1,-1)) for x in clusterdata])

    is_outlier = np.where(labels == -1)[0].shape[0]

    opt_value = 1.0 - (is_outlier / clusterdata.shape[0])

    opt_value = opt_value/(0.05*radius + 1)

    if trial.number != 0:
        print(f"------ current value and trial : ({np.round(opt_value, 3)}, {trial.number}) | best value and trial : ({np.round(study.best_trial.value, 3)}, {study.best_trial.number}) ------", end="\r")

    return opt_value


def radius_to_size(radius):
    return np.pi * (radius ** 2)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run this file from root of repository.")

    logger.info(f"Starting hyperparameter search with {args.n_trials} trials...")

    study = optuna.create_study(direction="maximize", study_name="kmeans")
    
    study.optimize(objective, n_trials=args.n_trials)
    print("", end="\n")
    logger.info(f"Finished hyperparameter search with {args.n_trials} trials")
    logger.info(f"Best trial and value : ({study.best_trial.number}, {study.best_trial.value})")
    
    logger.info("Running model with best trial params...")
    best_params = study.best_trials[0].params
    feature_scale = study.best_trials[0].user_attrs["feature_scale"]
    feature_to_scale = study.best_trials[0].user_attrs["feature_to_scale"]
    include_features = study.best_trials[0].user_attrs["include_features"]

    clusterdata, origdata = load_signal_data(remove_features=["Network Type"])
    
    kmeans = CustomKMeans(
        include_features=include_features,
        features_to_scale=feature_to_scale,
        feature_scales=feature_scale,
        n_clusters=best_params["n_clusters"],
        max_iter=best_params["max_iter"],
        radius=best_params["radius"],
        algorithm=best_params["algorithm"],
        tol=best_params["tol"],
        )
    
    kmeans.fit(clusterdata)
    labels = [kmeans.predict_one(x.reshape(1,-1)) for x in clusterdata]

    logger.info("Plotting results...")
    
    plt.figure(figsize=(8, 9))  # Made figure slightly taller to accommodate bottom text
    plt.scatter(origdata["Longitude"], origdata["Latitude"], c=labels, cmap='viridis')
    plt.scatter(kmeans.longlat_centroids[:, 0], kmeans.longlat_centroids[:, 1], c='red', s=radius_to_size(best_params["radius"]), alpha=0.5)

    annotation_text = (
        f"include_features={include_features}\n"
        f"feature_to_scale={feature_to_scale}\n"
        f"feature_scale={feature_scale}\n"
        f"n_clusters={best_params['n_clusters']}\n"
        f"max_iter={best_params['max_iter']}\n"
        f"radius={best_params['radius']}\n"
        f"algorithm={best_params['algorithm']}\n"
        f"tol={best_params['tol']}"
    )

    plt.subplots_adjust(bottom=0.3)  # Make room for text at bottom
    ax = plt.gca()
    ax.text(0.5, -0.2, annotation_text,
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plot_path = f"./plots/kmeans_hpsearch_{datetime.datetime.now()}.png"
    
    logger.info(f"Plot saved to : {plot_path}")

    plt.savefig(plot_path)

    