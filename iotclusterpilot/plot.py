import datetime
import os

import matplotlib.pyplot as plt
import numpy as np



def radius_to_size(radius):
    return np.pi * (radius ** 2)


def plot_cluster(
        x_longitudes : list, 
        x_latitudes : list,
        centroid_longitudes : list,
        centroid_latitudes : list, 
        labels : list, 
        include_features : list, 
        feature_to_scale : list, 
        feature_scale : list, 
        n_clusters : int, 
        max_iter : int, 
        radius : int, 
        algorithm : str, 
        tol : float,
        ):

    plt.figure(figsize=(8, 9))  # Made figure slightly taller to accommodate bottom text
    plt.scatter(x_longitudes, x_latitudes, c=labels, cmap='viridis')
    plt.scatter(centroid_longitudes, centroid_latitudes, c='red', s=radius_to_size(radius), alpha=0.5)

    annotation_text = (
        f"include_features={include_features}\n"
        f"feature_to_scale={feature_to_scale}\n"
        f"feature_scale={feature_scale}\n"
        f"n_clusters={n_clusters}\n"
        f"algorithm={algorithm}\n"
        f"max_iter={max_iter}\n"
        f"radius={radius}\n"
        f"tol={tol}"
    )

    plt.subplots_adjust(bottom=0.3)  # Make room for text at bottom
    ax = plt.gca()
    ax.text(0.5, -0.2, annotation_text,
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    savedir = "./plots"

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    plot_path = f"./plots/cluster_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png"
    
    plt.savefig(plot_path)

    