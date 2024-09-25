from typing import List

import pandas as pd


def load_data(features : List[str] = ["Locality", 
                                      "Latitude", 
                                      "Longitude", 
                                      "Network Type", 
                                      "Signal Strength (dBm)", 
                                      "Data Throughput (Mbps)"]):

    df = pd.read_csv("/home/eirik/data/signal_metrics.csv")
    df_pruned = df[features].reset_index(drop=True)
    df_pruned["Network Type"] = df_pruned["Network Type"].map({"3G": 0, "4G": 1, "5G": 2, "LTE": 3})

    return df_pruned