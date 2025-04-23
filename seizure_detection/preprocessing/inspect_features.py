"""
Script for inspecting and visualization the feature (distributions).
Steps:
1. Load the features and annotations databases and aggregate them.
2. Plot the feature distributions.
3. Check distributions after transformation.

@author: Seline de Rooij
@email: s.j.s.derooij@tudelft.nl
@github: sderooij
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

from sklearn.preprocessing import PowerTransformer, MaxAbsScaler

from seizure_data_processing.config import SEIZE_IT_DIR

DELIM_FEAT_CHAN = "|"
TRANS_FEATURES = [
    "kurtosis",
    "line_length",
    # "max",
    # "min",
    # "nzc",
    "mean_power_alpha",
    "mean_power_beta",
    "mean_power_delta",
    "mean_power_theta",
    "mean_power_HF",
    "RMS_amplitude",
    "total_power",
]


if __name__ == "__main__":
    # %% load annotations database
    annotations = pd.read_csv(SEIZE_IT_DIR + "data_info/annotations.csv")
    # select patients with at least a seizure
    patients = annotations["patient"].unique()

    # %% load features per patient
    features = []
    for patient in patients:
        patient_folder = SEIZE_IT_DIR + patient + "/"
        # check if folder contains features.parquet file
        if "features.parquet" not in os.listdir(patient_folder):
            continue
        temp = pd.read_parquet(patient_folder + "features.parquet")
        temp["patient"] = patient
        features.append(temp)

    features = pd.concat(features, axis=0)
    # %% visualize features
    feat_cols = [col for col in features.columns if DELIM_FEAT_CHAN in col]

    # %% make hist plots of the features
    feat_names = [col.split(DELIM_FEAT_CHAN)[0] for col in feat_cols]
    feat_names = np.unique(feat_names)
    for feat in feat_names:
        feat_cols = [col for col in features.columns if feat in col]
        for col in feat_cols:
            col_label = col.split(DELIM_FEAT_CHAN)[1]
            features[col].hist(bins=100, label=col_label)
        plt.legend()
        # export to png file
        plt.savefig("fig/hist_no_transform/hist_" + feat + ".png")
        plt.close()

    # %% transform
    # select columns that contain substrings that are in TRANS_FEATURES
    trans_cols = [
        col for col in feat_cols if col.split(DELIM_FEAT_CHAN)[0] in TRANS_FEATURES
    ]
    pt = PowerTransformer(method="yeo-johnson")
    features.loc[:, trans_cols] = pt.fit_transform(features[trans_cols].to_numpy())

    # %% use max-abs scaler for all features
    scaler = MaxAbsScaler()
    features.loc[:, feat_cols] = scaler.fit_transform(features[feat_cols])

    # %% make hist plots of the features
    feat_names = [col.split(DELIM_FEAT_CHAN)[0] for col in feat_cols]
    feat_names = np.unique(feat_names)

    for feat in feat_names:
        feat_cols = [col for col in features.columns if feat in col]
        for col in feat_cols:
            col_label = col.split(DELIM_FEAT_CHAN)[1]
            features[col].hist(bins=100, label=col_label)
        plt.legend()
        # export to png file
        plt.savefig("fig/hist_after_transform/hist_" + feat + ".png")
        plt.close()
