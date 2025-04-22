import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from seizure_data_processing.config import SEIZE_IT_DIR
from preprocess_features import select_train_idx

# %% GLOBAL VARS
RATIO = 50
FILENAME = "features_frontal_preprocessed"
SAVE_FILE = f"PF_group_frontal_ratio_{RATIO}.parquet"

# %% Load the features

feature_file = SEIZE_IT_DIR + FILENAME + ".parquet"
save_file = SEIZE_IT_DIR + SAVE_FILE

features = pd.read_parquet(feature_file)
feat_cols = [col for col in features.columns if "|" in col]
features = features.drop(columns=feat_cols)
features = features.drop(columns=["train"])
if "combined_annotation" not in features.columns:
    features["combined_annotation"] = np.where(features.loc[:,"seiz_id"] > 0, 1, -1)
else:
    same = np.isclose(features["combined_annotation"], np.where(features.loc[:,"seiz_id"] > 0, 1, -1)).all()
    if not same:
        raise ValueError("Combined annotation is not the same as the seiz_id")
# feat_df['file_id'] = feat_df['filename'].str.extract(r'_r(\d+)\.edf').astype(int)

pat_info = pd.read_csv(SEIZE_IT_DIR + "data_info/patient_info.csv")
pat_keep = pat_info.loc[pat_info["No. seizures"] > 4, "patient"].values
features = features.loc[features["patient"].isin(pat_keep)]


for patient in features["patient"].unique():
    pat_df = features.loc[features["patient"] == patient].copy()
    pat_idx = pat_df["index"].to_numpy()
    for group in pat_df["group"].unique():
        group_df = pat_df.loc[pat_df["group"] == group].copy()
        group_idx = group_df['index'].to_numpy()
        group_df = select_train_idx(group_df, ratio=RATIO, bhe=False, method="random",ann_col="combined_annotation")
        features.loc[group_idx, "train"] = group_df["train"].copy()


new_df = features[["train", "test", "index", "patient", "seiz_id", "combined_annotation", "group"]]
new_df.to_parquet(save_file)

    # check on the
