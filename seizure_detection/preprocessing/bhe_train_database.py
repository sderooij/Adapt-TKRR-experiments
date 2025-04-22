import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from seizure_data_processing.config import SEIZE_IT_DIR
from preprocess_features import select_train_idx

if __name__ == "__main__":
    feature_file= SEIZE_IT_DIR + "features_preprocessed.parquet"
    features = pd.read_parquet(feature_file)
    # features["combined_annotation"] = np.where(features.hemisphere[:, "seiz_id"] > 0, 1, -1)
    # data_info = pd.read_csv(SEIZE_IT_DIR + "data_info/annotations.csv", usecols=["seiz_id", "bhe"])
    # features = features.merge(data_info, on="seiz_id", how="left")
    # features["bhe"] = features["bhe"].fillna(0)
    # features.to_parquet(feature_file)


    # delete train and test col
    features = features.drop(columns=["train"])
    bhe = True
    for patient in features["patient"].unique():
        # print(f"Patient {patient}")
        # if patient == "P_ID70":
        #     pass
        pat_df = features.loc[features["patient"] == patient].copy()
        pat_idx = pat_df.index.to_numpy()
        pat_df = select_train_idx(pat_df, ratio=50, bhe=bhe, method="Kaat", window_time=2, overlap=0.5,
                                  ann_col="combined_annotation")
        features.loc[pat_idx, "train"] = pat_df["train"]

    # drop all columns except train, test, index, patient and seiz_id
    group_df = features[["train", "test", "index", "patient", "seiz_id", "combined_annotation"]]

    # check ratio's
    # for patient in group_df["patient"].unique():
    #     patient_df = group_df[group_df["patient"] == patient]
    #     patient_df = patient_df.hemisphere[patient_df['train']]
    #     ratio = patient_df["combined_annotation"].value_counts()
    #     # ratio = patient_df["train"].sum() / patient_df["test"].sum()
    #     print(f"Patient {patient} has a ratio of {ratio}")
    # save group_df
    group_df.to_parquet(SEIZE_IT_DIR + "bhe_group_df_large.parquet")