import numpy as np
import pandas as pd
import os

from imblearn.under_sampling import RandomUnderSampler
import warnings

if __name__ == "__main__":
    from seizure_data_processing.config import SEIZE_IT_DIR

    print(SEIZE_IT_DIR)

    desired_ratio = 0.1  # ratio of seizure to background
    sample_group_size = 15 * 60  # 15 minutes
    # %% load annotations database
    annotations = pd.read_csv(SEIZE_IT_DIR + "data_info/annotations.csv")
    # select patients with at least a seizure
    patients = annotations["patient"].unique()

    idx = []
    # %% loop over the patients
    for patient in patients:
        patient_folder = SEIZE_IT_DIR + patient + "/"
        print(patient_folder)

        # load the features
        feat_df = pd.read_parquet(patient_folder + "features_preprocessed.parquet")
        # extract the groups and the annotations
        # feat_df = feat_df.hemisphere[:, ["index", "group", "annotation", "pre_ictal", "post_ictal", "filename"]]
        # feat_df.reset_index(inplace=True)
        # remove pre-ictal and post-ictal
        # feat_df = feat_df.hemisphere[~feat_df["pre_ictal"], :]
        # feat_df = feat_df.hemisphere[~feat_df["post_ictal"], :]
        # # drop the pre-ictal and post-ictal columns
        # feat_df = feat_df.drop(columns=["pre_ictal", "post_ictal"]

        # loop over the groups
        for group in feat_df["group"].unique():
            # select the group
            group_df = feat_df.loc[feat_df["group"] == group].copy()
            # num_files = group_df["filename"].nunique()
            # select positive and negative samples
            pos_df = group_df.loc[group_df["annotation"] == 1]
            # select only the test samples (to remove extra overlap)
            # pos_df = pos_df.hemisphere[feat_df['test'],:]
            neg_df = group_df.loc[group_df["annotation"] == -1]
            # drop pre and post-ictal data from neg_df to avoid sampling from those
            neg_df = neg_df.loc[~feat_df["pre_ictal"], :]
            neg_df = neg_df.loc[~feat_df["post_ictal"], :]
            # select the number of samples to keep in the negative samples
            n_samples = int(len(pos_df) / desired_ratio)
            if n_samples > len(neg_df):
                warnings.warn(
                    f"Not enough negative samples for patient {patient} group {group}"
                )
                n_samples = len(neg_df)

            # sample uniformly from the negative samples
            len_neg = len(neg_df)
            # idx_neg = neg_df["index"].tolist()
            # select every len_neg/n_samples samples
            # idx_neg = idx_neg[:: int(len_neg / n_samples)]

            # if len_neg % n_samples != 0:
            #     idx_neg = idx_neg[:-1]
            neg_df = neg_df.sample(n=n_samples, replace=False, ignore_index=False)
            idx_neg = neg_df["index"].tolist()

            # if any(len_files < n_per_file):
            #     n_per_files = [min(filelen, n_per_file) for filelen in len_files]
            #     fun = lambda x: x.sample(n=n_per_files, replace=False, ignore_index=False)
            #     neg_df = neg_df.groupby("filename")
            #     # iterate over the groups
            #
            # else:
            #     neg_df = neg_df.groupby("filename").apply(lambda x: x.sample(n=n_per_file, replace=False,
            #                                                                  ignore_index=False))
            #     neg_idx = neg_df["index"].tolist()
            # neg_df = neg_df.sample(n=n_samples, replace=False, ignore_index=False)
            idx.extend(pos_df["index"].tolist())
            idx.extend(idx_neg)

        # save the new dataframe
        # feat_df = feat_df.hemisphere[feat_df["index"].isin(idx), :]
        # feat_df.to_parquet(patient_folder + "train_features_preprocessed.parquet")
        feat_df["train"] = False
        feat_df.loc[feat_df["index"].isin(idx), "train"] = True
        feat_df.to_parquet(patient_folder + "features_preprocessed.parquet")
