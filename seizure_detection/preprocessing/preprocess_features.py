"""
Script for preprocessing the features of the seize-it dataset.
Steps:
1. Load the features and annotations databases
2. Scale the features (note: look at dynamic normalization scheme)
3. Sort feature left and right, depending on the seizure side (note: how to deal with the case of both sides?)
4. Label each seizure with a seizure_id
5. Label the pre and post ictal periods.
6. Create a group dataframe with the validation groups (patients, seizures, fold splits)
7. Save the scaled and sorted features in a .parquet file with a column containing the validation groups.

@author: Seline de Rooij
@email: s.j.s.derooij@tudelft.nl
@github: sderooij
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.preprocessing import PowerTransformer, MaxAbsScaler, StandardScaler

from seizure_data_processing.config import SEIZE_IT_DIR
from seizure_data_processing.pre_processing.features import normalize_feature
from seizure_data_processing.post_processing.post_process import remove_overlap
from seizure_data_processing.datasets.seize_it import load_annotations

# GLOBAL VARS
PRE_ICTAL = 30
POST_ICTAL = 30
DELIM_FEAT_CHAN = "|"
TRANS_FEATURES = [
    "kurtosis",
    # "line_length",
    # "max",
    # "min",
    # "nzc",
    "mean_power_alpha",
    "mean_power_beta",
    "mean_power_delta",
    "mean_power_theta",
    "mean_power_HF",
    "normalized_power_alpha",
    "normalized_power_theta",
    "normalized_power_HF",
    "RMS_amplitude",
    "total_power",
    "shannon_entropy",
]  # features that need to be transformed using a power transformer (feature with distribution that is not Gaussian)

# NORM_FEATURES = [
#     "mean_power_alpha",
#     "mean_power_beta",
#     "mean_power_delta",
#     "mean_power_theta",
#     "mean_power_HF",
#     "RMS_amplitude",
#     "total_power"
# ]   # features to normalize using the median-decaying memory method


class PreProcessor:
    def __init__(
        self,
        patient: str,
        feat_file: str,
        pre_ictal: int = 30,
        post_ictal: int = 30,
        widow_time: float = 2.0,
        pos_overlap: float = 0.9,
        neg_overlap: float = 0.5,
        delim_feat_chan: str = "|",
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        self.patient = patient
        self.feat_file = feat_file
        self.pre_ictal = pre_ictal
        self.post_ictal = post_ictal
        self.widow_time = widow_time
        self.pos_overlap = pos_overlap
        self.neg_overlap = neg_overlap
        self.delim_feat_chan = delim_feat_chan
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.features = None

    def load_features(self):
        """
        load the features dataframe.
        """
        # check extension of the file
        if self.feat_file.endswith(".parquet"):
            self.features = pd.read_parquet(self.feat_file)
        elif self.feat_file.endswith(".csv"):
            self.features = pd.read_csv(self.feat_file)
        else:
            raise ValueError("File extension not supported. Use .parquet or .csv file.")

        self.features.reset_index(drop=True, inplace=True)
        self.features["index"] = self.features.index
        return self

    def preprocess(
        self,
        *,
        transform_method="yeo-johnson",
        trans_features=None,
        seiz_col="seiz_id",
        group_col="group",
        split_method="middle",
        ratio=5,
        ann_df=None,
    ):
        """
        Preprocess the features dataframe.
        """


        # %% transform features
        if transform_method is not None:
            self.features = transform_features(
                self.features, trans_features, method=transform_method
            )

        # %% Remove directory from filename
        self.features["filename"] = self.features["filename"].apply(
            lambda x: re.sub(r".*seize_it/", "", x)
        )

        # %% Give every seizure a unique ID
        # TODO: ensure that the features are correctly sorted
        self.features = label_seizures(self.features, self.patient, seiz_col=seiz_col, ann_df=ann_df)

        # %% add neurologists' annotations
        self.features = add_neurologist_annotations(self.features)

        # %% Label pre- and post-ictal periods
        self.features = add_pre_pos_ictal(
            self.features, pre_ictal=self.pre_ictal, post_ictal=self.post_ictal, ann_col='combined_annotation'
        )

        # %% Make splits for cross-validation
        self.features = split_seiz_groups(
            self.features,
            seiz_col=seiz_col,
            group_col=group_col,
            split_method=split_method,
        )

        # %% save test index (goes right because of reindexing), selecting all overlapping samples for testing purposes
        self.features = select_test_idx(
            self.features,
            pos_overlap=self.pos_overlap,
            neg_overlap=self.neg_overlap,
            desired_overlap=self.neg_overlap,
        )

        # TODO: select train idx
        self.features = select_train_idx(self.features, ratio=ratio)
        return self

    def rename_channels(self, mode='bhe', hemisphere='R'):
        if mode=='bhe':
            self.features = rename_bhe_channels(self.features, delim_feat_chan=self.delim_feat_chan)
        elif mode=='frontal':
            self.features = rename_frontal_channels(self.features, delim_feat_chan=self.delim_feat_chan,
                                                    hemisphere=hemisphere)
        return self

    def save(self, save_file):
        """
        Save the preprocessed features dataframe.
        """
        self.features.to_parquet(save_file)
        return self


def sort_features(features, channels=None):
    """
    Sort the features in the dataframe per feature name. The columns are sorted from low to high values.

    Args:
        features (pd.DataFrame): the features dataframe
        channels (list): list of channels to sort the features for

    Returns:
        pd.DataFrame: the sorted features dataframe
    """

    feat_cols = [col for col in features.columns if DELIM_FEAT_CHAN in col]
    if channels is not None:
        # select only the features of the channels
        feat_cols = [
            col for col in feat_cols if col.split(DELIM_FEAT_CHAN)[1] in channels
        ]

    feature_names = [col.split(DELIM_FEAT_CHAN)[0] for col in feat_cols]
    # remove duplicates
    feature_names = list(set(feature_names))

    # copy all columns except the feature columns
    new_feats = features.loc[
        :, [col for col in features.columns if col not in feat_cols]
    ]
    # group by feature name in column name
    for feat in feature_names:
        cols = [col for col in feat_cols if feat in col]
        feats = features.loc[:, cols].to_numpy()
        # sort columns from low to high values
        feats = np.sort(feats, axis=1)
        # create names for the new columns
        new_cols = [feat + DELIM_FEAT_CHAN + str(i) for i in range(feats.shape[1])]
        # add to dataframe
        new_feats.loc[:, new_cols] = feats

    return new_feats


def transform_features(
    features, feature_names="all", method="yeo-johnson", delim_feat_chan="|"
):
    """
    Transform the features using a power transformer. The features are transformed column-wise.

    Args:
        features (pd.DataFrame): the features dataframe
        feature_names (list): list of feature names to transform
        method (str): method of the power transformer
        delim_feat_chan (str): delimiter between feature name and channel name

    Returns:
        pd.DataFrame: the transformed features dataframe
    """

    # select columns that contain substrings that
    if feature_names != "all":
        trans_cols = [
            col
            for col in features.columns
            if col.split(delim_feat_chan)[0] in feature_names
        ]
    else:
        trans_cols = [col for col in features.columns if delim_feat_chan in col]

    transformer = PowerTransformer(method=method)
    features[trans_cols] = transformer.fit_transform(features[trans_cols].to_numpy())
    return features


def label_seizures(features, patient, *, seiz_col="seiz_id", ann_col='annotation', ann_df=None):
    """
    Give every seizure a unique ID. The seizure ID is created by concatenating the patient number with a number that is incremented for each seizure.

    Args:
        features (pd.DataFrame): the features dataframe
        patient (str): the patient number

    Returns:
        pd.DataFrame: the features dataframe with a column containing the seizure ID, "seiz_id".
    """
    if ann_df is None:
        # extract patient number
        pat_no = patient.replace("P_ID", "")
        features[seiz_col] = None  # create empty column
        # extract seizures
        seizures = features.loc[features[ann_col] == 1, :].copy()
        # check for gaps in index using diff
        gaps = seizures.loc[:, "epoch"].diff()
        no_seiz = sum(gaps != 1)  # number of seizures
        seiz_ids = [int(pat_no + str(s)) for s in range(1, no_seiz + 1)]
        seizures.loc[gaps != 1, seiz_col] = seiz_ids
        seizures.loc[:, seiz_col].fillna(method="ffill", inplace=True)

        # insert in features dataframe
        features.loc[features[ann_col] == 1, :] = seizures
    else:
        features[seiz_col] = None
        ann_df = ann_df.loc[ann_df["patient"] == patient, :]
        ann_df = ann_df.sort_values(by=["file_id", "start_time"])
        # replace None's with np.nan's in ann_df
        ann_df = ann_df.fillna(value=np.nan)
        features["bhe"] = None
        features["combined_annotation"] = -1
        for i, row in ann_df.iterrows():
            start_time = np.nanmin([row["start_time_neurologist"], row["start_time"]])
            stop_time = np.nanmax([row["stop_time_neurologist"], row['stop_time']])
            file = row["file"]
            seiz_id = row["seiz_id"]
            # insert in features dataframe
            seiz_indices = ((features["filename"] == file) &
                   (features["start_time"] >= start_time) &
                   (features["stop_time"] <= stop_time))
            features.loc[
                seiz_indices,
                seiz_col,
            ] = seiz_id
            features.loc[seiz_indices, "combined_annotation"] = 1
            features.loc[seiz_indices, "bhe"] = row["bhe"]

    return features


def add_neurologist_annotations(features):
    """
    Add the neurologists' annotations to the features dataframe.

    Args:
        features (pd.DataFrame): the features dataframe
        ann_files (list): list of filenames of the annotations

    Returns:
        pd.DataFrame: the features dataframe with a column containing the neurologists' annotations
    """
    # extract the annotations
    filenames = features["filename"].unique()
    # remove the part of de filenames before the seiz_it folder
    local_filenames = [re.sub(r".*seize_it/", "", file) for file in filenames]
    local_filenames = [SEIZE_IT_DIR + file for file in local_filenames]
    ann_files = [file.replace(".edf", "_a1.tsv") for file in local_filenames]
    features["neurologist_annotation"] = -1
    pos_idx = []
    for i, ann_file in enumerate(ann_files):
        ann_df = load_annotations(ann_file)
        feat = features.loc[features["filename"] == filenames[i], :]
        # check if the annotations are not empty
        if not ann_df.empty:
            for _, row in ann_df.iterrows():
                start = row["start_time"]
                stop = row["stop_time"]
                # check if the start and stop times are in the features
                if stop == "None" or stop is None:
                    stop = start + 10.0  # if no stop time given assume 10 seconds
                elif isinstance(stop, str):
                    stop = int(stop)

                idx = feat.loc[
                    (feat["start_time"] >= start) & (feat["stop_time"] <= stop),
                    "index",
                ].tolist()
                pos_idx.extend(idx)

    features.loc[pos_idx, "neurologist_annotation"] = 1

    return features


def add_pre_pos_ictal(features, pre_ictal=30, post_ictal=30, ann_col='annotation'):
    """
    Add pre and post ictal periods to the features dataframe.

    Args:
        features (pd.DataFrame): the features dataframe
    """
    features["pre_ictal"] = False
    features["post_ictal"] = False

    # get start and end of seizures
    seizures = (
        features[features[ann_col] == 1]
        .groupby("seiz_id")
        .agg({"start_time": ["min", "max"]})
    )

    for seiz_id, row in seizures.iterrows():
        start = row[("start_time", "min")]
        stop = row[("start_time", "max")]
        file = features.loc[features["seiz_id"] == seiz_id, "filename"].unique()[0]
        features.loc[
            (features["filename"] == file)
            & (features["start_time"] >= start - pre_ictal)
            & (features["start_time"] < start),
            "pre_ictal",
        ] = True
        features.loc[
            (features["filename"] == file)
            & (features["start_time"] > stop)
            & (features["start_time"] <= stop + post_ictal),
            "post_ictal",
        ] = True

    return features


def split_seiz_groups(
    features, seiz_col="seiz_id", group_col="group", split_method="middle"
):
    """

    :param features:
    :param group_col:
    :param split_method:
    :param overlap:
    :return:
    """
    # split in the middle of two seizures
    group_ids = features[seiz_col].dropna().unique()
    features[group_col] = None

    original_idx = features.index.copy()
    features = features.reset_index(drop=True)

    start = 0
    n_groups = len(group_ids)
    for i, group_id in enumerate(group_ids):
        # get last index of seizure
        cur_seiz_end = features.loc[features[seiz_col] == group_id, :].index.max()
        if i == n_groups - 1:
            stop = len(features) - 1
        else:
            next_seiz_start = features.loc[
                features[seiz_col] == group_ids[i + 1], :
            ].index.min()
            stop = int((cur_seiz_end + next_seiz_start) / 2)
        # check if epoch exists
        assert stop < len(features), f"stop index {stop} is out of range"
        # insert in features dataframe
        features.loc[start:stop, group_col] = group_id
        start = stop + 1

    features.index = original_idx
    return features


def select_test_idx(features, pos_overlap=0.9, neg_overlap=0.5, desired_overlap=0.5):
    """
    Select the indices of the training and test set based on the overlap of the annotations.

    Args:
        data (np.array): the annotations
        pos_overlap (float): the overlap of the positive annotations
        neg_overlap (float): the overlap of the negative annotations
        desired_overlap (float): the desired overlap of the annotations

    Returns:
        np.array: the indices of the test set
    """
    test_idx = remove_overlap(
        features["annotation"].to_numpy(),
        pos_overlap=pos_overlap,
        neg_overlap=neg_overlap,
        desired_overlap=desired_overlap,
    )
    features["test"] = False
    features.loc[test_idx, "test"] = True

    return features


def select_train_idx(features, ratio, method="Kaat", group_col=None, window_time=2.0, overlap=0.5, bhe=False, ann_col='annotation'):
    """
    Select the indices of the training set based on the ratio of the positive and negative samples.

    Args:
        features (pd.DataFrame): the features dataframe
        ratio (float): the ratio of the negative to positive class samples
        method (str): the method to use for selecting the training samples
        group_col (str): the column containing the group IDs, if None the samples are selected across the whole dataset

    Returns:
        np.array: the indices of the training set
    """
    features['train'] = False
    if bhe:
        features.loc[(features[ann_col]==1) & (features['bhe']==True), 'train'] = True
        des_num_neg = min(int(ratio * sum((features[ann_col] == 1) & (features['bhe']==True))), sum(features[ann_col]== -1))
    else:
        features.loc[features[ann_col]==1, 'train'] = True # set all positive samples to True
        des_num_neg = min(int(ratio * sum(features[ann_col] == 1)), sum((features[ann_col] == -1) & (features[
                                                                                                        'pre_ictal']==False) & (features['post_ictal']==False)))
    seg_time = int(window_time*overlap)
    idx_neg = features.loc[(features[ann_col]==-1) & (features['pre_ictal']==False) & (features['post_ictal']==False),
                                                                                       :].index.to_numpy()
    if method == "Kaat":
        #
        # reshape into 15 min windows first remove the last samples
        rem = len(idx_neg) % (seg_time*60)
        if rem != 0:
            idx_neg = idx_neg[:-rem]
        idx_neg = idx_neg.reshape((-1, seg_time*60))
        # select every 15th row of idx_neg if possible
        if len(idx_neg)/15 > des_num_neg:
            idx_neg = idx_neg[::15, :]

        # reshape back into 1D array
        idx_neg = idx_neg.flatten()
        # random sample
        idx_neg = np.random.choice(idx_neg, des_num_neg, replace=False)

    elif method == "random":
        idx_neg = np.random.choice(idx_neg, des_num_neg, replace=False)
    else:
        raise ValueError("Method not supported.")

    features.loc[idx_neg, 'train'] = True
    return features


def rename_bhe_channels(features, delim_feat_chan="|"):
    # Rename channels to 1 and 2
    feat_cols = [col for col in features.columns if delim_feat_chan in col]
    renamer = dict.fromkeys(feat_cols)
    for i, col in enumerate(feat_cols):
        if "OorLiTop-OorLiAchter" in col:
            renamer[col] = col.replace("OorLiTop-OorLiAchter", "Unilateral")
        elif "OorReTop-OorReAchter" in col:
            renamer[col] = col.replace("OorReTop-OorReAchter", "Unilateral")
        else:
            renamer[col] = col.replace("OorLiTop-OorReTop", "Crosshead")

    features.rename(columns=renamer, inplace=True)
    return features


def rename_frontal_channels(features, delim_feat_chan="|", hemisphere='R'):
    feat_cols = [col for col in features.columns if delim_feat_chan in col]
    renamer = dict.fromkeys(feat_cols)
    rm_cols = []
    for i, col in enumerate(feat_cols):
        if "F8-T4" in col and hemisphere== 'R':
            renamer[col] = col.replace("F8-T4", "Unilateral")
        elif "F7-T3" in col and hemisphere== 'L':
            renamer[col] = col.replace("F7-T3", "Unilateral")
        elif "T3-T4" in col:
            renamer[col] = col.replace("T3-T4", "Crosshead")
        else:
            renamer[col] = col
            rm_cols.append(col)

    features.rename(columns=renamer, inplace=True)
    features.drop(inplace=True, columns=rm_cols)
    return features


def main(
    patient,
    trans_features=None,
    pre_ictal=30,
    post_ictal=30,
    delim_feat_chan="|",
    seize_it_dir=SEIZE_IT_DIR,
    pos_overlap=0.9,
    neg_overlap=0.5,
    output=True,
    filename="features",
    ann_df=None,
    ratio=5,
    pat_info=None,
):
    """
    Main function for preprocessing the features of the seize-it dataset.
    """
    feature_file = seize_it_dir + patient + "/" + filename + ".parquet"

    PP = PreProcessor(
        patient,
        feature_file,
        pre_ictal=pre_ictal,
        post_ictal=post_ictal,
        delim_feat_chan=delim_feat_chan,
        pos_overlap=pos_overlap,
        neg_overlap=neg_overlap,
    )
    PP.load_features()
    if "front" in filename:
        hemi = pat_info.loc[pat_info["patient"] == patient, "hemisphere"].values[0]
        PP.rename_channels(mode='frontal', hemisphere=hemi)
    else:
        PP.rename_channels(mode='bhe')
    PP.preprocess(trans_features=trans_features, ann_df=ann_df, ratio=ratio)
    PP.save(seize_it_dir + patient + "/" + filename + "_preprocessed.parquet")
    if output:
        return PP.features
    else:
        return


if __name__ == "__main__":
    VERBOSE = True
    DEBUG = False
    N_JOBS = 4
    FILENAME="features_frontal"
    RATIO=50

    # %% load annotations database
    annotations = pd.read_csv(SEIZE_IT_DIR + "data_info/annotations.csv")
    # select patients with at least a seizure
    patients = annotations["patient"].unique()
    if "front" in FILENAME:
        pat_info = pd.read_csv(SEIZE_IT_DIR + "data_info/patient_info.csv")
    else:
        pat_info = None
    from functools import partial

    # create partial function
    pre_process = partial(
        main,
        trans_features=TRANS_FEATURES,
        pre_ictal=PRE_ICTAL,
        post_ictal=POST_ICTAL,
        delim_feat_chan=DELIM_FEAT_CHAN,
        seize_it_dir=SEIZE_IT_DIR,
        pos_overlap=0.9,
        neg_overlap=0.5,
        ratio=RATIO,
        ann_df=annotations,
        filename=FILENAME,
        pat_info=pat_info,
    )
    if DEBUG:
        feats = pre_process(patients[0])
    elif VERBOSE:
        from parallelbar import progress_map
        feats = progress_map(pre_process, patients, n_cpu=N_JOBS)
    else:
        from multiprocessing import Pool
        # create pool
        with Pool(N_JOBS) as p:
            feats = p.map(pre_process, patients)

    # concatenate features for PI model
    if VERBOSE:
        print("Concatenating features...")
    for idx, _ in enumerate(feats):
        feats[idx]["patient"] = patients[idx]
    features = pd.concat(feats, axis=0)
    features.reset_index(inplace=True, drop=True)
    features['index'] = features.index.tolist()
    features.to_parquet(SEIZE_IT_DIR + FILENAME + "_preprocessed.parquet")
    if VERBOSE:
        print("Done.")

