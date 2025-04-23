"""
    Create a feature database in .parquet format of features extract from the EEG bhe channels of the seizIT dataset.
"""

import pandas as pd
import numpy as np
from seizure_data_processing import EEG
from seizure_data_processing.config import SEIZE_IT_DIR
from seizure_data_processing.pre_processing.features import extract_features
import os
from multiprocessing.pool import Pool
from functools import partial
import argparse


def extract_features_per_patient(
    file,
    channels,
    montage,
    filter_params,
    *,
    epoch_remove_channel=None,
    epoch_remove_method=None,
    asym_channels=None,
    window_time=2,
    seiz_overlap=0.5,
    bckg_overlap=0.5
):
    eeg = EEG(file, channels)
    eeg.load()
    eeg.annotate()
    eeg.apply_montage(montage)
    results = extract_features(
        eeg,
        filter_params,
        window_time=window_time,
        seiz_overlap=seiz_overlap,
        bckg_overlap=bckg_overlap,
        epoch_remove=True,
        min_amplitude=13,
        max_amplitude=150,
        epoch_remove_method=epoch_remove_method,
        asymmetry_channels=asym_channels,
    )
    return results


def get_cmd_input():
    parser = argparse.ArgumentParser(description="Extract features from EEG data")
    parser.add_argument(
        "--last_patient",
        type=str,
        default=None,
        help="Patient to start from. If not provided, start from the first patient.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=2.0,
        help="Window time in seconds for feature extraction.",
    )
    parser.add_argument(
        "--seiz_ovlp",
        type=float,
        default=0.5,
        help="Overlap of seizure windows as fraction of window_time.",
    )
    parser.add_argument(
        "--bckg_ovlp",
        type=float,
        default=0.5,
        help="Overlap of background windows as fraction of window_time.",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="features",
        help="Name of the file to save the features.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="bhe",
        help="Channels to extract features from.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_cmd_input()
    print(args)
    LAST_PATIENT = args.last_patient  # incase of interruption
    window_time = args.window
    seiz_overlap = args.seiz_ovlp
    bckg_overlap = args.bckg_ovlp

    # %% load pat_df to get the hemispheres
    pat_df = pd.read_csv(SEIZE_IT_DIR + "data_info/patient_info.csv")
    patients = pat_df["patient"].values

    # %% load the EEG data per patient
    # find index of the last patient
    if LAST_PATIENT in patients:
        idx_pat = np.where(patients == LAST_PATIENT)[0][0]
        patients = patients[idx_pat + 1 :]
        pat_df = pat_df.loc[pat_df["patients"].isin(patients), :]

    print(patients)
    for idx, pat_info in pat_df.iterrows():
        patient = pat_info["patient"]
        patient_folder = SEIZE_IT_DIR + patient + "/"
        print(patient_folder)
        # get .edf files in folder
        files = os.listdir(patient_folder)
        files = [patient_folder + f for f in files if ".edf" in f]

        # %% Select hemisphere
        if args.channels == "bhe":
            if pat_info["hemisphere"] == "L":
                channels = ["OorLiAchter", "OorLiTop", "OorReTop"]
                montage = [
                    "OorLiTop-OorLiAchter",
                    "OorLiTop-OorReTop",
                ]
            elif pat_info["hemisphere"] == "R":
                channels = ["OorReAchter", "OorLiTop", "OorReTop"]
                montage = [
                    "OorReTop-OorReAchter",
                    "OorLiTop-OorReTop",
                ]
            else:
                raise ValueError("Hemisphere not recognized")
        elif args.channels == "front":
            channels = ['F7', 'F8', 'T3', 'T4']
            montage = ['F7-T3', 'F8-T4', 'F7-F8', 'T3-T4']

        # %% extract features
        filter_params = {"min_freq": 1, "max_freq": 25, "notch_freq": None}
        with Pool() as p:
            results = p.map(
                partial(
                    extract_features_per_patient,
                    channels=channels,
                    montage=montage,
                    filter_params=filter_params,
                    epoch_remove_method="bhe",
                    asym_channels=None,
                    window_time=window_time,
                    seiz_overlap=seiz_overlap,
                    bckg_overlap=bckg_overlap,
                ),
                files,
            )

        # %% save features
        features = pd.concat(results)
        features.to_parquet(patient_folder + args.filename + ".parquet")
