# %% Imports
import numpy as np
import pandas as pd
import os

# internal imports
from seizure_data_processing import EEG
from seizure_data_processing.datasets.seize_it import load_annotations
from seizure_data_processing.config import SEIZE_IT_DIR

SAVE_FOLDER = SEIZE_IT_DIR + "data_info/"

# %% Get all folders in SEIZE_IT_DIR
folders = os.listdir(SEIZE_IT_DIR)
folders = [f for f in folders if "P_ID" in f]

# get all .edf files in the folders
edf_files = []
for folder in folders:
    files = os.listdir(SEIZE_IT_DIR + folder)
    files = [folder + "/" + f for f in files if ".edf" in f]
    edf_files.extend(files)

# %% Create annotation database
# initialize
annotations = []
file_info = []

# loop over all edf files
for file in edf_files:
    eeg = EEG(SEIZE_IT_DIR + file)
    patient = os.path.basename(os.path.dirname(file))
    # get annotations
    eeg.annotate(version="a2")
    # append to annotations
    ann = eeg.annotations
    ann_a1 = load_annotations(SEIZE_IT_DIR + file, version="a1")

    if len(ann) > 0:
        ann["start_time_neurologist"] = ann_a1["start_time"]
        ann["stop_time_neurologist"] = ann_a1["stop_time"]
        ann["file"] = file
        ann["patient"] = patient
        annotations.append(eeg.annotations)
    # append file info
    duration = eeg.get_file_duration()
    channels = eeg.get_channels()
    fs = eeg.get_sampling_frequency()
    file_info.append([file, patient, duration, channels, fs])

# %% Create dataframes to save
ann_df = pd.concat(annotations)
ann_df = ann_df.reset_index(drop=True)
ann_df[["hemisphere", "origin", "bhe"]] = ann_df["comments"].str.split(",", expand=True)
ann_df["hemisphere"] = ann_df["hemisphere"].str.replace("Hem:", "")
ann_df["origin"] = ann_df["origin"].str.replace("Orig:", "")
ann_df["bhe"] = ann_df["bhe"].str.replace("bhe:", "")
ann_df["seizure_duration"] = ann_df["stop_time"] - ann_df["start_time"]
ann_df["seizure_duration_neurologist"] = (
    ann_df["stop_time_neurologist"] - ann_df["start_time_neurologist"]
)
ann_df.rename(columns={"annotation": "seizure_type"}, inplace=True)
ann_df.drop(columns=["comments"], inplace=True)
# add file id
ann_df["file_id"] = ann_df["file"].str.extract(r"_r(\d+)\.edf").astype(int)

# %% add seizure id
ann_df = ann_df.sort_values(by=["patient", "file_id", "start_time"])
ann_df["seiz_id"] = np.nan
for patient in ann_df["patient"].unique():
    pat_ann = ann_df[ann_df["patient"] == patient]
    pat_ann = pat_ann.sort_values(by=["file_id","start_time"])
    pat_no = patient.replace("P_ID", "")
    seiz_ids = [int(pat_no + str(s)) for s in range(1, len(pat_ann) + 1)]
    ann_df.loc[ann_df["patient"] == patient, "seiz_id"] = seiz_ids

ann_df.to_csv(SAVE_FOLDER + "annotations.csv", index=False)

file_info_df = pd.DataFrame(
    file_info, columns=["file", "patient", "file_duration", "channels", "fs"]
)
file_info_df.to_csv(SAVE_FOLDER + "file_info.csv", index=False)

# %% get patient info from first .tsv file in each folder
patient_info = []
for folder in folders:
    files = os.listdir(SEIZE_IT_DIR + folder)
    files = [folder + "/" + f for f in files if ".tsv" in f]
    tsv_file = SEIZE_IT_DIR + files[0]
    patient = folder
    with open(tsv_file, "r") as f:
        lines = f.readlines()
        gender = lines[3]
        gender = gender.split(":")[1].strip().rstrip("/n")
        age = lines[4]
        age = int(age.split(":")[1].strip().rstrip("/n"))
        # age = age.strip()
    patient_info.append([patient, gender, age])

# %% Create dataframe to save
pat_df = pd.DataFrame(patient_info, columns=["patient", "gender", "age"])
pat_df["No. seizures"] = pat_df["patient"].map(ann_df["patient"].value_counts())
# %% get hemisphere of patient
for patient in pat_df["patient"]:
    hem_count = ann_df[ann_df["patient"] == patient]["hemisphere"].value_counts()
    hem = hem_count.idxmax()
    if hem == "bi":
        hem_count.drop("bi", inplace=True)
        try:
            hem = hem_count.idxmax()
        except:
            hem = "L"
    elif hem == "NC":
        hem = "L"

    pat_df.loc[pat_df["patient"] == patient, "hemisphere"] = hem

pat_df.to_csv(SAVE_FOLDER + "patient_info.csv", index=False)


# %% TODO seizure count where bhe = 1
