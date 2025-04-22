import json
from config import TRACKING_URL
import mlflow_load as mload
import numpy as np
import pandas as pd
from timescoring import scoring
from seizure_data_processing.config import SEIZE_IT_DIR
from scoring_functions import calculate_scores_per_group, aggregate_results
import mlflow
import yaml
import os
# extract patient ID from run_name P_IDXX using re
import re


# %% Get the model from the experiment
# run_names = ["SVC_PI_LOPO_1741957533"] #["CPKRR_PI_LOPO_1731508385"]#"SVC_PI_LOPO_1732882976"
# model_type = "PS"
SEIZ_OVERLAP = 0.5
BCKG_OVERLAP = 0.5
DES_OVERLAP = 0.5
SAMPLE_DURATION = 2
BIAS_SCORE = "sensitivity"
min_sensitivity = 0.375
OPTIM_BIAS = True
# CLASSIFIER = "SVM"
AGGR_MODE = "mean"    # 'cumulative' or 'mean', aggregate the results
fs = int(SAMPLE_DURATION * DES_OVERLAP)
MODEL_FILE = "LOSI_example.json"
# mu = 1e-1   # only s
OUTPUT_FILE = "output.parquet"
SAVE_DIR = "directory/to/save/results"
# folder_add = "low_PI"
folder_add = ""
# %% Load the run_ids and run_names from the .json file
with open(MODEL_FILE, 'r') as f:
    feeds = json.load(f)
run_names = feeds['RUN_NAME']
run_ids = feeds['RUN_ID']
# if run_names is a string, convert to list
if isinstance(run_names, str):
    run_names = [run_names]
MODEL_TYPE = feeds['MODEL_TYPE']
model_type = MODEL_TYPE
group_col = feeds['GROUP_COL']

if MODEL_TYPE in ["PF", "PS", "PF_mu", "PA"]:
    LOSI = True
else:
    LOSI = False
    group_col = 'group'

############### FULL SET OF PATIENTS
patients = [
    "P_ID13", "P_ID16", "P_ID33", "P_ID36", "P_ID61",
    "P_ID65", "P_ID70", "P_ID71", "P_ID76", "P_ID78",
    "P_ID79", "P_ID82", "P_ID83", "P_ID94", "P_ID99"
]

# %% PARAMTERs
parameters = scoring.EventScoring.Parameters(
            toleranceStart=30,
            toleranceEnd=60,
            minOverlap=0,
            maxEventDuration=5 * 60,
            minDurationBetweenEvents=90
        )   # parameters for the scoring function, use the defaults


mlflow.set_tracking_uri(TRACKING_URL)

if model_type in ["PF", "PS", "PF_mu", "PA"]:
    LOSI = True
    with open("files_per_group.json", "r") as f:
        files_per_group = json.load(f)
else:
    LOSI = False

# %% Loop over the runs
for run_id_parent in run_ids:
# def save_results(run_name):
    # %% Get the run id of the parent and the child runs
    # run_id_parent = mload.get_runid_from_run_name(TRACKING_URL, run_name)
    run_name = mload.get_run_name_from_runid(TRACKING_URL, run_id_parent)
    if LOSI:
        # In case of LOSI: create a mask for certain files in each group, since the splits between the validation
        # groups can be in a file. Thus, this part of the file should not be used for scoring.
        patient_id = re.search(r"P_ID\d+", run_name).group()
        group_to_estimator = {}     # to be compatible with the files_per_group.json
        run_ids_child = mload.get_child_runs_from_parent(TRACKING_URL, run_id_parent)
        for run_id_child in run_ids_child['run_uuid'].tolist():
            # hacky code to fix inconsistency in the group names (estimator: 0, 1, 2, ... vs seizure_id)
            data = mlflow.get_run(run_id_child).data
            estimator = int(data.tags['group'])  # change the 0, 1, 2, ... to the seizure_id in training
            seiz_id = re.findall(r'\d+',data.params['train'])[0] # extract seizure_id used in training
            group_to_estimator[seiz_id] = estimator # map the seizure id to the estimator id
        mask_files = files_per_group[patient_id]
        # rename the keys of the dictionary
        mask_files = {group_to_estimator[group]: mask_files[group] for group in mask_files.keys()}  # files
        # containing the training data start and stop times for each group
    else:
        mask_files = None

    # %% Load the outputs of the child runs
    ml = mload.MLFlowLoader(TRACKING_URL, run_id_parent)
    output = ml.load_artifact(OUTPUT_FILE)
    if model_type == "PI":
        output = output[output["group"].isin(patients)]

    # %% Get the file information
    file_df = pd.read_csv(SEIZE_IT_DIR + "data_info/file_info.csv")

    # %% Calculate the scores per validation group
    sample_results, event_results = calculate_scores_per_group(
        output,
        file_df,
        parameters,
        group_col=group_col,
        optim_bias=OPTIM_BIAS,
        bias_score=BIAS_SCORE,
        fs=fs,
        sample_duration=SAMPLE_DURATION,
        des_overlap=DES_OVERLAP,
        mask_per_group=mask_files,
        min_sensitivity=min_sensitivity,
    )

    # %% Aggregate the results, mode = 'cumulative' or 'mean'
    aggregated_sample_results, aggregated_event_results = aggregate_results(
        sample_results, event_results, mode=AGGR_MODE
    )

    # %% Save aggregated results
    if model_type == "PF" or model_type == "PS":
        # get name of model_file without extension
        model_name = MODEL_FILE.split("/")[-1].split(".")[0]
        if not os.path.exists(f"{SAVE_DIR}/{MODEL_TYPE}/{model_name}{folder_add}"):
            os.makedirs(f"{SAVE_DIR}/{MODEL_TYPE}/{model_name}{folder_add}")
        json_file = f"{SAVE_DIR}/{MODEL_TYPE}/{model_name}/{run_name}.json"
    else:
        json_file = f"{SAVE_DIR}/{MODEL_TYPE}/{run_name}.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "sample_results": aggregated_sample_results,
                "event_results": aggregated_event_results,
            },
            f,
            indent=2,
            sort_keys=False,
        )

    # %% SAVE per group in csv
    df = pd.DataFrame(columns=['sensitivity', 'precision', 'f1', 'fpRate'])
    for group in event_results.keys():
        df.loc[group] = [getattr(event_results[group], metric) for metric in ["sensitivity", "precision", "f1", "fpRate"]]
    if model_type == "PF" or model_type == "PS":
        df.to_csv(f"{SAVE_DIR}/{MODEL_TYPE}/{model_name}{folder_add}/{run_name}.csv")
    else:
        df.to_csv(f"{SAVE_DIR}/{MODEL_TYPE}/{run_name}.csv")


# %% Loop over the runs
