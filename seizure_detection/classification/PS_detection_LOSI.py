#%%
from time import time
from datetime import datetime
import os
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from config import (
    TRACKING_URL,
)
import cloudpickle
import pandas as pd
import numpy as np
from classification import get_features, extract_feature_group_labels
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, cross_validate, LeavePGroupsOut
import json
# import mlflow_utils as mutils
import seizure_data_processing.classification.mlflow_utils as mutils
from seizure_data_processing.config import SEIZE_IT_DIR
import mlflow_load as mload
from copy import deepcopy
from seizure_data_processing.classification.classification import SeizureClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from tensorlibrary.learning.t_krr import CPKRR
from sklearn.preprocessing import MinMaxScaler

# %% SET PARENT ID of PI models
EXPERIMENT_NAME ="SEIZEIT_PS_CPKRR_LOSI"

RUN_NAME_PI = "CPKRR_PI_LOPO_1731508385"

# %% Set Global Variables
FEATURE_FILE = (
    SEIZE_IT_DIR + "features_preprocessed.parquet"
)  # .parquet file of the ann_df
GROUP_FILE = (
    SEIZE_IT_DIR + "PF_group_df.parquet"
)  # .parquet file of the validation groups
MODEL_TYPE = "PS"
GRID_SEARCH = "full"
CROSS_VAL_TYPE = "LOSI"
GRID_SEARCH_SCORING = "roc_auc"
DELIM_FEAT_CHAN = "|"
n_jobs = 3
SAVE_MLFLOW = True
SCALER="new"
REFIT_SCALER=False  # refit the scaler on the test set
old=True

# %% SET HYPERPARAMETERS
if "CPKRR" in RUN_NAME_PI:
    fixed_par = {
        "num_sweeps": 3,
        "train_loss_flag": False,
        "random_init": True,
        # "mu": 0,
        # "reg_par":0,
        # "mu": 1e-4
        # "max_iter": 10,
    }
    if MODEL_TYPE == "PF":
        # hyperpar = {
        #     "mu" : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        # }
        hyperpar = {"mu": [1e-3]}
        # fixed_par['mu'] = 1e-4
        fixed_par['reg_par'] = 0
    else:
        hyperpar = {
            "reg_par": [1e-5, 1e-4, 1e-3],
            "max_rank": [5, 10, 15]
        }
elif "SVC" in RUN_NAME_PI:
    fixed_par = {
    }
    hyperpar = {
        # "C": [0.1, 1, 10, 100],
    }

# %% Set tags
TAGS = {
    "dataset": "seize_it",
    "model_type": MODEL_TYPE,  #
    "cross_val_type": CROSS_VAL_TYPE,  # Leave one patient out
    "grid_search": GRID_SEARCH,
    "scaler": SCALER,
}

all_scores = {
    "AUC": "roc_auc",
    "Accuracy": "accuracy",
    "F1": "f1",
    "Precision": "precision",
    "Recall": "recall",
    "average_precision": "average_precision",
}
# %% Load data
pat_info = pd.read_csv(SEIZE_IT_DIR + "data_info/patient_info.csv")
pat_keep = pat_info.loc[pat_info["No. seizures"] > 4, "patient"].values

# %% Get the run id of the parent and the child runs
PARENT_ID_PI = mload.get_runid_from_run_name(TRACKING_URL, RUN_NAME_PI)

# %% search child runs of parent id
child_runs = mload.get_child_runs_from_parent(TRACKING_URL, PARENT_ID_PI)
# alpha numeric sort by patient
child_runs = child_runs.sort_values("patient")
child_runs = child_runs[child_runs["patient"].isin(pat_keep)]

# %% iterate over child runs (i.e. per patient)
for i, child_run in child_runs.iterrows():
    # %% Load the child run data
    model_file = "runs:/" + child_run["run_uuid"] + "/sk_model"
    patient = child_run["patient"]
    classifier_name = child_run["classifier"]

    #%% Load PI model for hyperparameters
    ml = mload.MLFlowLoader(TRACKING_URL, child_run["run_uuid"], local_project_drive='U:/secureseizuredata',
                            split_on="secureseizuredata")
    PI_model = ml.load_model('sk_model', model_type="sklearn")
    if old:  # to be compatible with the old code for CPKRR
        PI_model.named_steps["clf"].train_loss_flag = False
        PI_model.named_steps["clf"].loss = "l2"
        PI_model.named_steps["clf"].penalty = "l2"
        PI_model.named_steps["clf"].debug = False
        PI_model.named_steps["clf"].random_init = False

# %% set up LOSI model (and gridsearch)
    if TAGS["scaler"] == "new":
        scaler = MinMaxScaler((-0.5, 0.5))
    elif TAGS["scaler"] == "from-PI":
        PI_scaler = deepcopy(PI_model.named_steps["scaler"])  # deepcopy PI scaler
        scaler = FrozenEstimator(PI_scaler)  # Freeze the scaler object to prevent refitting

    clf = clone(PI_model.named_steps["clf"])    # clone the classifier, to copy hyperpar
    clf.set_params(**fixed_par)

    # %% Set up tags
    tags = TAGS.copy()
    tags["patient"] = patient
    tags["parent_run_id"] = child_run["run_uuid"]

    # %% Set up SeizureClassifier
    seiz_clf = SeizureClassifier(
        classifier=clf,
        hyperparams=hyperpar,
        model_type="PF",    # PF needed for crossvalidation setting (LOSI)
        preprocess_steps={"scaler": scaler},
        cv_obj=None,        # changed internally to LeavePGroupsOut, where P is num_seiz-1
        patient = patient,
        dataset = "seiz_it",
        feature_file = FEATURE_FILE,
        group_file = GROUP_FILE,
        grid_search = GRID_SEARCH,
        tags = tags,
        n_jobs = n_jobs,
        verbose = 1)

    # %% Fit the model
    seiz_clf.cross_validate(annotation_column='combined_annotation')
    # %% score the model
    predictions = seiz_clf.predict(FEATURE_FILE, GROUP_FILE, annotation_column='combined_annotation', mode='test',
                                   refit_scaler=REFIT_SCALER)

    #%% Score and print scores
    seiz_clf.score()
    seiz_clf.print_scores()
    if SAVE_MLFLOW:
        seiz_clf.log_mlflow(tracking_url=TRACKING_URL, experiment_name=EXPERIMENT_NAME, temp_dir="temp/",
                            save_cv_obj=False)
    else:
        # get unix time
        t = int(time())
        dt = datetime.fromtimestamp(t).strftime('%Y-%m-%d')
        # check if folder exists if not create it
        folder_name = SEIZE_IT_DIR + "models/" + dt
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # save model
        model_name = f"/CPKRR_PF_LOSI_{patient}.pickle"
        model_name = folder_name + model_name
        seiz_clf.save_local(model_name)

