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
from svm_adapt import SVM_Adapt

# %% SET PARENT ID of PI models, CHANGE THIS
EXPERIMENT_NAME = "SEIZEIT_PF_CPKRR_LOSI"

RUN_NAME = "CPKRR_PI_LOPO_1731508385"

# %% Set Global Variables, CHANGE THIS
FEATURE_FILE = (
    SEIZE_IT_DIR + "features_preprocessed.parquet"
)  # .parquet file of the ann_df
GROUP_FILE = (
    SEIZE_IT_DIR + "PF_group_df.parquet"
)  # .parquet file of the validation groups
MODEL_TYPE = "PF"
GRID_SEARCH = "none"
CROSS_VAL_TYPE = "LOSI"
GRID_SEARCH_SCORING = "roc_auc"
DELIM_FEAT_CHAN = "|"
n_jobs = 3
RANDOM_INIT = False
SAVE_MLFLOW = True
SCALER="from-parent"
old=True

# %% load al_parameters from json file
# import os
#
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)
# with open("parameters_PF_CPKRR.json") as f:
#     parameters = json.load(f)
if "CPKRR" in RUN_NAME:
    fixed_par = {
        "num_sweeps": 1,
        "train_loss_flag": False,
        "random_init": RANDOM_INIT,
        # "mu": 0,
        # "reg_par":0,
        # "mu": 1e-4
        # "max_iter": 10,
    }
    if MODEL_TYPE == "PF":
        # hyperpar = {
        #     "mu": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        # }
        fixed_par['mu'] = 1e-3
        # hyperpar = {"mu": [1e-3]}
        # fixed_par['mu'] = 1e-4
        fixed_par['reg_par'] = 0
        hyperpar = None
    else:
        hyperpar = {
            "reg_par": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        }
elif "SVC" in RUN_NAME:
    fixed_par = {"gamma": "from-source", "class_weight": "balanced", "verbose": 0, "C": 10}
    hyperpar = None
    # hyperpar = {
    #     "C": [10],
    # }

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

# pat_rm = ["P_ID13"]
# %% Get the run id of the parent and the child runs
PARENT_ID_PI = mload.get_runid_from_run_name(TRACKING_URL, RUN_NAME)

# %% search child runs of parent id
child_runs = mload.get_child_runs_from_parent(TRACKING_URL, PARENT_ID_PI)
# alpha numeric sort by patient
child_runs = child_runs.sort_values("patient")
child_runs = child_runs[child_runs["patient"].isin(pat_keep)]

# %% iterate over child runs (i.e. per patient)
for i, child_run in child_runs.iterrows():
    model_file = "runs:/" + child_run["run_uuid"] + "/sk_model"
    patient = child_run["patient"]
    classifier_name = child_run["classifier"]
    print("Adapting model for patient: ", patient)

    # %% Load PI model
    ml = mload.MLFlowLoader(TRACKING_URL, child_run["run_uuid"], local_project_drive='U:/secureseizuredata',
                            split_on="secureseizuredata")
    PI_model = ml.load_model('sk_model', model_type="sklearn")
    if old and "CPKRR" in RUN_NAME:
        PI_model.named_steps["clf"].train_loss_flag = False
        PI_model.named_steps["clf"].loss = "l2"
        PI_model.named_steps["clf"].penalty = "l2"
        PI_model.named_steps["clf"].debug = False
        PI_model.named_steps["clf"].random_init = False

    # %% set up LOSI model (and gridsearch)
    if TAGS["scaler"] == "fine-tuned":
        scaler = clone(PI_model.named_steps["scaler"])
    else:
        scaler = FrozenEstimator(deepcopy(PI_model.named_steps["scaler"]))  # Freeze the scaler object to prevent refitting

    if "CPKRR" in RUN_NAME:
        clf = deepcopy(PI_model.named_steps["clf"])    #TODO: clone or deepcopy?, deepcopy for now
        clf.set_params(**fixed_par)
        clf.set_params(w_init=deepcopy(PI_model["clf"].weights_))
    elif "SVC" in RUN_NAME:
        source_model = FrozenEstimator(deepcopy(PI_model.named_steps["clf"]))
        clf = SVM_Adapt(source_model=source_model, **fixed_par)
    else:
        raise ValueError("Model type not recognized, must be CPKRR or SVC")

    # %% Set up tags
    tags = TAGS.copy()
    tags["patient"] = patient
    tags["parent_run_id"] = child_run["run_uuid"]

    # %% Set up SeizureClassifier
    seiz_clf = SeizureClassifier(
        classifier=clf,
        hyperparams=hyperpar,
        model_type=MODEL_TYPE,
        preprocess_steps={"scaler": scaler},
        cv_obj=None,        # changed internally to LeavePGroupsOut, where P is num_seiz-1
        patient = patient,
        dataset = "seiz_it",
        feature_file = FEATURE_FILE,
        group_file = GROUP_FILE,
        grid_search = GRID_SEARCH,
        grid_search_scoring= GRID_SEARCH_SCORING,
        tags = tags,
        n_jobs = n_jobs,
        verbose = 1,
        k_folds = 5)

    # %% Fit the model
    seiz_clf.cross_validate(annotation_column='combined_annotation')
    # %% score the model
    predictions = seiz_clf.predict(FEATURE_FILE, GROUP_FILE, annotation_column='combined_annotation', mode='test')
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
        if "CPKRR" in RUN_NAME:
            model_name = f"/CPKRR_PF_LOSI_{patient}_{t}.pickle"
        elif "SVC" in RUN_NAME:
            model_name = f"/SVC_PF_LOSI_{patient}_{t}.pickle"
        model_name = folder_name + model_name
        seiz_clf.save_local(model_name)

    print(f"Successfully adapted model and saved, to next patient...")

print("Success!")

