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
import json
from seizure_data_processing.config import SEIZE_IT_DIR
import mlflow_load as mload
from copy import deepcopy
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from tensorlibrary.learning.t_krr import CPKRR
from svm_adapt import SVM_Adapt
import matplotlib.pyplot as plt


# %% Set Global Variables
FEATURE_FILE = (
    SEIZE_IT_DIR + "features_preprocessed.parquet"
)  # .parquet file of the ann_df
GROUP_FILE = (
    SEIZE_IT_DIR + "PF_group_df.parquet"
)  # .parquet file of the validation groups
MODEL_FILE = "LOSI_example.json"
MODEL_TYPE = "PF"
GRID_SEARCH = "on-test"
CROSS_VAL_TYPE = "LOSI"
DELIM_FEAT_CHAN = "|"
n_jobs = 1
RANDOM_INIT = False
SAVE_MLFLOW = False
SCALER="from-parent"
old=True

# %% load model id from json file
with open(MODEL_FILE, 'r') as f:
    feeds = json.load(f)
    RUN_NAME = feeds['RUN_NAME']
    RUN_ID = feeds['RUN_ID']

# %% Set tags
TAGS = {
    "dataset": "seize_it",
    "model_type": MODEL_TYPE,  #
    "scaler": SCALER,
}
#
# all_scores = {
#     "AUC": "roc_auc",
#     "Accuracy": "accuracy",
#     "F1": "f1",
#     "Precision": "precision",
#     "Recall": "recall",
#     "average_precision": "average_precision",
# }
# %% Set hyperparameters
fixed_par = {
    "mu": 1e-3,
    "num_sweeps": 10,
    "train_loss_flag": True,
}
hyperpar = {
    "random_init": [True, False],
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

# %% Load Features for Training
features = pd.read_parquet(FEATURE_FILE)
group_df = pd.read_parquet(GROUP_FILE)
idx_train = group_df.loc[group_df['train'], 'index'].values
features = features.loc[features.index.isin(idx_train)]
group_df = group_df.loc[group_df['index'].isin(idx_train)]
group_df.sort_values("index", inplace=True)
features.sort_values("index", inplace=True)
feat_cols = [col for col in features.columns if DELIM_FEAT_CHAN in col]

# %% iterate over child runs (i.e. per patient)
mlflow.set_tracking_uri(TRACKING_URL)
# mlflow.set_tracking_uri(TRACKING_URL)
mlflow.set_experiment("Convergence_Adapt_CPKRR")
# get unix timestamp
now = datetime.now()
run_name = f"Convergence_Adapt_CPKRR_{now}"

# %% Set variables
random_convergence = {}
source_convergence = {}
with mlflow.start_run(run_name=run_name) as parent_run:
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
        scaler = FrozenEstimator(deepcopy(PI_model.named_steps["scaler"]))  # Freeze the scaler object to prevent refitting
        clf = deepcopy(PI_model.named_steps["clf"])
        clf.set_params(**fixed_par)
        clf.set_params(w_init=deepcopy(PI_model["clf"].weights_))

        pipeline = Pipeline([("scaler", scaler), ("clf", clf)])
        # %% Set up tags
        tags = TAGS.copy()
        tags["patient"] = patient
        tags["parent_run_id"] = child_run["run_uuid"]

        # %% Fit the models
        unique_groups = group_df.loc[group_df["patient"] == patient, "group"].unique()
        # randomly select 1 group for testing
        np.random.seed(42)
        selected_group = np.random.choice(unique_groups, 1)[0]
        idx_group = group_df.loc[group_df["group"] == selected_group, "index"].values
        X = features.loc[features['index'].isin(idx_group), feat_cols].to_numpy()
        y = features.loc[features['index'].isin(idx_group), "combined_annotation"].values

        random_init_model = deepcopy(pipeline)
        random_init_model.named_steps["clf"].random_init = True
        random_init_model.fit(X, y)

        source_init_model = deepcopy(pipeline)
        source_init_model.named_steps["clf"].random_init = False
        source_init_model.fit(X, y)

        # %% Save the models
        random_model_name = f"random_{patient}"
        source_model_name = f"source_{patient}"
        mlflow.sklearn.log_model(random_init_model, random_model_name, signature=infer_signature(X))
        mlflow.sklearn.log_model(source_init_model, source_model_name, signature=infer_signature(X))

        # %% Get the convergence of the models
        random_convergence[patient] = random_init_model.named_steps["clf"].train_loss
        source_convergence[patient] = source_init_model.named_steps["clf"].train_loss

    # %% Save the convergence as json artifact
    # with open("random_convergence.json", "w") as f:
    #     json.dump(random_convergence, f)
    # with open("source_convergence.json", "w") as f:
    #     json.dump(source_convergence, f)
    mlflow.log_dict(random_convergence, "random_convergence.json")
    mlflow.log_dict(source_convergence, "source_convergence.json")

    # Calculate the mean and std of the convergence
    mean_random_convergence = np.mean([np.array(val) for val in random_convergence.values()], axis=0)
    std_random_convergence = np.std([np.array(val) for val in random_convergence.values()], axis=0)
    mean_source_convergence = np.mean([np.array(val) for val in source_convergence.values()], axis=0)
    std_source_convergence = np.std([np.array(val) for val in source_convergence.values()], axis=0)

    # Save the mean and std of the convergence
    mlflow.log_param("mean_random_convergence", mean_random_convergence)
    mlflow.log_param("std_random_convergence", std_random_convergence)
    mlflow.log_param("mean_source_convergence", mean_source_convergence)
    mlflow.log_param("std_source_convergence", std_source_convergence)
    # Save the run name
    # # colormap = plt.cm.get_cmap("colorblind")
    # plt.style.use('tableau-colorblind10')
    # plt.figure()
    # plt.plot(mean_random_convergence, label="Random Initialization", color=colormap(0))
    # plt.fill_between(range(len(mean_random_convergence)), mean_random_convergence - std_random_convergence,
    #                  mean_random_convergence + std_random_convergence, alpha=0.3, color=colormap(0))
    # plt.plot(mean_source_convergence, label="Source Initialization", color=colormap(1))
    # plt.fill_between(range(len(mean_source_convergence)), mean_source_convergence - std_source_convergence,
    #                  mean_source_convergence + std_source_convergence, alpha=0.3, color=colormap(1))
    # plt.legend()
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    #
    # plt.savefig("convergence_plot.png")
    # mlflow.log_artifact("convergence_plot.png")

print("Success!")


