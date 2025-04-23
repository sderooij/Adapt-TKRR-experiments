import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns
from config import TRACKING_URL
import mlflow_load as mload
from plot_utils import plot_n_average_curves
import yaml
import mlflow
import matplotlib
import re
import json
matplotlib.use('TkAgg')
# matplotlib.rcParams.update({'font.size': 14})
# matplotlib.use("pgf")
# #
matplotlib.rcParams.update(
    {
        'font.size': 20,
#         'text.usetex': True,
        # 'pgf.rcfonts': False,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
    }
)

def load_runs_from_timestamps(model_id, yaml_file, experiment_id):
    with open(yaml_file, 'r') as stream:
        timestamps = yaml.safe_load(stream)
    time = timestamps[f"MODEL_{model_id}"]
    all_runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=f"""tags.mlflow.runName LIKE "CPKRR_P%_LOSI_P_ID%" AND status="FINISHED" """)
    runs = all_runs[all_runs["start_time"].dt.strftime('%Y-%m-%d %H:%m').between(time[0], time[1])]
    outputs = []
    for i, row in runs.iterrows():
        run_id = row['run_id']
        model = mload.MLFlowLoader(TRACKING_URL, run_id)
        output = model.load_artifact("output.parquet")
        patient = row['tags.patient']
        output['patient'] = patient
        outputs.append(output)
    outputs = pd.concat(outputs)
    return outputs

def load_runs_from_json_file(json_file):

    with open(json_file, 'r') as f:
        feeds = json.load(f)
    run_ids = feeds['RUN_ID']
    if isinstance(run_ids, str):    # PI MODEL
        model = mload.MLFlowLoader(TRACKING_URL, run_ids)
        output = model.load_artifact("output.parquet")
        output.rename(columns={'group': 'patient'}, inplace=True)
        return output

    outputs = []
    for idx, run_id in enumerate(run_ids):  # PS and PF models
        model = mload.MLFlowLoader(TRACKING_URL, run_id)
        output = model.load_artifact("output.parquet")
        # get patients from run_name
        run_name = feeds['RUN_NAME'][idx]
        patient = re.search(r'P_ID\d+', run_name).group(0)
        output['patient'] = patient
        outputs.append(output)

    outputs = pd.concat(outputs)
    return outputs

# %% MODEL config files, CHANGE THIS
PI_MODEL_CPKRR = "models/PI_CPKRR_MODEL.json"
PF_MODEL_CPKRR = "models/PF_CPKRR_MODEL.json"
PS_MODEL_CPKRR = "models/PS_CPKRR_MODEL.json"
PI_MODEL_SVM = "models/PI_SVM_MODEL.json"
PF_MODEL_SVM = "models/PF_SVM_MODEL.json"
PS_MODEL_SVM = "models/PS_SVM_MODEL.json"

mlflow.set_tracking_uri(uri=TRACKING_URL)

PF_output_CPKRR = load_runs_from_json_file(PF_MODEL_CPKRR)
PS_output_CPKRR = load_runs_from_json_file(PS_MODEL_CPKRR)
PI_output_CPKRR = load_runs_from_json_file(PI_MODEL_CPKRR)
PF_output_SVM = load_runs_from_json_file(PF_MODEL_SVM)
PS_output_SVM = load_runs_from_json_file(PS_MODEL_SVM)
PI_output_SVM = load_runs_from_json_file(PI_MODEL_SVM)

patients = PF_output_CPKRR['patient'].unique()
PI_output_CPKRR = PI_output_CPKRR[PI_output_CPKRR['patient'].isin(patients)]
PI_output_SVM = PI_output_SVM[PI_output_SVM['patient'].isin(patients)]

# make list of the outputs

num_points = 100
bounds = False
# import seaborn as sns

# %% PLOT SEPARATE CURVES
# sns.set_palette("colorblind")
outputs_CPKRR = [PI_output_CPKRR, PF_output_CPKRR, PS_output_CPKRR]
model_names = ["PI", "PA", "PS"]
plot_n_average_curves(outputs_CPKRR,
                      model_names=model_names,
                      group_column="patient",
                      output_column="predicted_output",
                      label_column="true_label",
                      num_points=num_points,
                      bounds=bounds,
                      fig_size=(9,8),
                      save_file="ROC_curves_TKRR.pdf"
                      )

outputs_SVM = [PI_output_SVM, PF_output_SVM, PS_output_SVM]
model_names = ["PI", "PA", "PS"]
# import seaborn as sns
# sns.set_palette("colorblind")
plot_n_average_curves(outputs_SVM,
                      model_names=model_names,
                      group_column="patient",
                      output_column="predicted_output",
                      label_column="true_label",
                      num_points=num_points,
                      bounds=bounds,
                      fig_size=(9,8),
                      save_file="ROC_curves_SVM.pdf"
                      )
