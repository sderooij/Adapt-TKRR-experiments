from sklearn.model_selection import LeaveOneGroupOut, cross_validate, LeavePGroupsOut
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
# from classification import SeizureClassifier
from seizure_data_processing.config import SEIZE_IT_DIR
from seizure_data_processing.classification import SeizureClassifier, get_scaler, get_classifier
import pandas as pd
import json
import argparse


def get_cmd_input():
    parser = argparse.ArgumentParser(description="Classify EEG data")
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=None,
        help="JSON file with hyperparameters of the classifier.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="SVM",
        help="Classifier to use SVM or CPKRR (now supported).",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="Location to save model.",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="min-max2",
        help="Scaler to use standard, min-max, max-abs.",
    )
    parser.add_argument(
        "--grid_search",
        type=str,
        default="full",
        help="Grid search type to use.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of jobs to run in parallel.",
    )
    parser.add_argument(
        "--grid_search_scoring",
        type=str,
        default="average_precision",
        help="Scoring metric for grid search.",
    )
    parser.add_argument(
        "--group_file",
        type=str,
        default=SEIZE_IT_DIR + f"/features_preprocessed.parquet",
        help="Validation groups file.",
    )

    parser.add_argument(
        "--feature_file",
        type=str,
        default=SEIZE_IT_DIR + f"/features_preprocessed.parquet",
        help="Feature file.",
    )

    args = parser.parse_args()
    return args


DELIM_FEAT_CHAN = "|"
# %% SET experiment !!!!! CHECK BEFORE RUNNING !!!
MODEL_TYPE = "PI"
CLASSIFIER_NAME = "CPKRR"    # overruled by command line argument if given
CROSS_VAL_TYPE = "LOPO"
DATASET = "seize_it"

# %% Get command line arguments
args_cmd = get_cmd_input()
if args_cmd.hyperparams is not None:
    import json

    with open(args_cmd.hyperparams) as f:
        hyperparams = json.load(f)
else:
    # %% set hyperparams
    hyperparams = {
        'M': [10],
        'max_rank': [10],
        'reg_par': [1e-5],
        'num_sweeps': [1],
        'map_param': [0.3],
        'feature_map': ['rbf'],
    }

if args_cmd.classifier is not None:
    CLASSIFIER_NAME = args_cmd.classifier

SCALER = args_cmd.scaler

GRID_SEARCH = args_cmd.grid_search
n_jobs = args_cmd.n_jobs
GRID_SEARCH_SCORING = args_cmd.grid_search_scoring

# %%
if GRID_SEARCH_SCORING == 'f3':
    from sklearn.metrics import make_scorer, fbeta_score

    f3_scorer = make_scorer(fbeta_score, beta=3)
    GRID_SEARCH_SCORING = f3_scorer
elif GRID_SEARCH_SCORING == 'f2':
    from sklearn.metrics import make_scorer, fbeta_score

    f2_scorer = make_scorer(fbeta_score, beta=2)
    GRID_SEARCH_SCORING = f2_scorer

# %%
print(hyperparams)

tags = {
    "dataset": DATASET,
    "model_type": MODEL_TYPE,
    "classifier": CLASSIFIER_NAME,
    "cross_val_type": CROSS_VAL_TYPE,
    "grid_search": GRID_SEARCH,
    "grid_search_scoring": GRID_SEARCH_SCORING,
    "scaler": SCALER,
    "annotation_column": "combined_annotation",
    "bhe": True,
}

file_inf_df = pd.read_csv(SEIZE_IT_DIR + "data_info/file_info.csv")
duration = file_inf_df["file_duration"].sum()

feature_file = args_cmd.feature_file
group_file = args_cmd.group_file
# group_file = feature_file

clf = get_classifier(CLASSIFIER_NAME)
scaler = get_scaler(SCALER)
if scaler is not None:
    preprocessing_steps = {'scaler': scaler}
else:
    preprocessing_steps = {}
cv = LeaveOneGroupOut()

seiz_clf = SeizureClassifier(
    clf,
    hyperparams,
    MODEL_TYPE,
    preprocessing_steps,
    cv,
    None,
    DATASET,
    feature_file,
    group_file,
    GRID_SEARCH_SCORING,
    GRID_SEARCH,
    tags,
    n_jobs,
    verbose=1
)

seiz_clf.cross_validate(annotation_column='combined_annotation')
test_features = feature_file
seiz_clf.save_local(args_cmd.save_file)  # checkpoint
seiz_clf.predict(feature_file=test_features, group_file=test_features,
                 annotation_column='combined_annotation')
seiz_clf.save_local(args_cmd.save_file)  # checkpoint
seiz_clf.score()
seiz_clf.print_scores()
seiz_clf.save_local(args_cmd.save_file)
