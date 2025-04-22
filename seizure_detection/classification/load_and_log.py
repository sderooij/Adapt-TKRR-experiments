"""
Load a locally saved model and log it to MLflow.
"""
import mlflow
from config import TRACKING_URL, DEFAULT_ARTIFACT_LOCATION
import sklearn
from sklearn.svm import SVC
from seizure_data_processing.classification.classification import SeizureClassifier, get_scaler, get_classifier
import pickle
import joblib
import cloudpickle
from sklearn.metrics import *

FILE = "..." # path to the model file of local save

EXPERIMENT_NAME = "SEIZEIT_PF_SVM_LOSI_2025" #  MFLOW experiment name
PF = True
PF_DIRECTORY = "..."
# find all files with give string
if PF:
    import os
    import glob
    # files = glob.glob(r"C:\Users\selinederooij\Documents\seize_it\models\2025-03-28\*.pickle")
    files = glob.glob(PF_DIRECTORY + "*.pickle")
    # exclude FILE from files
    # files = [file for file in files if file != FILE]
    for file in files:
        # load pickle file
        with open(file, "rb") as f:
            model = pickle.load(f)


        # %% test/predict if not already done
        # group_file = "U:/secureseizuredata/seize_it/bhe_group_df.parquet"
        # features_file = "U:/secureseizuredata/seize_it/features_preprocessed.parquet"
        # model.predict(features_file, group_file, annotation_column="combined_annotation")
        # model.save_local(FILE)
        # model.score()
        # model.print_scores()
        # model.save_local(FILE)
        #%%

        model.log_mlflow(
            tracking_url=TRACKING_URL,
            experiment_name=EXPERIMENT_NAME,
            temp_dir="temp/",
        )
else:
    with open(FILE, "rb") as f:
        model = pickle.load(f)
    model.log_mlflow(
        tracking_url=TRACKING_URL,
        experiment_name=EXPERIMENT_NAME,
        temp_dir="temp/",
    )

