from config import TRACKING_URL
import mlflow_load as mload
import numpy as np
import pandas as pd
import sys
import time
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import re


class ModelTester:
    def __init__(self, run_id=None, run_name=None, tracking_url=None):
        if run_id is None and run_name is None:
            raise ValueError("Either run_id or run_name must be provided.")
        elif run_id is None and run_name is not None:
            run_id = mload.get_runid_from_run_name(tracking_url, run_name)
        elif run_id is not None and run_name is None:
            run_name = mload.get_run_name_from_runid(tracking_url, run_id)
        self.run_id = run_id
        self.run_name = run_name
        self.tracking_url = tracking_url
        self.ml = mload.MLFlowLoader(tracking_url, run_id)
        self.num_model_params = None
        self.inference_time = None
        self.classifier = None

    def load_classifier(self, folder_name="sk_model", model_type="sklearn"):
        model = self.ml.load_model(folder_name, model_type)
        self.classifier = deepcopy(model.named_steps["clf"])
        return self

    def get_model_size(self):
        if "SVC" in self.run_name:
            n_sv = self.classifier.support_vectors_.shape[0]
            n_features = self.classifier.support_vectors_.shape[1]
            self.num_model_params = n_sv*(n_features+1)     # +1 for the coefficients
            return self.num_model_params
        elif "CPKRR" in self.run_name:
            self.num_model_params = np.sum([w.size for w in self.classifier.weights_])
            return self.num_model_params
        elif "SVM_Adapt" in self.run_name:
            n_sv = self.classifier.source_model.support_vectors_.shape[0]
            n_features = self.classifier.source_model.support_vectors_.shape[1]
            n_sv += self.classifier.support_vectors_.shape[0]

            self.num_model_params = n_sv*(n_features+1)
            return self.num_model_params
        else:
            raise ValueError("Model type not recognized, must be CPKRR or SVC")

    def get_inference_time_per_sample(self, X):
        num_samp = X.shape[0]
        start = time.time()
        self.classifier.predict(X)
        end = time.time()
        self.inference_time = (end - start)/num_samp
        return self.inference_time


if __name__ == "__main__":
    patients = [
        "P_ID13", "P_ID16", "P_ID33", "P_ID36", "P_ID61",
        "P_ID65", "P_ID70", "P_ID71", "P_ID76", "P_ID78",
        "P_ID79", "P_ID82", "P_ID83", "P_ID94", "P_ID99"
    ]
    import json
    #%% Get the model from the experiment
    json_file = "LOSI_example.json"
    with open(json_file, 'r') as f:
        feeds = json.load(f)
        RUN_NAME = feeds['RUN_NAME']
        run_id_parent = feeds['RUN_ID']
    # %% Get the run id of the parent and the child runs
    # run_id_parent = mload.get_runid_from_run_name(TRACKING_URL, RUN_NAME)
    run_ids_child = mload.get_child_runs_from_parent(TRACKING_URL, run_id_parent)

    # %% Load the outputs of the child runs
    FEATURE_FILE = "features_preprocessed.parquet"
    query = [('test', '==', True)]
    features = pd.read_parquet(FEATURE_FILE, engine='pyarrow', filters=query)
    # sample the features
    N_samples = 1000
    features = features.sample(n=N_samples, random_state=42, axis=0)
    feat_cols = [col for col in features.columns if "|" in col]
    X = features[feat_cols].to_numpy()
    scaler = MinMaxScaler((-0.5, 0.5))
    X = scaler.fit_transform(X)
    del features

    # %% Iterate over the child runs and get the average model size and inference time
    model_size = []
    inference_time = []
     # counter for for loop
    # import tqdm
    results = [] #pd.DataFrame(columns=['model_size', 'inference_time', 'patient'])

    for i, row in run_ids_child.iterrows():
        run_id = row['run_uuid']
        run_name = row['mlflow.runName']
        try:
            patient = re.search(r"P_ID\d+", run_name).group()
        except:
            parent_id = row['mlflow.parentRunId']
            temp = mload.get_run_name_from_runid(TRACKING_URL, parent_id)
            patient = re.search(r"P_ID\d+", temp).group()

        if patient not in patients:
            continue
        mt = ModelTester(run_id=run_id, tracking_url=TRACKING_URL)
        mt.load_classifier()
        model_size.append(mt.get_model_size())
        inference_time.append(mt.get_inference_time_per_sample(X))
        results.append(pd.Series({'model_size': mt.num_model_params, 'inference_time': mt.inference_time, 'patient': patient}))
        del mt  # to avoid memory overhead
        prev_patient = patient

    results = pd.concat(results, axis=1).T
    results_by_patient = results.groupby('patient').mean()
    # std = results.groupby('patient').std()
    print(f"Run name: {RUN_NAME}")
    print(f"Average model size: {results_by_patient['model_size'].mean()}")
    print(f"Average inference time per sample: {results_by_patient['inference_time'].mean()}")
    # print(f"Std model size: {np.std(model_size)}")
    # print(f"Std inference time per sample: {np.std(inference_time)}")

    with open(json_file, mode='w') as f:
        entry = {"model_size": results_by_patient['model_size'].mean(),
                 "inference_time": results_by_patient['inference_time'].mean(),
                 "std_model_size": results_by_patient['model_size'].std(),
                 "std_inference_time": results_by_patient['inference_time'].std()}
        feeds.update(entry)
        json.dump(feeds, f, indent=4)
    print("Done!")