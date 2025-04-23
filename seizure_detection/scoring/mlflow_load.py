import pandas
import sqlalchemy
import pandas as pd
import numpy as np
import cloudpickle
import mlflow
import json

from mlflow import MlflowClient
import mlflow.artifacts

from config import TRACKING_URL


class MLFlowLoader:
    def __init__(self, tracking_url, run_id, *, run_name=False, local_project_drive=None, split_on="staff-umbrella"):
        self.tracking_url = tracking_url
        if run_name:
            self.run_name = run_id
            run_id = get_runid_from_run_name(tracking_url, run_id)
        else:
            self.run_name = get_run_name_from_runid(tracking_url, run_id)
        self.run_id = run_id
        self.artifact_path = self._get_artifact_path()
        self.experiment_name = get_experiment_name(tracking_url, run_id)
        self.local_project_drive = local_project_drive
        if local_project_drive is not None:
            self.update_artifact_path(local_project_drive, split_on=split_on)
        # self.client = MlflowClient(tracking_url)

    def _get_artifact_path(self):
        engine = sqlalchemy.create_engine(self.tracking_url)
        conn = engine.connect()
        runs = pd.read_sql_table("runs", conn)
        artifact_path = runs[runs["run_uuid"] == self.run_id]["artifact_uri"].values[0]
        return artifact_path

    def update_artifact_path(self, local_project_drive, *, split_on="staff-umbrella"):
        begin, end = self.artifact_path.split(split_on)
        self.artifact_path = local_project_drive + end
        return self

    def load_artifact(self, artifact_name, **kwargs):
        """Load artifact from mlflow. Only for .csv or .parquet files.

        Args:
            artifact_path (str): name of artifact (including extension)

        Returns:
            pd.DataFrame: artifact
        """
        artifact_path = self.artifact_path + "/" + artifact_name
        if artifact_name.endswith(".csv"):
            artifact = pd.read_csv(artifact_path, **kwargs)
        elif artifact_name.endswith(".parquet"):
            artifact = pd.read_parquet(artifact_path, **kwargs)
        elif artifact_name.endswith(".json"):
            if self.local_project_drive is not None:
                with open(artifact_path, "r") as f:
                    artifact = json.load(f, **kwargs)
            else:
                temp = mlflow.artifacts.download_artifacts(run_id=self.run_id, artifact_path=artifact_name, tracking_uri=self.tracking_url)
                with open(temp, "r") as f:
                    artifact = json.load(f, **kwargs)

        else:
            raise ValueError("Only .csv and .parquet files are supported.")
        return artifact

    def load_model(self, model_folder, model_type="sklearn"):
        """Load model from mlflow.

        Args:
            model_folder (str): name of model folder

        Returns:
            model: model
        """
        model_path = self.artifact_path + "/" + model_folder
        if model_type == "sklearn":
            model = mlflow.sklearn.load_model(model_path)
        elif model_type == "pytorch":
            model = mlflow.pytorch.load_model(model_path)
        elif model_type == "keras":
            model = mlflow.keras.load_model(model_path)
        elif model_type == "tensorflow":
            model = mlflow.tensorflow.load_model(model_path)
        else:
            raise ValueError("Only sklearn and pytorch models are supported.")
        return model

    # def fetch_logged_data(self):
    #     """Fetch logged data from mlflow.
    #
    #     Returns:
    #         pd.DataFrame: logged data
    #     """
    #     data = self.client.get_run(self.run_id).data
    #     tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    #     artifacts = [f.path for f in self.client.list_artifacts(self.run_id, "sk_model")]
    #     return data.params, data.metrics, tags, artifacts
    #
    # def get_tags(self):
    #     """Get tags from mlflow.
    #
    #     Returns:
    #         pd.DataFrame: tags
    #     """
    #     data = self.client.get_run(self.run_id).data
    #     tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    #     return tags

def get_experiment_name(sqlite_db, run_id):
    """Get experiment name from run id.

    Args:
        sqlite_db (uri): uri of sqlite database
        run_id (str): id of run

    Returns:
        str: experiment name
    """
    engine = sqlalchemy.create_engine(sqlite_db)
    conn = engine.connect()
    runs = pd.read_sql_table("runs", conn)
    experiment_id = runs[runs["run_uuid"] == run_id]["experiment_id"].values[0]
    experiments = pd.read_sql_table("experiments", conn)
    experiment_name = experiments[experiments["experiment_id"] == experiment_id]["name"].values[0]
    return experiment_name

def get_child_runs_from_parent(sqlite_db, parent_id):
    """Get child runs id's from parent run id.

    Args:
        sqlite_db (uri): uri of sqlite database
        parent_id (str): id of parent run (or list of parent runs)

    Returns:
        list: list of child runs
    """
    engine = sqlalchemy.create_engine(sqlite_db)
    conn = engine.connect()
    tags = pd.read_sql_table("tags", conn)
    tags = tags.pivot(index="run_uuid", columns="key", values="value")
    tags = tags.reset_index()
    if isinstance(parent_id, list):
        child_runs = []
        for parent in parent_id:
            child_runs.append(tags[tags["mlflow.parentRunId"] == parent])
        child_runs = pd.concat(child_runs)
    else:
        child_runs = tags[tags["mlflow.parentRunId"] == parent_id]
    return child_runs


def get_runid_from_run_name(sqlite_db, run_name):
    """Get run id from run name.

    Args:
        sqlite_db (uri): uri of sqlite database
        run_name (str): name of the run

    Returns:
        str: run id
    """
    engine = sqlalchemy.create_engine(sqlite_db)
    conn = engine.connect()
    runs = pd.read_sql_table("runs", conn)
    run_id = runs[runs["name"] == run_name]["run_uuid"].values[0]
    return run_id


def get_run_name_from_runid(sqlite_db, run_id):
    """Get run name from run id.

    Args:
        sqlite_db (uri): uri of sqlite database
        run_id (str): id of the run

    Returns:
        str: run name
    """
    engine = sqlalchemy.create_engine(sqlite_db)
    conn = engine.connect()
    runs = pd.read_sql_table("runs", conn)
    run_name = runs[runs["run_uuid"] == run_id]["name"].values[0]
    return run_name


if __name__ == "__main__":
    parent_id = "ef5b495d3f3e4563a14071a2c774a23c"
    child_runs = get_child_runs_from_parent(TRACKING_URL, parent_id)
    print(child_runs)
