import mlflow
from config import TRACKING_URL, DEFAULT_ARTIFACT_LOCATION

# EXPERIMENT_NAME = "CPKRR_PI_TUSZ_TLE_2024"

EXPERIMENT_NAME = "Convergence_Adapt_CPKRR"
artifact_location = "file:///U:/secureseizuredata/mlflow/artifacts"
# TAGS = {
#     "data": DATASET,
#     "model": MODEL,
#     "experiment_type": EXPERIMENT_TYPE,
#     "cross_val_type": CV_TYPE,
# }

mlflow.set_tracking_uri(uri=TRACKING_URL)

mlflow.create_experiment(
    name=EXPERIMENT_NAME,
    artifact_location=artifact_location,
    # tags=TAGS,
)
