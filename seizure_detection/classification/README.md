# Classification

This directory contains the code for training and evaluating the classification models for seizure detection. The models are trained on the preprocessed data from the `preprocessing` directory.
We relied heavily on MLflow for tracking experiments. Therefore, this code is not "plug-and-play" and modifications need to be made if you want to run your own experiments.

## Directory Structure
```bash
classification/
├── README.md
├── config.py                   # Contains the Mlflow tracking URI and artifact location, needs to be modified
├── parameters_PI_CPKRR.json    # Contains the parameters for the grid-search of the PI_CPKRR model
├── parameters_PI_SVM.json      # Contains the parameters for the grid-search of the PI_SVM model
├── PI_experiment.py            # Contains the code for training and evaluating the PI models, don't run this file directly but use the run_experiment_slurm files.
├── run_experiment_slurm.sh     # Slurm script to run the PI experiment for the CPKRR model
├── run_experiment_slurm_svm.sh # Slurm script to run the PI experiment for the SVM model
├── PF_detection_LOSI.py        # Contains the code for training and evaluating the patient-adapted models (not that PF is used instead of PA for legacy reasons).
├── PS_experiment.py            # Contains the code for training and evaluating the patient-specific models.
├── load_and_log.py             # Contains the code for loading the locally saved SeizureClassifier object and logging the results to MLflow.
├── new_experiment.py           # Create a new MLflow experiment.

```

## Notes

- For the training (and crossvalidating) the PI models a SLURM cluster was used (DelftBlue). Therefore, to run locally some modifications will need to be made to the shell scripts.
- The PA models are called PF models in the code. This is for legacy reasons and would have been too much work to change.
