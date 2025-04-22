# Scoring
This folder contain the code to generate the results for the seizure detection models.

## Directory Structure
```bash
classification/
├── README.md
├── config.py                   # Contains the Mlflow tracking URI and artifact location, needs to be modified
├── mlflow_load.py              # Contain a helper Class to load the runs from the MLflow tracking server.
├── sz_scoring.py               # Calculated the event-based scores for the models using the `scoring_functions.py` file.
├── scoring_functions.py        # Contains the functions to calculate the event-based scores for the models. Uses the timescoring package.
├── aggregate_LOSI_scoring.py   # To average the event-based scores over the LOSI crossvalidation folds, to get overall and per-patient scores.
├── barplot_f1_per_patient.py    # To create the barplot of the F1 scores per patient.
├── barplot_different_mu_per_patient.py # To create the barplot of the different mu scores per patient.
├── plot_utitls.py              # Contains function to plot average ROC curves.
├── plot_average_curves.py      # To plot the average ROC curves for the models.
├── plot_convergence.py         # To obtain the convergence plot.
├── LOSI_example.json         # Example .json file with a model's configuration, i.e. run_id and run_name.
├── Convergence_Adapt_TKRR.py # To run the experiments for the convergence plot.
├── get_model_size_and_inference_time.py # To get the model size and inference time for the models.
```

## Notes

- Due to the splits in the LOSI crossvalidation folds being in a .edf file (`.../files_per_group.json`), the scoring function for the event-based scoring are adapted to be able to include a 'mask', for the parts of the recording that are not in the test data.
- Again everything relies on a MLFlow tracking server, so the `config.py` file needs to be modified to point to the correct server and artifact location. As well as the correct experiment name and run_id needs to be provided. An example .json file with a model's configuration is provided in `LOSI_example.json`.
- 