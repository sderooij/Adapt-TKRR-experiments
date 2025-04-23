import pandas as pd
import json
folder = f"U:/secureseizuredata/seize_it/results/PF/PF_CPKRR_MODEL_1e-1/"
# get all .json files in the folder
import re
import os
files = os.listdir(folder)
files = [f for f in files if f.endswith(".json")]
df = pd.DataFrame(columns=['sensitivity', 'precision', 'f1', 'fpRate'])
for file in files:
    if file == "aggregated_results.json":
        continue
    with open(folder + file) as f:
        data = json.load(f)
    # df.loc[file,:] = [data['event_results']['f1'], data['event_results']['precision'], data['event_results'][
    #     'sensitivity'], data['event_results']['fpRate']]
    # find patient name in the file name (P_IDXX), with regex
    patient = re.search(r"P_ID\d+", file).group()
    df.loc[patient, "f1"] = data['event_results']['f1']
    df.loc[patient, "precision"] = data['event_results']['precision']
    df.loc[patient, "sensitivity"] = data['event_results']['sensitivity']
    df.loc[patient, "fpRate"] = data['event_results']['fpRate']

# calculate the mean and std
mean = df.mean()
std = df.std()
# save the mean and std to a .json file with same format as the input files
with open(f"{folder}/aggregated_results.json", "w") as f:
    json.dump(
        {
            "event_results": {
                "sensitivity": mean['sensitivity'],
                "sensitivity_std": std['sensitivity'],
                "precision": mean['precision'],
                "precision_std": std['precision'],
                "f1": mean['f1'],
                "f1_std": std['f1'],
                "fpRate": mean['fpRate'],
                "fpRate_std": std['fpRate'],
            },
        },
        f,
        indent=2,
        sort_keys=False,
    )

df.to_csv(f"{folder}/aggregated_results.csv")

