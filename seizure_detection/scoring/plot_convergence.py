from mlflow_load import MLFlowLoader
from config import TRACKING_URL
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# matplotlib.use('TkAgg')

matplotlib.use("pgf")
#
matplotlib.rcParams.update(
    {
        'font.size': 16,
        'text.usetex': False,
        # 'pgf.rcfonts': False,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    }
)
def convert_string_to_list(input_string):
    string_list = input_string.split()
    string_list[0] = string_list[0].replace('[', '')
    string_list[-1] = string_list[-1].replace(']', '')
    # remove empty strings
    string_list = list(filter(None, string_list))
    # Convert the list of strings to a list of floats
    float_list = [float(x) for x in string_list]
    return float_list


def convert_dicts(input_dict):
    for key, value in input_dict.items():
        input_dict[key] = convert_string_to_list(value)
    return input_dict


colormap = sns.color_palette('colorblind', as_cmap=True)
run_id = "381b498ddd1d4e3b9b6b3896e1ee2196"
ml = MLFlowLoader(run_id=run_id, tracking_url=TRACKING_URL)
source_convergence = ml.load_artifact('source_convergence.json')
random_convergence = ml.load_artifact('random_convergence.json')
source_convergence = convert_dicts(source_convergence)
random_convergence = convert_dicts(random_convergence)

# Calculate the mean and std of the convergence
mean_random_convergence = np.mean([np.array(val) for val in random_convergence.values()], axis=0)
std_random_convergence = np.std([np.array(val) for val in random_convergence.values()], axis=0)
mean_source_convergence = np.mean([np.array(val) for val in source_convergence.values()], axis=0)
std_source_convergence = np.std([np.array(val) for val in source_convergence.values()], axis=0)

max_iter = 141
mean_random_convergence = mean_random_convergence[:max_iter]
std_random_convergence = std_random_convergence[:max_iter]
mean_source_convergence = mean_source_convergence[:max_iter]
std_source_convergence = std_source_convergence[:max_iter]

plt.figure()
plt.plot(mean_random_convergence, label="Random Initialization", color=colormap[0], linewidth=2, marker='o',
         markersize=5, markevery=5)
plt.fill_between(range(len(mean_random_convergence)), mean_random_convergence - std_random_convergence,
                 mean_random_convergence + std_random_convergence, alpha=0.3, color=colormap[0])
plt.plot(mean_source_convergence, label="Source Initialization", color=colormap[1], linewidth=2, marker='s',
         markersize=5, markevery=5)
plt.fill_between(range(len(mean_source_convergence)), mean_source_convergence - std_source_convergence,
                 mean_source_convergence + std_source_convergence, alpha=0.3, color=colormap[1])
plt.legend(loc='upper right', fontsize=16)
plt.xlabel("Iteration", fontsize=16)
# leave a bit more space for ylabel
plt.ylabel("Loss", fontsize=16, labelpad=10)
plt.tight_layout()
plt.savefig("convergence_plot.pgf")

diff_optim_loss = mean_random_convergence[-1] - mean_source_convergence[-1]
print(f"Optimization loss difference between random and source initialization: {diff_optim_loss}")