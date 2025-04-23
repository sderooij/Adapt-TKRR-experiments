import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools
import matplotlib.patches as mpatches
matplotlib.use('TkAgg')

# matplotlib.use("pgf")
#
matplotlib.rcParams.update(
    {
        'font.size': 20,
        'text.usetex': True,
        # 'pgf.rcfonts': False,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
    }
)

PA_mu_1 = "U:/secureseizuredata/seize_it/results/PF/PF_CPKRR_MODEL_1e-1/aggregated_results.csv"
PA_mu_2 = "U:/secureseizuredata/seize_it/results/PF/PF_CPKRR_MODEL_1e-2/aggregated_results.csv"
PA_mu_3 = "U:/secureseizuredata/seize_it/results/PF/PF_CPKRR_MODEL_1e-3/aggregated_results.csv"
PA_mu_4 = "U:/secureseizuredata/seize_it/results/PF/PF_CPKRR_MODEL_1e-4/aggregated_results.csv"

model_names = ["$\mu = 0.1$", "$\mu = 0.01$", "$\mu = 0.001$", "$\mu = 0.0001$"]
df_list = []
for idx, name in enumerate(model_names):
    temp = pd.read_csv(eval(f"PA_mu_{idx+1}"), index_col=0)
    temp['Patient'] = temp.index
    temp['Model'] = name
    df_list.append(temp.copy())

# concetenate the dataframes
df = pd.concat(df_list, axis=0)
df = df.fillna(0)

# rename the values in the patient column to add "\" before the underscore
# df['Patient'] = df['Patient'].str.replace("_", "\_")
# rename so only the number remains
df['Patient'] = df['Patient'].str.extract(r"(\d+)")

# get colorblind colormap
cmap = sns.color_palette("colorblind")
sns.set_palette(cmap)
sns.set_style("whitegrid")
# sns.set_style("ticks")
plt.figure(figsize=(16, 6))

ax = sns.barplot(data=df, x='Patient', y='f1', hue='Model')
# hatches = itertools.cycle(['/', '\\', '-', 'x', '//', '*', 'o', 'O', '.'])
hatches = itertools.cycle(['/', '\\', '|']) # '-', '+', 'x', 'o', 'O', '.', '*']
num_locations = len(df['Patient'].unique())
# hatches = ["//", "=", "\\\\","||"]
# # Loop over the bars
# for bars, hatch in zip(ax.containers, hatches):
#     # Set a different hatch for each group of bars
#     for bar in bars:
#         bar.set_hatch(hatch)

# handles = [mpatches.Rectangle((0, 0), 1, 1, facecolor=cmap[i], label=label, hatch=hatches[i]) for i,
# label in enumerate(model_names)]
# plt.legend(handles=handles,ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.18), fancybox=True, shadow=False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=4)

plt.xlabel('Patient', labelpad=15)
# add some space between y-axis label and y ticks
plt.ylabel('F1 score', labelpad=15)


# # rotate x labels
# plt.xticks(rotation=45)
#
plt.tight_layout()
# plt.show()
plt.savefig("barplot_diff_mu_per_patient.pdf")
