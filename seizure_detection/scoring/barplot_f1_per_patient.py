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

PS_file = "U:/secureseizuredata/seize_it/results/PS/PS_CPKRR_MODEL/aggregated_results.csv"
PI_file = "U:/secureseizuredata/seize_it/results/PI/CPKRR_PI_LOPO_1731508385.csv"
PF_file = "U:/secureseizuredata/seize_it/results/PF/PF_CPKRR_MODEL_1e-2/aggregated_results.csv"

PS_df = pd.read_csv(PS_file, index_col=0)
PS_df['Patient'] = PS_df.index
PS_df['Model'] = 'PS'
patients = PS_df["Patient"].values
PI_df = pd.read_csv(PI_file, index_col=0)
PI_df['Patient'] = PI_df.index
PI_df['Model'] = 'PI'
PI_df = PI_df.loc[patients]
PF_df = pd.read_csv(PF_file, index_col=0)
PF_df['Patient'] = PF_df.index
PF_df['Model'] = 'PF'

# concetenate the dataframes
df = pd.concat([PI_df, PF_df, PS_df], axis=0)
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
# hatches = itertools.cycle(['/', '\\', '|']) # '-', '+', 'x', 'o', 'O', '.', '*']
# num_locations = len(df['Patient'].unique())
# hatches = ["//", "\\\\", "+"]
# # Loop over the bars
# for bars, hatch in zip(ax.containers, hatches):
#     # Set a different hatch for each group of bars
#     for bar in bars:
#         bar.set_hatch(hatch)
#
handles = [mpatches.Rectangle((0, 0), 1, 1, facecolor=cmap[i], label=label) for i,
    label in enumerate([
        "PI", "PA", "PS"])]
plt.legend(handles=handles,ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18), fancybox=True, shadow=False)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, labels=["PI", "PA", "PS"])

plt.xlabel('Patient', labelpad=15)
# add some space between y-axis label and y ticks
plt.ylabel('F1 score', labelpad=15)


# # rotate x labels
# plt.xticks(rotation=45)
#
plt.tight_layout()
# plt.show()
plt.savefig("barplot_f1_per_patient.pdf", dpi=300, bbox_inches='tight')
