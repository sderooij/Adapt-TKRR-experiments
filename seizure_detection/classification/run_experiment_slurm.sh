#!/bin/bash
#
#SBATCH --job-name="seize_it_PI"
#SBATCH --partition=compute-p1
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBARCH --nodes=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=research-eemcs-me
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/selinederooij/src/TL-TKRR-experiments/outfiles/PI_experiment_%j_frontal.out

module load 2023r1
module load miniconda3
module load openssh
module load git

# Set variables
CLASSIFIER='CPKRR'
HYPERPARAMS='parameters_PI_CPKRR_grid.json'
SAVE_FILE='/scratch/selinederooij/seize_it/models/PI_CPKRR_RBF_frontal_250129.pkl'
GRID_SEARCH='full'
SCALER='min-max2'
N_JOBS=15
GROUP_FILE='/scratch/selinederooij/seize_it/bhe_group_df.parquet'
FEATURE_FILE='/scratch/selinederooij/seize_it/features_preprocessed.parquet'

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Activate conda environment"
conda activate seizv2 #seizure_env

echo "Starting python"

srun python PI_experiment.py --classifier $CLASSIFIER --hyperparams $HYPERPARAMS --save_file $SAVE_FILE --grid_search $GRID_SEARCH --n_jobs $N_JOBS --scaler $SCALER --group_file $GROUP_FILE --feature_file $FEATURE_FILE

conda deactivate
echo $SAVE_FILE