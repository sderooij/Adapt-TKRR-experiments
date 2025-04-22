#!/bin/bash

EXTRACT_FEAT=1
PREPROCESS=0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate seizure_env

if [ $EXTRACT_FEAT -eq 1 ]; then
  window=2.0
  seiz_ovlp=0.9
  bckg_ovlp=0.5
  filename="features_frontal"
  channels="front"

  echo "Extracting features ..."
  python create_feature_database.py --window $window --seiz_ovlp $seiz_ovlp --bckg_ovlp $bckg_ovlp --filename $filename --channels $channels
fi
if [ $PREPROCESS -eq 1 ]; then
    echo "Preprocessing features ..."
    python preprocess_features.py
#    echo "Selecting training samples ..."
#    python create_training_sets.py
#    echo "Combining datasets for PI case..."
#    python combine_datasets.py
fi

echo "exit 0"