#! /bin/bash
set -e
DATASET=$1
MODEL=$2
SAMPLING_SETUP=$3
NUM_CPU=$4
SAVE_NAME=$5
TRIALS=$6
shift 6

python run_trials.py $DATASET $MODEL $SAMPLING_SETUP \
    --output_dir=$SAVE_NAME \
    --num_trials=$TRIALS \
    --num_workers=$NUM_CPU \
    -cleanup \
    $@