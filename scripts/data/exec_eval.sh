#! /bin/bash
DATASET=$1
NUM_WORKERS=$2
shift 2

python exec_eval.py $DATASET.raw.jsonl.gz --num_workers=$NUM_WORKERS --output_dir=. --output_name=output $@