#! /bin/bash
NUM_WORKERS=$1
DATASET=$2
SHARD_FILE=$3
shift 3

python exec_shard.py --num_workers=$NUM_WORKERS --timeout=30 --command_timeout=10 --output_dir=. --output_name=output $@ $DATASET $SHARD_FILE $DATASET.jsonl.gz