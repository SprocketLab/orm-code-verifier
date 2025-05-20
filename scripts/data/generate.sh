#! /bin/bash
QUERY_DIR=$1
shift 1
set -e
DATASETS=(
    "code_contests"
    "gsm8k"

)

for dataset in ${DATASETS[@]}; do
    for shard in $QUERY_DIR/$dataset/*.jsonl.gz; do
        echo "Processing $shard"
        python scripts/data/gen_train_shard.py $shard --quantized $@
    done
done

