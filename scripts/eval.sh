#! /bin/bash
set -e

SEEDS=(
    # "1"
    # "1999"
    "2024"
)

TRAIN_DIR="/hdd5/gorlanski/corm-project/train/"


SUITES=(
    "zero_shot_syntax"
    # "zero_shot_lint"
)

GENEMODELS=(
    # "qc-inst-7b"
    # "qc-inst-3b"
    # "qc-inst-1_5b"
    "qc-inst-500m"
    # "qc-inst-14b"
)

for seed in "${SEEDS[@]}"; do
    for genemodel in "${GENEMODELS[@]}"; do
        if [ "${genemodel}" == "qc-inst-7b" ] && [ "${seed}" == "1999" ]; then
            continue
        fi
        accelerate launch --gpu_ids 0 \
            --mixed_precision=bf16 \
            --config_file=configs/accelerate.yaml \
            evaluate_model.py \
            --precision=bf16 \
            --device=0 \
            -group="rm_qsol" \
            --max_tokens_per_batch=500000 \
            --seed="${seed}" \
            --overwrite \
            --num_workers=16 \
            "${genemodel}" \
            t1.0_n128 \
            checkpoint \
            "${TRAIN_DIR}/qwen25-coder-500m-rm_qsol/rm_qsol_seed${seed}_ex6" \
            zero_shot_lint
        accelerate launch --gpu_ids 0 \
            --mixed_precision=bf16 \
            --config_file=configs/accelerate.yaml \
            evaluate_model.py \
            --precision=bf16 \
            --device=0 \
            -group="rm_qsol" \
            --overwrite \
            --max_tokens_per_batch=200000 \
            --seed="${seed}" \
            --num_workers=16 \
            "${genemodel}" \
            t1.0_n128 \
            checkpoint \
            "${TRAIN_DIR}/qwen25-coder-1_5b-rm_qsol/rm_qsol_seed${seed}_ex6" \
            zero_shot_lint
    done
done




# SAMPLING_SETUP=(
#     "t0.2_n128"
#     "t0.4_n128"
#     "t0.6_n128"
#     "t0.8_n128"
# )
# SUITES=(
#     "zero_shot"
#     "zero_shot_3s1t"
#     "zero_shot_3s3t"
#     # "zero_shot_3s5t"
#     "zero_shot_3s10t"
# )
# for seed in "${SEEDS[@]}"; do
#     for sam_setup in "${SAMPLING_SETUP[@]}"; do
#         for suite in "${SUITES[@]}"; do
#             accelerate launch --gpu_ids 0 \
#                 --mixed_precision=bf16 \
#                 --config_file=configs/accelerate.yaml \
#                 evaluate_model.py \
#                 --precision=bf16 \
#                 --device=0 \
#                 -group="rm_qsol" \
#                 --max_tokens_per_batch=500000 \
#                 --seed="${seed}" \
#                 --overwrite \
#                 --num_workers=16 \
#                 "qc-inst-7b" \
#                 "${sam_setup}" \
#                 checkpoint \
#                 "${TRAIN_DIR}/qwen25-coder-500m-rm_qsol/rm_qsol_seed${seed}_ex6" \
#                 "${suite}"
#             accelerate launch --gpu_ids 0 \
#                 --mixed_precision=bf16 \
#                 --config_file=configs/accelerate.yaml \
#                 evaluate_model.py \
#                 --precision=bf16 \
#                 --device=0 \
#                 -group="rm_qsol" \
#                 --overwrite \
#                 --max_tokens_per_batch=200000 \
#                 --seed="${seed}" \
#                 --num_workers=16 \
#                 "qc-inst-7b" \
#                 "${sam_setup}" \
#                 checkpoint \
#                 "${TRAIN_DIR}/qwen25-coder-1_5b-rm_qsol/rm_qsol_seed${seed}_ex6" \
#                 "${suite}"
#         done
#     done
# done 

# for seed in "${SEEDS[@]}"; do

#     accelerate launch --gpu_ids 0 \
#         --mixed_precision=bf16 \
#         --config_file=configs/accelerate.yaml \
#         evaluate_model.py \
#         --precision=bf16 \
#         --device=0 \
#         -group="dpo_qsol" \
#         --max_tokens_per_batch=24000 \
#         --seed="${seed}" \
#         --overwrite \
#         --num_workers=16 \
#         "qc-inst-7b" \
#         t1.0_n128 \
#         checkpoint \
#         "${TRAIN_DIR}/qwen25-coder-500m-dpo_qsol/dpo_qsol_seed${seed}_ex6" \
#         "zero_shot"
#     accelerate launch --gpu_ids 0 \
#         --mixed_precision=bf16 \
#         --config_file=configs/accelerate.yaml \
#         evaluate_model.py \
#         --precision=bf16 \
#         --device=0 \
#         -group="dpo_qsol" \
#         --overwrite \
#         --max_tokens_per_batch=16000 \
#         --seed="${seed}" \
#         --num_workers=16 \
#         "qc-inst-7b" \
#         t1.0_n128 \
#         checkpoint \
#         "${TRAIN_DIR}/qwen25-coder-1_5b-dpo_qsol/dpo_qsol_seed${seed}_ex6" \
#         "zero_shot"
#     accelerate launch --gpu_ids 0 \
#         --mixed_precision=bf16 \
#         --config_file=configs/accelerate.yaml \
#         evaluate_model.py \
#         --precision=bf16 \
#         --device=0 \
#         -group="clm_qsol" \
#         --max_tokens_per_batch=24000 \
#         --seed="${seed}" \
#         --overwrite \
#         --num_workers=16 \
#         "qc-inst-7b" \
#         t1.0_n128 \
#         checkpoint \
#         "${TRAIN_DIR}/qwen25-coder-500m-clm_qsol/clm_qsol_seed${seed}_ex6" \
#         "zero_shot"
#     accelerate launch --gpu_ids 0 \
#         --mixed_precision=bf16 \
#         --config_file=configs/accelerate.yaml \
#         evaluate_model.py \
#         --precision=bf16 \
#         --device=0 \
#         -group="clm_qsol" \
#         --overwrite \
#         --max_tokens_per_batch=16000 \
#         --seed="${seed}" \
#         --num_workers=16 \
#         "qc-inst-7b" \
#         t1.0_n128 \
#         checkpoint \
#         "${TRAIN_DIR}/qwen25-coder-1_5b-clm_qsol/clm_qsol_seed${seed}_ex6" \
#         "zero_shot"
# done 