#! /bin/bash
set -e
# Help function
print_usage() {
    echo "Usage: $0 EXPERIMENT MODEL DEVICE SEED [options]"
    echo ""
    echo "Required Arguments:"
    echo "  EXPERIMENT         Name of the experiment"
    echo "  MODEL             Name of the model"
    echo "  DEVICE            Device ID to run on"
    echo "  SEED              Random seed"
    echo ""
    echo "Optional Arguments:"
    echo "  --help            Show this help message"
    echo "  --variant         Specify variant name"
    echo "  --batch_size      Training batch size"
    echo "  --real_batch_size Real batch size (gradient_accumulation = real_batch_size/batch_size)"
    echo "  --eval_batch_tokens     Maximum number of tokens to use in evaluation"
    echo "  --num_workers     Number of worker processes (default: 4)"
    echo "  --disable_wandb   Disable Weights & Biases logging"
    echo "  --debug           Enable debug mode"
    echo "  --precision       Model precision (default: bf16)"
    echo "  --val_batch_tokens Maximum number of tokens per validation batch"
    echo "  --skip_eval       Skip the evaluation step"
    exit 0
}

# Check for help argument
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    print_usage
fi
OUTPUT_DIR=${DEFAULT_OUTPUT_DIR:-"$(pwd)/outputs"}
OUTPUT_DIR="${OUTPUT_DIR}/corm-project"


# Default values
VARIANT=""
BATCH_SIZE=4
REAL_BATCH_SIZE=64
EVAL_TOKENS=2048
EXAMPLES_PER_PROBLEM=6
NUM_WORKERS=4
DISABLE_WANDB=true 
DEBUG=false
PRECISION="bf16"
VAL_BATCH_TOKENS=2048
SKIP_EVAL=false
# Ensure minimum number of arguments
if [ $# -lt 4 ]; then
    echo "Error: Missing required positional arguments"
    print_usage
fi

echo $@
# Get required positional arguments first
EXPERIMENT=$1
MODEL=$2
DEVICE=$3
SEED=$4
shift 4
EVAL_SUITE="zero_shot"
EXTRA_ARGS=()
EVAL_ARGS=()
# Parse optional named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            print_usage
            ;;
        --variant=*)
            VARIANT="${1#*=}"
            shift
            ;;
        --batch_size=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        --real_batch_size=*)
            REAL_BATCH_SIZE="${1#*=}"
            shift
            ;;
        --eval_batch_tokens=*)
            EVAL_TOKENS="${1#*=}"
            shift
            ;;
        --val_batch_tokens=*)
            VAL_BATCH_TOKENS="${1#*=}"
            shift
            ;;
        --examples_per_problem=*)
            EXAMPLES_PER_PROBLEM="${1#*=}"
            shift
            ;;
        --num_workers=*)
            NUM_WORKERS="${1#*=}"
            shift
            ;;
        --track)
            DISABLE_WANDB=false
            shift
            ;;
        --debug)
            EXTRA_ARGS+=(
                "--debug"
                "num_problems=100"
                "num_train_examples=1000"
            )
            EVAL_ARGS+=(
                "--debug"
                "--per_prob=10"
            )
            DEBUG=true
            shift
            ;;
        --precision=*)
            PRECISION="${1#*=}"
            EXTRA_ARGS+=("precision=$PRECISION")
            shift
            ;;
        --skip_eval)
            SKIP_EVAL=true
            shift
            ;;
        --eval_suite=*)
            EVAL_SUITE="${1#*=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

GROUP_NAME="${EXPERIMENT}"
EXP_DIR="${OUTPUT_DIR}/train/${MODEL}-${EXPERIMENT}"
RUN_NAME="${EXPERIMENT}"
if [[ "$DEBUG" == "true" ]]; then
    RUN_NAME="debug-${RUN_NAME}"
fi
EXTRA_ARGS+=("eval_batch_tokens=$VAL_BATCH_TOKENS")
GROUP_NAME="${EXPERIMENT}"
if [[ ! -z "$VARIANT" ]]; then
    RUN_NAME="${RUN_NAME}-${VARIANT}"
    EXP_DIR="${EXP_DIR}/${MODEL}_${VARIANT}"
    GROUP_NAME="${GROUP_NAME}_${VARIANT}"
    EVAL_ARGS+=("--variant=$VARIANT")
fi
RUN_NAME="${RUN_NAME}_seed${SEED}_ex${EXAMPLES_PER_PROBLEM}"
EXP_DIR="${EXP_DIR}/${RUN_NAME}"

if [[ "$DISABLE_WANDB" == "true" ]]; then
    EXTRA_ARGS+=("--enable_wandb")
    EVAL_ARGS+=("--enable_wandb")
fi

echo "EXP_DIR: ${EXP_DIR}"
echo "RUN_NAME: ${RUN_NAME}"
echo "GROUP_NAME: ${GROUP_NAME}"
echo "DEVICE: ${DEVICE}"
echo "SEED: ${SEED}"
echo "VARIANT: ${VARIANT}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "REAL_BATCH_SIZE: ${REAL_BATCH_SIZE}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "EVAL_TOKENS: ${EVAL_TOKENS}"
echo "VAL_BATCH_TOKENS: ${VAL_BATCH_TOKENS}"
echo "EXAMPLES_PER_PROBLEM: ${EXAMPLES_PER_PROBLEM}"
echo "DISABLE_WANDB: ${DISABLE_WANDB}"
echo "DEBUG: ${DEBUG}"
echo "SKIP_EVAL: ${SKIP_EVAL}"

accelerate launch --mixed_precision=$PRECISION train.py $EXPERIMENT \
    model.name=$MODEL \
    --output_dir="${OUTPUT_DIR}/train" \
    --device $DEVICE \
    seed=$SEED \
    --variant=$VARIANT \
    --batch_size=$BATCH_SIZE \
    --real_batch_size=$REAL_BATCH_SIZE \
    eval_batch_tokens=$EVAL_TOKENS \
    num_workers=$NUM_WORKERS \
    "${EXTRA_ARGS[@]}"

if [ "$SKIP_EVAL" = false ]; then
    accelerate launch \
        --mixed_precision=bf16 \
        evaluate_model.py \
        --precision=bf16 \
        --device=0 \
        -group="$GROUP_NAME" \
        --overwrite \
        --max_tokens_per_batch=$EVAL_TOKENS \
        --seed="${SEED}" \
        --num_workers=16 \
        qc-inst-7b \
        t1.0_n128 \
        checkpoint \
        "${EXP_DIR}" \
        "${EVAL_SUITE}" \
        "${EVAL_ARGS[@]}"
fi