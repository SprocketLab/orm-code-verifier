# Reward Models Enable Scalable Code Verification by Trading Accuracy for Throughput

# Installation
## Clone the repository
```sh
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

## Create a virtual environment
```sh
# Install Overall Dependencies for Training and Eval
pip install -r requirements.txt 
```

# Data Generation
## Train Data Generation
To generate train data
```sh
python generate.py --overwrite train-data Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int8 code_contests --num_per=128 --max_num_seqs=256 -chat
```
Then follow the instructions in [the scripts/data README to execute the data and create the HF dataset](/scripts/data/)

## Eval Data Generation
```sh
python generate.py eval-ds Qwen/Qwen2.5-Coder-7B-Instruct --max_num_seqs=128 -n 128 --is_chat
```

Again follow the instructions in [the scripts/data README to execute the data and create the HF dataset](/scripts/data/)


## Execution Trials
First you need to create a separate environment for execution (We recommend using Docker)
```sh
pip install -r scripts/exec_trials/exec_requirements.txt
```

Once you have your evaluation sets, or use ours from huggingface, you can run the different execution trials. To run the strongest verifier you would run:
```sh
bash scripts/exec_trials/trial.sh {DATASET} {MODEL} {SAMPLING_SETUP} {NUM_CPU} {SAVE_PATH} {NUM TRIALS}
```
Here is what the strongest verifier for our setup looks like:
```sh
bash scripts/exec_trials/trial.sh code_contests qc-inst-7b t1.0_n128 32 outputs/ftp32_code_contets 5
```

To create the filters then follow the instructions [in that directory](scripts/exec_trials). We already provide all of the filters we used. 

# Training
You first need to run the [`make_train_data.py`](scripts/make_train_data.py) script. Here is how we did it for our experiments:
```sh
python scripts/make_train_data.py --require_pf --black_format --num_proc=16 
```

To train the model, you can use the provided [`experiment.sh`](scripts/experiment.sh) script. Here is how we would run the 1.5B model:
```sh
bash scripts/experiment.sh rm_qsol qwen25-coder-1_5b 0 1 \
    --precision=bf16 \
    --num_workers=4 \
    --real_batch_size=64 \
    --overwrite \
    --batch_size=2 \
    --val_batch_tokens=12000 \
    gradient_checkpointing=True \ 
    --eval_batch_tokens=200000 
```

This additionally evaluates the model on the 0 Shot setting.

For our experiments we ran with the seeds `1, 1999, 2024`

# Evaluation

To run on a specific suite, with a specific filter, you can do the following:
```sh
accelerate launch \
    --gpu_ids 0 \
    --mixed_precision=bf16 \
    --config_file=configs/accelerate.yaml \
    evaluate_model.py \
    --precision=bf16 \
    --device=0 \
    -group={WANB_GROUP_NAME} \
    --overwrite \
    --max_tokens_per_batch=6000 \
    --seed={SEED} \
    --num_workers=16 \
    qc-inst-7b \
    t1.0_n128 \
    checkpoint {CHECKPOINT_PATH} \
    zero_shot_3s10t
```

The following configs correspond to the weak verifiers used:
* [zero_shot](configs/suite/zero_shot.yaml) --- Base
* [zero_shot_syntax](configs/suite/zero_shot_syntax.yaml) --- Syntax
* [zero_shot_lint](configs/suite/zero_shot_lint.yaml) --- Lint
* [zero_shot_3s1t](configs/suite/zero_shot_3s1t.yaml) --- 1 Test
* [zero_shot_3s3t](configs/suite/zero_shot_3s3t.yaml) --- 3 Tests
* [zero_shot_3s10t](configs/suite/zero_shot_3s10t.yaml) --- 10 Tests