# Data Processing Module

This module handles the generation, downloading, and execution of training and evaluation data. The pipeline is designed to work with large-scale datasets while managing memory constraints through sharded processing.

## Setup

### Requirements

Install the required packages using:

```bash
pip install -r exec_requirements.txt
```

Key dependencies include:
- **Data Processing**: datasets (2.21.0), pandas (2.2.3), numpy (2.2.4)
- **Visualization**: matplotlib (3.10.0)
- **Code Quality**: black (24.8.0), pylint (3.3.4)
- **Math Utilities**: sympy (1.13.3), scipy (1.15.2)

### Custom Dependencies

This module relies on two custom forks with specific functionality:

1. **Simple Code Execution** ([simple-code-execution](https://github.com/gabeorlanski/simple-code-execution.git))
   - Provides core code execution functionality
   - Used at commit `bf0cb87`

2. **EvalPlus** ([evalplus](https://github.com/gabeorlanski/evalplus.git))
   - Modified version of EvalPlus with:
     - Test limiting capabilities
     - Enhanced timing controls
     - Additional execution metrics

## Workflow

The data processing pipeline consists of three main steps:

1. **Generate Data** (`generate.py` in root directory)
   - Creates files for evaluation or training
   - Supports both chat and non-chat formats
   - Configurable sampling parameters
   ```bash
   python generate.py eval-ds [MODEL_NAME] [OPTIONS]
   ```

2. **Download Execution Data** (`download_exec_data.py`)
   - Downloads and processes datasets:
     - DeepMind Code Contests
     - GSM8K (math word problems)
   - Filters and combines train/validation splits
   - Saves in compressed JSONL format
   ```bash
   python download_exec_data.py [OUTPUT_DIR]
   ```

3. **Execute Shards** (`exec_shard.py`)
   - Distributed execution of synthetic data
   - Supports both Code Contests and GSM8K tasks
   - Includes syntax validation and deduplication
   ```bash
   python exec_shard.py [COMMAND] [SHARD_PATH] [DATASET_INFO_PATH]
   ```

## Performance Considerations

The pipeline is designed to handle large-scale data processing with the following optimizations:

1. **Sharded Processing**
   - Generation and execution are split into shards
   - Manages GPU and CPU memory constraints
   - Parallel processing with configurable workers

2. **Resource Management**
   - Configurable timeouts for execution
   - Test limiting to prevent resource exhaustion
   - Deduplication of predictions to reduce computation

## Environment Configuration

The following environment variables can be configured:

- `CUDA_VISIBLE_DEVICES`: Control GPU device selection
- `DEFAULT_OUTPUT_DIR`: Set output directory (defaults to "outputs")
- `TOKENIZERS_PARALLELISM`: Control tokenizer parallelism
- `ACCELERATE_DOWNCAST_BF16`: Control BF16 precision
- `PRECISION`: Set model precision

## Output Structure

Generated data and execution results are organized as follows:

```
outputs/
├── eval_ds_raw_preds/          # Raw model predictions
│   └── [MODEL_NAME]/
│       └── [SAMPLING_PARAMS]/
├── code_contests.jsonl.gz      # Processed code contests data
└── gsm8k.jsonl.gz             # Processed GSM8K data
```
