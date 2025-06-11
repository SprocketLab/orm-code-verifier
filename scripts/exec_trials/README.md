# Execution Trials Module

This module provides tools for benchmarking code execution performance across different models, datasets, and evaluation methods. It supports three types of trials: execution timing, syntax validation, and linting checks.

## Requirements

- Python 3.7+ (with asyncio support)
- Linux/Unix-based OS (Windows not supported)
- Docker (recommended for sandboxed execution)

### Dependencies

```bash
# Core dependencies
datasets==2.21.0 
rich==13.8.0
tqdm==4.66.5
black==24.8.0
click==8.1.7
ujson==5.10.0
numpy==2.2.4
pandas==2.2.3
matplotlib==3.10.0
sympy==1.13.3
scipy==1.15.2
pylint==3.3.4

# Git dependencies
code_execution @ git+https://github.com/gabeorlanski/simple-code-execution.git@bf0cb87ba5987216e5c537028b74bdb18fa382d4
evalplus @ git+https://github.com/gabeorlanski/evalplus.git
```

These are all detailed in [exec_requirements](/scripts/exec_trials/exec_requirements.txt)

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r exec_requirements.txt
   ```


## Usage

### Directory Structure

```
Input Directory:
{input_dir}/
    {dataset}_{model}_{sampling_setup}.jsonl.gz

Output Directory:
{output_dir}/
    {trial_num}.parquet          # Individual trial results
    overall.json                 # Overall trial statistics
    results.parquet             # Final processed results
    log.log                     # Execution logs
```

### Running Trials

The module supports three types of trials:

1. **Execution Timing**
   ```bash
   python run_trials.py {dataset} {model} {sampling_setup} --trial_type exec
   ```

2. **Syntax Validation**
   ```bash
   python run_trials.py {dataset} {model} {sampling_setup} --trial_type syntax
   ```

3. **Pylint Checks**
   ```bash
   python run_trials.py {dataset} {model} {sampling_setup} --trial_type pylint
   ```

### Supported Datasets

- **HumanEval & MBPP**: Evaluated using `evalplus`
- **Code Contests**: Evaluated using `code_execution` with public, private, and generated tests
- **GSM8K**: Evaluated using `code_execution`

### Configuration Parameters

#### Sampling Setup
The sampling setup parameter follows the format: `t{temperature}_n{num_samples}`
- Example: `t1.0_n128` = temperature of 1.0 with 128 samples per problem
- Supported configurations:
  - `t1.0_n128` (default)
  - `t0.2_n128`
  - `t0.4_n128`
  - `t0.6_n128`
  - `t0.8_n128`
  - `t1.0_n256`

#### Timeout Parameters

1. Code Execution Timeouts:
   - `--first_command_timeout`: Initial command execution timeout (default: 30s)
   - `--command_timeout`: Subsequent commands timeout (default: 10s)

2. EvalPlus Specific Timeouts:
   - `--ep_min_timeout`: Minimum timeout (default: 1.0s)
   - `--ep_gt_timeout_factor`: Timeout factor (default: 4.0)
   - `--ep_max_timeout`: Maximum timeout (optional)

### Additional Options

- `--num_trials`: Number of evaluation trials to run (default: 5)
- `--num_workers`: Number of parallel workers (default: 1)
- `--max_tests`: Maximum number of tests to run per problem
- `--force_timeout`: Force timeout values even if problem has defined limits
- `--run_all_tests`: Run all tests instead of stopping on first failure
- `--cleanup_trial_files`: Remove individual trial files after completion

## Processing Results

Use `clean_filter_data.py` to process and analyze trial results:

```bash
python clean_filter_data.py {filter_data_path} {output_path}
```

This will:
1. Aggregate results across trials
2. Calculate timing statistics
3. Generate summary metrics
4. Save processed results in parquet format

Run trials in container:
   ```bash
   ./trial.sh {dataset} {model} {sampling_setup}
   ```


## Security Considerations

1. Always run untrusted code in a sandboxed environment
2. Monitor resource usage (memory, CPU)
3. Set appropriate timeouts to prevent infinite loops
4. Consider additional isolation measures beyond Docker if running untrusted code 