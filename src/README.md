# Pareto Optimal Code Generation

Source code for *Pareto Optimal Code Generation*. This contains the implementation for training and evaluating code verification systems using transformer-based outcome reward models with staged verification.

## Overview

This codebase implements Pareto-optimal code generation using outcome reward models (ORMs) and staged verification strategies. It provides tools for:

- Training and evaluating code verification models
- Scoring code solutions using various methods (binary logit, classification, reward modeling)
- Processing and executing code across multiple benchmark datasets
- Comprehensive evaluation metrics and analysis

## Repository Structure

```
src/
├── evaluation/           # Evaluation suite and benchmarks
│   ├── configs.py       # Evaluation configurations
│   ├── evaluator.py     # Core evaluation logic
│   ├── execution.py     # Code execution handling
│   ├── filter_functions.py  # Solution filtering
│   └── prompts.py       # Prompt templates and handling
├── modeling.py          # Model architectures and configurations
├── preprocessing.py     # Data preparation and formatting
├── scoring.py          # Solution scoring implementations
├── metrics.py          # Evaluation metrics
├── utils.py            # Utility functions
└── training/           # Training pipeline components
```

## Core Components

### Modeling (`modeling.py`)
- Model configuration and initialization
- Support for various transformer architectures (GPT-2, Pythia, Qwen, etc.)
- Precision and dropout handling
- Model loading and setup utilities

### Scoring (`scoring.py`)
Multiple scoring methods for code verification:
- Binary logit scoring over sequences
- Classification-based scoring
- Reward model scoring
- Log probability scoring
- Single token scoring variants

### Preprocessing (`preprocessing.py`)
- Formats problems and solutions for model input
- Customizable templates using Jinja2
- Support for in-context learning examples
- Configurable delimiters and formatting options

### Evaluation Suite (`evaluation/`)
Comprehensive evaluation system supporting:
- Multiple benchmark datasets (HumanEval, MBPP, GSM8K, CodeContests)
- Code execution and testing
- Solution filtering and deduplication
- Metric computation and analysis
- Configurable evaluation pipelines

### Training (`training/`)
Training pipeline components for:
- Model training and fine-tuning
- Data handling and batching
- Training configuration

## Key Features

1. **Flexible Scoring Methods**
   - Multiple scoring approaches for different verification strategies
   - Configurable scoring parameters and thresholds
   - Support for both sequence-level and token-level scoring

2. **Comprehensive Evaluation**
   - Support for major code evaluation benchmarks
   - Safe code execution environment
   - Detailed metrics and analysis tools
   - Configurable evaluation suites

3. **Efficient Processing**
   - Solution deduplication
   - Batched processing
   - Parallel execution support
   - Memory-efficient handling of large datasets

4. **Model Support**
   - Compatible with HuggingFace Transformers
   - Support for multiple model architectures
   - Configurable precision and performance options

## Usage

The system is designed to be used by ML researchers and engineers working on code verification systems. Key usage patterns include:

1. **Model Evaluation**
```python
from src.evaluation import evaluate_suite, EvalSuite
from src.modeling import ModelConfig
from src.scoring import ScoringConfig

# Configure evaluation
eval_config = EvalSuite(
    tasks=["humaneval", "mbpp"],
    num_workers=4,
    timeout=30
)

# Set up model and scoring
model_config = ModelConfig(name="pythia-160m")
scoring_config = ScoringConfig(scoring_method="binary_logit")

# Run evaluation
results = evaluate_suite(
    model_config=model_config,
    scoring_config=scoring_config,
    eval_suite=eval_config
)
```

2. **Custom Scoring**
```python
from src.scoring import load_scoring_method
from src.preprocessing import PreprocessorConfig, Preprocessor

# Configure preprocessing
preproc_config = PreprocessorConfig(
    problem="# Question\n{{ problem }}",
    program="# Solution\n{{ program }}",
    outcome="{{ outcome }}"
)

# Initialize scoring
scoring_fn = load_scoring_method(
    scoring_cfg=scoring_config,
    tokenizer=tokenizer,
    pass_choice_str="[Yes]",
    fail_choice_str="[No]"
)

# Score solutions
scores = scoring_fn(dataset, model, max_tokens_per_batch=1024)
```

3. **Evaluation Pipeline**
```python
from src.evaluation import Evaluator
from src.evaluation.execution import process_and_execute_raw_preds

# Initialize evaluator
evaluator = Evaluator(
    seed=42,
    scoring_cfg=scoring_config,
    preprocessor_cfg=preproc_config,
    num_workers=4,
    max_tokens_per_batch=1024
)

# Run evaluation
timings, scores = evaluator(
    dataset=dataset,
    model=model,
    tokenizer=tokenizer,
    suite=eval_config
)
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics including:
- Pass@k scores
- Ranking metrics
- Execution timing statistics
- Solution quality metrics
- Filtering effectiveness measures

Results are saved in a structured format for further analysis and comparison.
