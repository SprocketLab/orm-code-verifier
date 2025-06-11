import gc
import gzip
import logging
from dataclasses import asdict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Optional

import ujson
from accelerate import Accelerator
from datasets import Dataset
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from src.evaluation.configs import EvalSuite
from src.evaluation.configs import TaskConfig
from src.evaluation.evaluator import Evaluator
from src.evaluation.preproc import load_raw_eval_dataset
from src.evaluation.preproc import make_icl_examples
from src.evaluation.preproc import process_raw_dataset
from src.metrics import process_task_scores
from src.preprocessing import PreprocessorConfig
from src.scoring import ScoringConfig

logger = logging.getLogger(__name__)

PCT_KEYS = {
    "mrr",
    "b_of_4",
    "b_of_16",
    "b_of_64",
    "b_of_2",
    "rs",
    "f1",
    "precision",
    "recall",
    "pct_collisions",
    "pct_marked_passing",
}

MODELS = [
    "llama3-8b",
    "qwen25-coder-500m",
    "qwen25-coder-1_5b",
    "qwen25-coder-3b",
    "qwen25-coder-7b",
]
KEYS_TO_PRINT = {
    "all/rs",
    "all/b_of_64",
    "all/ncdg@all",
    "pct_collisions",
    "public/rs",
    "all/mrr",
    "pps/inference",
    "pps/get_ds_scores",
    "pps/scoring_function",
    "timing/seconds_per_prog",
    "timing/seconds_per_problem",
}


def evaulate_task(
    generator_model: str,
    sampling_setup: str,
    dataset: Dataset,
    task_name: str,
    suite: EvalSuite,
    evaluator: Evaluator,
    preprocessor_cfg: PreprocessorConfig,
    accelerator: Optional[Accelerator],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_workers: int,
    debug_num: int,
    icl_map: Dict[str, List[str]],
    preproc_batch_size: int,
):
    task_cfg = suite.tasks[task_name]

    start_time = datetime.now(timezone.utc)
    dataset, filter_ds, filter_time = process_raw_dataset(
        generator_model=generator_model,
        sampling_setup=sampling_setup,
        raw_ds=dataset,
        dataset_name=task_cfg.dataset,
        task_cfg=task_cfg,
        preproc_cfg=preprocessor_cfg,
        num_proc=num_workers,
        filter_function=suite.filter_function,
        icl_examples=icl_map.get(task_cfg.icl_examples, []),
        rstrip_completion=suite.rstrip_query,
    )

    timings, scores = evaluator(
        dataset=dataset,
        accelerator=accelerator,
        suite=suite,
        preproc_batch_size=preproc_batch_size,
        model=model,
        tokenizer=tokenizer,
        debug_num=debug_num,
    )
    del dataset
    timings.overall = (datetime.now(timezone.utc) - start_time).total_seconds()
    timings.filtering = filter_time
    return timings, scores, filter_ds


def evaluate_suite(
    generator_model: str,
    sampling_setup: str,
    seed: int,
    scoring_cfg: ScoringConfig,
    suite: EvalSuite,
    preprocessor_cfg: PreprocessorConfig,
    accelerator: Optional[Accelerator],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_workers: int,
    max_tokens_per_batch: int,
    out_directory: Path,
    debug_num_probs: int = None,
    debug_num: int = None,
    preproc_batch_size: int = 5000,
    sort_by_length: bool = False,
    k_vals: Optional[List[int]] = None,
):
    """Evaluates a suite of tasks.

    Args:
        seed (int): Seed to use.
        scoring_cfg (ScoringConfig): The scoring configuration.
        suite (EvalSuite): The evaluation suite.
        preprocessor_cfg (Preprocessor): The preprocessing config.
        model (PreTrainedModel): The model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        num_workers (int): Number of workers.
        max_tokens_per_batch (int): Maximum tokens per batch.
        out_directory (Path): Output directory.
        debug_num_probs (int, optional): Number of programs per problem. Defaults to None.
        debug_num (int, optional): Number of problems. Defaults to None.
        preprocess_batch_size (int, optional): Preprocess batch size. Defaults to 5000.
        sort_by_length (bool, optional): Whether to sort by length. Defaults to False.
        positive_threshold (float, optional): Threshold for positive predictions if
            using a thresholded scoring function. Defaults to 0.25.
    Returns:
        The results and the wandb tables.
    """
    logger.info(f"Evaluating suite '{suite.name}'")
    evaluator = Evaluator(
        seed=seed,
        scoring_cfg=scoring_cfg,
        preprocessor_cfg=preprocessor_cfg,
        num_workers=num_workers,
        max_tokens_per_batch=max_tokens_per_batch,
        sort_by_length=sort_by_length,
    )

    if k_vals is None:
        k_vals = [4, 16, 64]

    icl_map = make_icl_examples(suite, preproc_cfg=preprocessor_cfg)
    logger.info(
        f"Evaluating {len(suite.tasks)} task(s) on {generator_model}-{sampling_setup}"
    )
    results_by_task = {}
    scores_by_task = {}
    for task_name in suite.tasks:
        logger.info(f"Loading dataset for '{task_name}'")
        raw_ds = load_raw_eval_dataset(
            generator_model=generator_model,
            sampling_setup=sampling_setup,
            task_dataset=suite.tasks[task_name].dataset,
            preproc_cfg=preprocessor_cfg,
            debug_num_probs=debug_num_probs,
        )

        timings, scores, filter_ds = evaulate_task(
            generator_model=generator_model,
            sampling_setup=sampling_setup,
            dataset=raw_ds,
            task_name=task_name,
            suite=suite,
            evaluator=evaluator,
            preprocessor_cfg=preprocessor_cfg,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            num_workers=num_workers,
            debug_num=debug_num,
            preproc_batch_size=preproc_batch_size,
            icl_map=icl_map,
        )

        results_by_task[task_name], save_scores = process_task_scores(
            scores=evaluator.group_scores(
                scores=scores, filtered_out=filter_ds
            ),
            timings=asdict(timings),
            k_vals=k_vals,
        )
        del raw_ds, scores, filter_ds
        scores_by_task[task_name] = save_scores

        gc.collect()
    out_results = {}
    logger.info("Results:")
    for task_name, results in sorted(
        results_by_task.items(), key=lambda x: x[0]
    ):
        logger.info("-" * 60)
        logger.info(f"{task_name}")
        for k, v in sorted(results.items(), key=lambda x: x[0]):
            p_val = v
            if any(k.endswith(suffix) for suffix in PCT_KEYS):
                p_val *= 100
            if k in KEYS_TO_PRINT:
                logger.info(f"{k:>32} = {p_val:0.3f}")
            out_results[f"{task_name}/{k}"] = p_val

    logger.info(f"Saving results to {out_directory / 'results.jsonl.gz'}")
    with gzip.open(out_directory / "results.jsonl.gz", "wt") as f:
        for task_name, scores in scores_by_task.items():
            for score in scores:
                f.write(ujson.dumps({"task": task_name, **score}) + "\n")

    return out_results
