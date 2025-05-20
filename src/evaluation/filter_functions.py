"""Module for running filter functions on ranking datasets."""

import functools
import io
import logging
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import pylint
import pylint.lint
import ujson
from code_execution import is_valid_python
from code_execution.eval_dataset import code_contests
from code_execution.eval_dataset import gsm8k
from datasets import Dataset
from datasets import load_dataset
from evalplus.eval import PASS
from evalplus.eval import TIMEOUT
from evalplus.evaluate import evaluate as evalplus
from pylint.reporters.text import TextReporter
from tqdm import tqdm
import gzip
import pandas as pd

logger = logging.getLogger(__name__)
LINE_WIDTH = 120
FILTER_DATA_LOC = Path(__file__).parents[2] / "data" / "filter_data"

EXEC_FILTERS = {}
FILTER_DF = pd.DataFrame()
TIMING_DATA = pd.read_csv(
    FILTER_DATA_LOC / "filter_timings_summary.csv"
).set_index(["generator", "sampling_setup", "dataset", "filter_name"])

for m_dir in FILTER_DATA_LOC.glob("*"):
    model_name, sampling_setup = m_dir.name.split(".", 1)
    if model_name not in EXEC_FILTERS:
        EXEC_FILTERS[model_name] = {}
    if sampling_setup not in EXEC_FILTERS[model_name]:
        EXEC_FILTERS[model_name][sampling_setup] = {}
    for d_file in m_dir.glob("*.csv.gz"):
        FILTER_DF = pd.concat(
            [FILTER_DF, pd.read_csv(d_file, compression="gzip")]
        )

FILTER_DF.set_index(["generator", "sampling_setup", "dataset"], inplace=True)
FILTER_DF.sort_index(inplace=True)


def run_filter_on_dataset(
    model: str,
    sampling_setup: str,
    dataset: Dataset,
    dataset_name: str,
    num_procs: int,
    filter_function: Optional[str] = None,
    remove_timeouts: bool = False,
) -> Tuple[Dataset, Dict[Tuple, Dict], float]:
    expected_num_solutions = sum(
        sum(c["count"] for c in r["solutions"]) for r in dataset
    )
    logger.info(f"Running filter function: {filter_function}")
    logger.info(f"Initial dataset size: {len(dataset)}")
    logger.debug(f"Dataset columns: {dataset.column_names}")
    logger.debug(f"{num_procs=}")
    if filter_function.startswith("execute_"):
        filter_name = "e" + filter_function.split("_")[-1]
        if dataset_name == "gsm8k":
            logger.info("GSM8K dataset detected, skipping filter")
            return dataset, {}, 0
    else:
        filter_name = filter_function
    filtered_data = (
        FILTER_DF.loc[
            pd.IndexSlice[model, sampling_setup, dataset_name],
            ["task_id", "sid", filter_name, f"{filter_name}_timeout"],
        ]
        .copy()
        .reset_index()
        .drop(columns=["generator", "sampling_setup", "dataset"])
    )

    if pd.isna(filtered_data).any().any():
        raise ValueError(
            f"Missing data for filter {filter_name} on {model} {sampling_setup} {dataset_name}"
        )
    if remove_timeouts:
        filtered_data["keep"] = filtered_data[filter_name]
    else:
        filtered_data["keep"] = (
            filtered_data[filter_name] | filtered_data[f"{filter_name}_timeout"]
        )
    filtered_data.drop(
        columns=[filter_name, f"{filter_name}_timeout"], inplace=True
    )
    filtered_data.set_index(["task_id", "sid"], inplace=True)
    filter_time = TIMING_DATA.loc[
        pd.IndexSlice[model, sampling_setup, dataset_name, filter_name], "time"
    ]

    logger.info("Gathering the filtering results...")

    def move_solutions(problem: Dict) -> Dict:
        filtered = []
        kept = []

        for s in problem["solutions"]:
            res = filtered_data.loc[
                pd.IndexSlice[problem["task_id"], s["sid"]], "keep"
            ]
            if res:
                kept.append(s)
            else:
                filtered.append(s)
        return {
            "filtered": filtered,
            "solutions": kept,
        }

    dataset = dataset.map(
        move_solutions,
        load_from_cache_file=False,
        desc=f"{filter_name}",
    )

    filtered_out = (
        dataset.filter(
            lambda x: len(x["filtered"]) if x["filtered"] else False,
            num_proc=num_procs,
            desc="Gathering filtered out solutions",
        )
        .remove_columns(["solutions", "problem"])
        .rename_column("filtered", "solutions")
    )
    num_filtered = sum(
        sum(c["count"] for c in r["solutions"]) for r in filtered_out
    )
    dataset = dataset.filter(
        lambda x: len(x["solutions"]) if x["solutions"] else False,
        num_proc=num_procs,
        desc="Gathering kept solutions",
    ).remove_columns(["filtered"])
    num_kept = sum(sum(c["count"] for c in r["solutions"]) for r in dataset)
    num_unique_solutions = sum(len(r["solutions"]) for r in dataset)
    logger.info(
        f"Filtered dataset size: {len(dataset):,} "
        f"({num_kept:,} solutions, {num_unique_solutions:,} unique solutions)"
    )

    logger.debug(
        f"Number of problems in filtered out: {len(filtered_out)} ({sum(sum(c['count'] for c in r['solutions']) for r in filtered_out):,} solutions)"
    )
    logger.info(
        f"Expected number of solutions: {expected_num_solutions:,} "
        f"({num_filtered:,} + {num_kept:,} = {num_filtered + num_kept:,} )"
    )
    return dataset, filtered_out, filter_time
