import functools
import gzip
import logging
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pylint
import pylint.lint
import click
import numpy as np
import pandas as pd
import ujson
from code_execution.eval_dataset import code_contests
from code_execution.eval_dataset import gsm8k
from code_execution.code_trees import is_valid_python
from code_execution.utils import run_in_parallel
from code_execution import Executable, execute_predictions, ExecutionConfig
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval import NOT_RAN
from evalplus.eval import PASS
from evalplus.eval import TIMEOUT
from evalplus.evaluate import evaluate as evalplus
from evalplus.sanitize import sanitize
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
import time
from collections import defaultdict
from utils import FINAL_RESULTS_FILE
from utils import OVERALL_FILE
from utils import PROG_RESULTS_FILE
from utils import CodeContestsProblem
from utils import EvalPlusProblem
from utils import GSM8KProblem
from utils import Problem
from utils import extract_imports
from utils import get_builtin_modules
from utils import process_raw_results
import io
from pylint.reporters.text import TextReporter
import re

CONSOLE = Console(width=156)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("black").setLevel(logging.WARNING)
logging.getLogger("blib2to3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

LINE_WIDTH = 80
HE_TESTS = {
    k: (len(v["base_input"]), len(v["plus_input"]))
    for k, v in get_human_eval_plus().items()
}
MBPP_TESTS = {
    k: (len(v["base_input"]), len(v["plus_input"]))
    for k, v in get_mbpp_plus().items()
}

GSM8K_COLUMNS = [
    "trial",
    "task_id",
    "sid",
    "timeout",
    "num_ran",
    "had_error",
    "return_code",
    "passed",
    "writing_time",
    "execution_time",
    "cleanup_time",
    "preprocess_time",
    "postprocess_time",
]

CODE_CONTESTS_COLUMNS = [
    "trial",
    "task_id",
    "sid",
    "timeout",
    "num_ran",
    "had_error",
    "return_code",
    "passed",
    "passed_public",
    "passed_private",
    "passed_generated",
    "writing_time",
    "execution_time",
    "cleanup_time",
    "preprocess_time",
    "postprocess_time",
    "n_test_timings",
    "num_passed",
]


@dataclass
class Config:
    """Configuration class for execution timing evaluation.

    Attributes:
        dataset (str): Dataset to evaluate
        model (str): Model to evaluate
        sampling_setup (str): Sampling setup to evaluate
        output_dir (Path): Directory to save evaluation results
        num_trials (int): Number of evaluation trials to run
        num_workers (int): Number of parallel workers for evaluation
        max_tests (int, optional): Maximum number of tests to run per problem
        first_command_timeout (float): Timeout for first command execution in seconds
        command_timeout (float): Timeout for subsequent commands in seconds
        force_timeout (bool): Force the timeout amounts to be used even if problem timeout exists.
        ep_min_timeout (float): Minimum timeout for evalplus
        ep_gt_timeout_factor (float): Timeout factor for evalplus
        ep_max_timeout (float, optional): Maximum timeout for evalplus
        ep_plus_only (bool): Only run evalplus with plus tests
        run_all_tests (bool): Run all tests for all datasets
    """

    dataset: str
    model: str
    sampling_setup: str
    output_dir: Path
    num_trials: int
    num_workers: int
    max_tests: Optional[int] = None
    first_command_timeout: float = 10.0
    command_timeout: float = 5.0
    force_timeout: bool = False
    ep_min_timeout: float = 1.0
    ep_gt_timeout_factor: float = 4.0
    ep_max_timeout: Optional[float] = None
    ep_plus_only: bool = False
    run_all_tests: bool = False

    def get_run_info(self) -> Dict:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "sampling_setup": self.sampling_setup,
            "num_workers": self.num_workers,
            "max_tests": self.max_tests,
            "first_command_timeout": self.first_command_timeout,
            "command_timeout": self.command_timeout,
            "force_timeout": self.force_timeout,
            "ep_min_timeout": self.ep_min_timeout,
            "ep_gt_timeout_factor": self.ep_gt_timeout_factor,
            "ep_max_timeout": self.ep_max_timeout,
            "ep_plus_only": self.ep_plus_only,
            "run_all_tests": self.run_all_tests,
        }


@dataclass
class ExecutionResult:
    """Stores execution results and timing information for a single solution.

    Attributes:
        task_id (str): Unique identifier for the task/problem
        sid (str): Solution identifier
        timeout (bool): Whether the execution timed out
        had_error (bool): Whether any errors occurred during execution
        return_code (int): Process return code from execution
        timing (Dict[str, float]): Dictionary of timing measurements for different execution phases
        num_ran (int): Number of tests run
        passed (bool): Whether the solution passed all tests
        passed_public (bool): Whether the solution passed public tests
        passed_private (bool): Whether the solution passed private tests
        passed_generated (bool): Whether the solution passed generated tests
        num_passed (int): Number of tests passed
    """

    task_id: str
    sid: str
    timeout: bool
    had_error: bool
    return_code: int
    timing: Dict[str, float]
    num_ran: int
    num_passed: int
    passed: bool
    passed_public: bool = None
    passed_private: bool = None
    passed_generated: bool = None


def _process_trial_results(cfg: Config, trial_results: Dict) -> None:
    """Process and log the trial results statistics.

    Args:
        output_dir (Path): Directory to save results
        cfg (Config): Configuration object
        trial_results (Dict): Dictionary containing timing results for each trial
    """
    df = pd.DataFrame.from_dict(trial_results, orient="index")
    num_preds = df["num_preds"].iloc[0]

    time_columns = [c for c in df.columns if c.endswith("_time")]

    pps_means = df[time_columns].mean()

    pps_means = num_preds / pps_means
    logger.info("Mean PPS:")
    for k, v in pps_means.to_dict().items():
        logger.info(f"{k:>24}={v:0.2f}")

    with open(cfg.output_dir / OVERALL_FILE, "w") as fd:
        ujson.dump(trial_results, fd, indent=2)


def single_trial(
    cfg: Config,
    trial: int,
    evaluator: Callable,
    row_process_fn: Callable,
) -> tuple[dict, int, list[dict]]:
    """Execute a single trial and process its results.

    Args:
        trial: Current trial number
        processed_problems: List of problems to evaluate
        evaluator: Function to evaluate the problems
        csv_writer: CSV writer for recording results
        cfg: Configuration object
        rng: Random number generator

    Returns:
        tuple of (trial results, number of predictions processed, list of mismatched results)
    """
    num_preds = 0

    trial_results, results = evaluator()

    logger.info(f"Saving results for trial {trial}")
    builtin_modules = get_builtin_modules()
    rows = []
    # Process results
    for problem in results:
        for solution in problem["predictions"]:
            num_preds += 1

            # Create ExecutionResult instance
            result = ExecutionResult(
                task_id=problem["task_id"],
                sid=solution["sid"],
                timeout=solution["timeout"],
                had_error=solution["had_error"],
                return_code=solution["return_code"],
                timing=solution["timing"],
                passed=solution["passed"],
                passed_public=solution.get("passed_public"),
                passed_private=solution.get("passed_private"),
                passed_generated=solution.get("passed_generated"),
                num_ran=solution.get("num_ran", 1),
                num_passed=sum(solution.get("outcomes", [])),
            )
            rows.append(row_process_fn(result, trial_num=trial))

    pd.DataFrame(rows).to_parquet(cfg.output_dir / f"{trial}.parquet")
    trial_results["programs_passed"] = sum([row["passed"] for row in rows])
    trial_results["num_preds"] = num_preds
    return trial_results


def run_trials(
    cfg: Config,
    evaluator: Callable,
    row_process_fn: Callable,
    pps_func: Callable,
    dataset_name: str,
) -> Dict:
    """Execute multiple trials and save results to CSV.

    Args:
        cfg: Configuration object
        processed_problems: List of problems to evaluate
        evaluator: Function to evaluate the problems
        row_process_fn: Function to process each result row
        pps_func: Function to calculate PPS
        dataset_name: Name of dataset to evaluate
    Returns:
        Dict containing results from all trials
    """

    trial_results = {}
    for trial in range(1, cfg.num_trials + 1):
        logger.info(
            f"Running trial {trial}/{cfg.num_trials} for {dataset_name}"
        )
        trial_results[trial] = single_trial(
            trial=trial,
            evaluator=evaluator,
            row_process_fn=row_process_fn,
            cfg=cfg,
        )

        trial_results[trial]["pps"] = pps_func(trial_results[trial])
        trial_results[trial].update(cfg.get_run_info())

    # Process results using the new function
    processed_df = process_raw_results(cfg.output_dir)
    # Save processed results
    processed_results_file = cfg.output_dir / FINAL_RESULTS_FILE
    processed_df.to_parquet(processed_results_file, index=False)

    _process_trial_results(cfg=cfg, trial_results=trial_results)
    return trial_results


def _code_exec_default_process(
    result: ExecutionResult, trial_num: int, columns: List[str]
) -> Dict:
    row = {k: v for k, v in asdict(result).items() if k in columns}
    row["trial"] = trial_num
    for k, v in result.timing.items():
        if k in {
            "writing",
            "preprocess",
            "postprocess",
            "cleanup",
            "execution",
        }:
            row[f"{k}_time"] = v

    return row


def parse_pylint_output(pylint_output: str) -> Dict[str, Optional[float]]:
    """
    Parse the output from running pylint on a Python program, extracting just
    the score and counts of warnings/errors.

    Args:
        pylint_output (str): The complete output text from running pylint

    Returns:
        Dict: A dictionary containing:
            - 'score' (float): The numerical score (0-10) or None if not found
            - 'error_count' (int): Number of errors (E) found
            - 'warning_count' (int): Number of warnings (W) found
            - 'convention_count' (int): Number of convention issues (C) found
            - 'refactor_count' (int): Number of refactor suggestions (R) found
            - 'fatal_count' (int): Number of fatal errors (F) found
    """
    # Initialize result dictionary
    result = {
        "score": None,
        "error_count": 0,
        "warning_count": 0,
        "convention_count": 0,
        "refactor_count": 0,
        "fatal_count": 0,
    }

    # Extract score using regex - usually appears at the end in format "Your code has been rated at X.XX/10"
    score_match = re.search(
        r"Your code has been rated at (\d+\.\d+)/10", pylint_output
    )
    if score_match:
        result["score"] = float(score_match.group(1))

    # Count different message types
    error_count = len(re.findall(r":[0-9]+:[0-9]+: E[0-9]+:", pylint_output))
    warning_count = len(re.findall(r":[0-9]+:[0-9]+: W[0-9]+:", pylint_output))
    convention_count = len(
        re.findall(r":[0-9]+:[0-9]+: C[0-9]+:", pylint_output)
    )
    refactor_count = len(re.findall(r":[0-9]+:[0-9]+: R[0-9]+:", pylint_output))
    fatal_count = len(re.findall(r":[0-9]+:[0-9]+: F[0-9]+:", pylint_output))

    result["error_count"] = error_count
    result["warning_count"] = warning_count
    result["convention_count"] = convention_count
    result["refactor_count"] = refactor_count
    result["fatal_count"] = fatal_count

    return result


def has_no_lint_errors(
    args: Tuple,
) -> Tuple[str, str, str, float, int]:
    """Check if code passes pylint with fewer errors than threshold."""
    # Create a string IO to capture pylint output
    task_id, sid, solution = args
    pylint_output = io.StringIO()
    reporter = TextReporter(pylint_output)

    # Run pylint with some arguments to make it more lenient
    args = [
        "--disable=C0111,C0103,C0303,W0621,C0301,C0304,C0200,R0903,R0913,R0914,W0702",  # Disable some too-strict checks
        f"--max-line-length={LINE_WIDTH}",
    ]

    start_time = time.time()
    # Create a temporary file to run pylint on
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w+", delete=False
    ) as tmp_file:
        code = solution

        tmp_file.write(code)
        tmp_file_path = tmp_file.name

    try:
        # Run pylint on the file
        pylint.lint.Run([tmp_file_path] + args, reporter=reporter, exit=False)
        output = pylint_output.getvalue()
        had_error = False
    except Exception as e:
        output = str(e)
        had_error = True

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
    elapsed = time.time() - start_time
    if had_error:
        return (task_id, sid, solution, elapsed, 100)

    parse_info = parse_pylint_output(output)

    return (
        task_id,
        sid,
        solution,
        elapsed,
        parse_info["error_count"] + parse_info["fatal_count"],
    )


def execute_pylint(
    rng: np.random.RandomState, dataset: List[Problem], cfg: Config
) -> Dict:
    """Execute pylint on a dataset of problems."""
    logger.info("Executing pylint on dataset.")

    def evaluate():
        progs = [
            (p.task_id, s.sid, s.solution) for p in dataset for s in p.solutions
        ]
        rng.shuffle(progs)
        start = time.time()
        results = run_in_parallel(
            has_no_lint_errors,
            progs,
            num_workers=cfg.num_workers,
            desc="Pylint Trial",
        )
        end = time.time()
        out = defaultdict(list)
        total_found = 0
        num_passed = 0
        for tid, sid, solution, s_elapsed, error_count in results:
            num_passed += error_count == 0
            out[tid].append(
                {
                    "sid": sid,
                    "solution": solution,
                    "had_error": error_count > 0,
                    "return_code": error_count,
                    "timeout": False,
                    "timing": {"execution": s_elapsed},
                    "passed": error_count == 0,
                    "num_passed": 1 if error_count == 0 else 0,
                    "num_ran": 1,
                }
            )
            total_found += 1
        results = [{"task_id": tid, "predictions": out[tid]} for tid in out]
        metrics = {
            "net_time": end - start,
            "pct_passed": num_passed / total_found,
        }
        return metrics, results

    def make_pylint_row(exec_result: ExecutionResult, trial_num: int):
        out = {
            "trial_num": trial_num,
            "task_id": exec_result.task_id,
            "sid": exec_result.sid,
            "had_error": exec_result.had_error,
            "passed": exec_result.passed,
            "timeout": exec_result.timeout,
            "num_errors": exec_result.return_code,
            "elapsed": exec_result.timing["execution"],
        }
        return out

    trial_results = run_trials(
        cfg=cfg,
        evaluator=evaluate,
        row_process_fn=make_pylint_row,
        pps_func=lambda m: m["num_preds"] / m["net_time"],
        dataset_name="pylint",
    )
    return trial_results


def is_solution_valid(args: Tuple) -> Tuple[str, str, str, bool, int]:
    task_id, sid, solution = args
    start = time.time()
    try:
        is_valid = is_valid_python(solution)
    except Exception as e:
        is_valid = False
    elapsed = time.time() - start

    return (
        task_id,
        sid,
        solution,
        elapsed,
        is_valid,
    )


def execute_syntax(
    rng: np.random.RandomState, dataset: List[Problem], cfg: Config
) -> Dict:
    """Execute pylint on a dataset of problems."""
    logger.info("Executing pylint on dataset.")

    def evaluate():
        progs = [
            (p.task_id, s.sid, s.solution) for p in dataset for s in p.solutions
        ]
        rng.shuffle(progs)
        start = time.time()
        results = run_in_parallel(
            is_solution_valid,
            progs,
            num_workers=cfg.num_workers,
            desc="Syntax Trial",
        )
        end = time.time()
        out = defaultdict(list)
        total_found = 0
        num_passed = 0
        for tid, sid, solution, s_elapsed, is_valid in results:
            out[tid].append(
                {
                    "sid": sid,
                    "solution": solution,
                    "had_error": not is_valid,
                    "return_code": 0 if is_valid else 1,
                    "timeout": False,
                    "timing": {"execution": s_elapsed},
                    "passed": is_valid,
                    "num_passed": is_valid,
                    "num_ran": 1,
                }
            )
            num_passed += is_valid
            total_found += 1
        results = [{"task_id": tid, "predictions": out[tid]} for tid in out]
        metrics = {
            "net_time": end - start,
            "pct_passed": num_passed / total_found,
        }
        return metrics, results

    def make_row(exec_result: ExecutionResult, trial_num: int):
        out = {
            "trial_num": trial_num,
            "task_id": exec_result.task_id,
            "sid": exec_result.sid,
            "passed": exec_result.passed,
            "timeout": exec_result.timeout,
            "num_errors": exec_result.return_code,
            "elapsed": exec_result.timing["execution"],
        }
        return out

    trial_results = run_trials(
        cfg=cfg,
        evaluator=evaluate,
        row_process_fn=make_row,
        pps_func=lambda m: m["num_preds"] / m["net_time"],
        dataset_name="syntax",
    )
    return trial_results


def execute_code_contests(
    rng: np.random.RandomState, cfg: Config, dataset: List[CodeContestsProblem]
) -> Dict:
    """Execute Code Contests evaluation.

    Args:
        rng: Random number generator for shuffling
        cfg: Configuration object
        dataset: List of CodeContestsProblem

    Returns:
        Dict: Results from all trials
    """
    logger.info("Executing Code Contests")

    logger.info("Creating executable predictions")

    processed_problems = list(
        map(
            lambda x: code_contests.process_problem(x, max_tests=cfg.max_tests),
            map(asdict, dataset),
        )
    )

    def process_row(result: ExecutionResult, trial_num: int) -> Dict:

        return _code_exec_default_process(
            result, trial_num, CODE_CONTESTS_COLUMNS
        )

    def evaluate():
        rng.shuffle(processed_problems)
        for pred in processed_problems:
            rng.shuffle(pred["solutions"])
        return code_contests.evaluate(
            predictions=processed_problems,
            num_workers=cfg.num_workers,
            first_command_timeout=cfg.first_command_timeout,
            command_timeout=cfg.command_timeout,
            num_stdout_save=1,
            early_stopping=not cfg.run_all_tests,
            execution_kwargs={
                "log_freq": 2500,
                "buffer_size": 100,
            },
            force_command_timeout=cfg.force_timeout,
        )

    trial_results = run_trials(
        cfg=cfg,
        evaluator=evaluate,
        row_process_fn=process_row,
        pps_func=lambda m: m["num_preds"]
        / (
            m["execution_time"]
            + m["preprocessing_time"]
            + m["postprocessing_time"]
        ),
        dataset_name="code_contests",
    )

    return trial_results


def execute_evalplus(
    rng: np.random.RandomState,
    dataset: List[EvalPlusProblem],
    cfg: Config,
    source: str,
) -> Dict:
    """Processes evalplus results for a given task."""
    logger.info("Using evalplus to evaluate predictions.")

    def write_predictions(pred_directory: Path):
        rng.shuffle(dataset)
        with open(pred_directory / "samples.jsonl", "w", encoding="utf-8") as f:
            for i, row in enumerate(dataset):
                task_id = row.task_id
                for j, pred in enumerate(row.solutions):
                    f.write(
                        ujson.dumps(
                            {
                                "completion_id": j,
                                "completion": pred.solution,
                                "task_id": task_id,
                                "meta": {
                                    "completion_id": pred.sid,
                                    "row_id": i,
                                    "pred_id": j,
                                },
                            }
                        )
                        + "\n"
                    )

    def evaluate():
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            logger.debug(f"Created temporary directory: {tmp_dir}")
            logger.debug("Writing predictions to temporary directories.")
            write_predictions(tmp_dir)

            evalplus(
                dataset=source,
                samples=str((tmp_dir / "samples.jsonl").absolute()),
                parallel=cfg.num_workers,
                min_time_limit=cfg.ep_min_timeout,
                gt_time_limit_factor=cfg.ep_gt_timeout_factor,
                max_test_cases=cfg.max_tests,
                max_time_limit=cfg.ep_max_timeout,
                plus_only=cfg.ep_plus_only,
                disable_cache=False,
                test_details=cfg.run_all_tests,
            )

            logger.info("Reading evalplus results.")
            with open(
                tmp_dir / "samples.eval_results.json", encoding="utf-8"
            ) as f:
                eval_results = ujson.load(f)

        out = []
        num_preds = 0
        for tid, res in eval_results["eval"].items():
            sol_list = []
            for sol in res:
                num_preds += 1
                passed_base = sol["base_status"] == PASS
                passed_plus = sol["plus_status"] == PASS
                if sol["base_status"] == NOT_RAN:
                    passed = passed_plus
                    outcomes = sol["plus_details"]
                elif sol["plus_status"] == NOT_RAN:
                    passed = passed_base
                    outcomes = sol["base_details"]
                else:
                    outcomes = sol["plus_details"] + sol["base_details"]
                    passed = passed_base and passed_plus
                num_ran = len(outcomes)
                num_passed = sum(outcomes)
                sol_list.append(
                    {
                        "sid": sol["completion_id"],
                        "solution": sol["solution"],
                        "timeout": sol["base_status"] == TIMEOUT
                        or sol["plus_status"] == TIMEOUT,
                        "had_error": False,
                        "return_code": 0,
                        "timing": {
                            "execution": sol["elapsed"],
                            "plus_exec": sum(sol["plus_timings"]),
                            "base_exec": sum(sol["base_timings"]),
                        },
                        "passed": passed,
                        "passed_public": passed_base,
                        "passed_private": passed_plus,
                        "outcomes": outcomes,
                        "num_ran": num_ran,
                        "num_passed": num_passed,
                    }
                )
            out.append(
                {
                    "task_id": tid,
                    "predictions": sol_list,
                }
            )

        metrics = {
            "net_time": eval_results["execution_elapsed"]
            + eval_results["get_gt_elapsed"],
            "execution_time": eval_results["execution_elapsed"],
            "gt_time": eval_results["get_gt_elapsed"],
            **{
                f"{s}_{kk}": vv
                for s, v in eval_results["pass_at_k"].items()
                for kk, vv in v.items()
            },
        }
        return metrics, out

    def make_evalplus_row(exec_result: ExecutionResult, trial_num: int):
        out = asdict(exec_result)
        out.update({f"{k}_time": v for k, v in exec_result.timing.items()})
        out["trial"] = trial_num
        return out

    trial_results = run_trials(
        cfg=cfg,
        evaluator=evaluate,
        row_process_fn=make_evalplus_row,
        pps_func=lambda m: m["num_preds"] / m["execution_time"],
        dataset_name=source,
    )

    return trial_results


def execute_gsm8k(
    rng: np.random.RandomState, cfg: Config, dataset: List[GSM8KProblem]
) -> Dict:
    """Execute GSM8K evaluation.

    Args:
        rng: Random number generator for shuffling
        cfg: Configuration object
        dataset: Dataset containing GSM8K problems

    Returns:
        Dict: Results from all trials
    """
    processed_predictions = []
    logger.info("Creating executable predictions")
    for prob in tqdm(dataset, desc="Getting Info"):
        new_dict = {
            "task_id": prob.task_id,
            "answer": prob.answer,
        }
        new_dict["solutions"] = [asdict(d) for d in prob.solutions]
        processed_predictions.append(new_dict)

    def evaluate():
        rng.shuffle(processed_predictions)
        for pred in processed_predictions:
            rng.shuffle(pred["solutions"])
        return gsm8k.evaluate(
            predictions=processed_predictions,
            num_workers=cfg.num_workers,
            timeout=cfg.first_command_timeout,
            execution_kwargs={
                "log_freq": 2500,
                "buffer_size": 100,
            },
        )

    trial_results = run_trials(
        cfg=cfg,
        evaluator=evaluate,
        row_process_fn=functools.partial(
            _code_exec_default_process, columns=GSM8K_COLUMNS
        ),
        pps_func=lambda m: m["num_preds"]
        / (
            m["execution_time"]
            + m["preprocessing_time"]
            + m["postprocessing_time"]
        ),
        dataset_name="gsm8k",
    )
    return trial_results


@click.command()
@click.argument(
    "ds_name",
    type=click.Choice(["gsm8k", "code_contests", "humaneval", "mbpp"]),
)
@click.argument(
    "model_name",
    type=click.Choice(
        [
            "qc-inst-500m",
            "qc-inst-3b",
            "qc-inst-1_5b",
            "qc-inst-7b",
            "qc-inst-14b",
        ]
    ),
)
@click.argument(
    "sampling_setup",
    type=click.Choice(
        [
            "t1.0_n128",
            "t0.2_n128",
            "t0.4_n128",
            "t0.6_n128",
            "t0.8_n128",
            "t1.0_n256",
        ]
    ),
    default="t1.0_n128",
)
@click.option(
    "--input_dir",
    type=click.Path(dir_okay=True, path_type=Path, file_okay=False),
    default=None,
)
@click.option(
    "--output_dir",
    type=click.Path(dir_okay=True, path_type=Path, file_okay=False),
    default=None,
)
@click.option("--num_trials", type=int, default=5)
@click.option("--num_workers", type=int, default=1)
@click.option(
    "--max_tests",
    type=lambda x: int(x) if x.strip() != "None" else None,
    default=None,
)
@click.option("--seed", type=int, default=1)
@click.option("--command_timeout", type=float, default=10.0)
@click.option("--first_command_timeout", type=float, default=30.0)
@click.option("--ep_min_timeout", type=float, default=1.0)
@click.option("--ep_gt_timeout_factor", type=float, default=4.0)
@click.option("--ep_max_timeout", type=float, default=None)
@click.option("--ep_plus_only", is_flag=True)
@click.option("--debug_num", type=int, default=None)
@click.option("--debug_num_sols", type=int, default=None)
@click.option("--force_timeout", is_flag=True)
@click.option("--cleanup_trial_files", "-cleanup", is_flag=True)
@click.option("--run_all_tests", "-full", is_flag=True)
@click.option(
    "--trial_type",
    type=click.Choice(["pylint", "exec", "syntax"]),
    default="exec",
)
def cli(
    ds_name: str,
    model_name: str,
    sampling_setup: str,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    num_trials: int,
    num_workers: int,
    max_tests: Optional[int],
    debug_num: int,
    command_timeout: float,
    first_command_timeout: float,
    ep_min_timeout: float,
    ep_gt_timeout_factor: float,
    ep_max_timeout: float,
    ep_plus_only: bool,
    seed: int,
    debug_num_sols: int,
    force_timeout: bool,
    cleanup_trial_files: bool,
    run_all_tests: bool,
    trial_type: str,
):
    """Command line interface for execution timing evaluation.

    Args:
        ds_name: Name of dataset to evaluate ('gsm8k' or 'code_contests')
        model_name: Name of model to evaluate
        sampling_setup: Sampling setup to use
        input_dir: Directory to read input from
        output_dir: Directory to save results
        num_trials: Number of evaluation trials
        num_workers: Number of parallel workers
        max_tests: Maximum number of tests per problem
        debug_num: Number of problems to use for debugging
        command_timeout: Timeout for commands in seconds
        first_command_timeout: Timeout for first command in seconds
        seed: Random seed
        debug_num_sols: Number of solutions to use for debugging
        force_timeout: Force timeout for all problems
        cleanup_trial_files: Cleanup trial files
        ep_min_timeout: Minimum timeout for evalplus
        ep_gt_timeout_factor: Timeout factor for evalplus
        ep_max_timeout: Maximum timeout for evalplus
        ep_plus_only: Only run plus tests for evalplus
        trial_type: Type of trial to run
    """
    if output_dir is None:
        output_dir = Path("outputs", "exec_timing")
        save_dir = output_dir.resolve() / ".".join(
            (
                model_name,
                ds_name,
                f"t{num_trials}_n{num_workers}{'_m'+str(max_tests ) if max_tests is not None else ''}",
            )
        )
    else:
        save_dir = output_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        save_dir / "log.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(console=CONSOLE, level=logging.INFO),
            file_handler,
        ],
    )

    logger.info(f"Dataset: {ds_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Sampling setup: {sampling_setup}")
    logger.info(f"Output dir: {save_dir.absolute()}")

    if input_dir is None:
        input_dir = Path(__file__).parent
    input_file = input_dir / f"{ds_name}_{model_name}_{sampling_setup}.jsonl.gz"
    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist")

    problem_class = None
    if ds_name == "code_contests":
        problem_class = CodeContestsProblem
    elif ds_name == "gsm8k":
        problem_class = GSM8KProblem
    elif ds_name in {"humaneval", "mbpp"}:
        problem_class = EvalPlusProblem
    else:
        raise NotImplementedError(ds_name)

    with gzip.open(input_file, "rt") as f:
        dataset = [
            problem_class.from_dict(line) for line in map(ujson.loads, f)
        ]

    rng = np.random.RandomState(seed)
    if debug_num is not None:
        rng.shuffle(dataset)
        dataset = dataset[:debug_num]
    if debug_num_sols is not None:
        for i in range(len(dataset)):
            rng.shuffle(dataset[i].solutions)
            dataset[i].solutions = dataset[i].solutions[:debug_num_sols]
    logger.info(f"Dataset size: {len(dataset)}")
    cfg = Config(
        dataset=ds_name,
        model=model_name,
        sampling_setup=sampling_setup,
        output_dir=save_dir,
        num_trials=num_trials,
        num_workers=num_workers,
        max_tests=max_tests if max_tests != -1 else None,
        first_command_timeout=first_command_timeout,
        command_timeout=command_timeout,
        force_timeout=force_timeout,
        ep_min_timeout=ep_min_timeout,
        ep_gt_timeout_factor=ep_gt_timeout_factor,
        ep_max_timeout=ep_max_timeout,
        ep_plus_only=ep_plus_only,
        run_all_tests=run_all_tests,
    )
    logger.info(f"Creating RNG for seed {seed}")
    if trial_type == "exec":
        if ds_name == "code_contests":
            execute_code_contests(rng, cfg, dataset)
        elif ds_name == "gsm8k":
            execute_gsm8k(rng, cfg, dataset)
        elif ds_name in {"humaneval", "mbpp"}:
            execute_evalplus(
                rng,
                cfg=cfg,
                dataset=dataset,
                source=ds_name,
            )

        else:
            raise NotImplementedError(ds_name)
    elif trial_type == "pylint":
        execute_pylint(rng, dataset, cfg)
    elif trial_type == "syntax":
        execute_syntax(rng, dataset, cfg)
    else:
        raise NotImplementedError(trial_type)

    if cleanup_trial_files:
        logger.info("Cleaning up trial files")
        for file in save_dir.glob("*.parquet"):
            if file.name != "results.parquet":
                os.remove(file)


if __name__ == "__main__":
    cli()
