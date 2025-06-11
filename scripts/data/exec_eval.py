import functools
import gzip
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import click
import ujson
from code_execution import run_in_parallel
from code_execution.eval_dataset import code_contests
from code_execution.eval_dataset import gsm8k
from code_execution.utils import swallow_io
from evalplus.data import get_human_eval_plus
from evalplus.data import get_mbpp_plus
from evalplus.eval import FAIL
from evalplus.eval import PASS
from evalplus.eval import TIMEOUT
from evalplus.evaluate import evaluate as evalplus
from evalplus.sanitize import sanitize
from rich.console import Console
from rich.logging import RichHandler

sys.path.append(str(Path(__file__).parents[2].absolute()))

logger = logging.getLogger(__name__)
CONSOLE = Console(width=156)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("black").setLevel(logging.WARNING)
logging.getLogger("blib2to3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)


def dedup_predictions(
    predictions: List[Dict[str, Any]], clean_fn: Callable
) -> List[Dict[str, Any]]:
    """Deduplicates the predictions for the problem. Adds a count field to the prediction that is the sum of the counts of the duplicates.

    Takes the MINIMUM value for the other keys.
    """
    # Track total count to verify at end
    total_count = len(predictions)

    # Group predictions by their solution text
    grouped: Dict[str, List[Dict]] = {}
    for pred in predictions:
        solution = clean_fn(pred["prediction"])
        if solution not in grouped:
            grouped[solution] = []
        grouped[solution].append(pred)

    # Combine duplicates
    deduped = []
    for cleaned_solution, preds in grouped.items():
        # Start with first prediction as base
        combined = preds[0].copy()

        count = 1

        # Combine with remaining duplicates
        for pred in preds[1:]:
            count += 1
            # Take minimum value for all numeric fields
            for k, v in pred.items():
                if isinstance(v, (int, float)) and k in combined:
                    combined[k] = min(combined[k], v)
                else:
                    combined[k] = v
        combined["prediction"] = cleaned_solution
        combined["count"] = count
        deduped.append(combined)

    # Verify total count is preserved
    assert (
        sum(p["count"] for p in deduped) == total_count
    ), "Total count mismatch after deduplication"

    return deduped


def execute_evalplus(
    dataset_name: str,
    predictions: List[Dict[str, Any]],
    min_time_limit: float = 1.0,
    gt_time_limit_factor: float = 4.0,
    num_workers: int = 4,
    run_all_tests: bool = False,
) -> Dict:
    """Processes evalplus results for a given task."""
    logger.info("Using evalplus to evaluate predictions.")

    pred_map = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for i, row in enumerate(predictions):
            task_id = row["task_id"]
            task_dir = tmp_dir / task_id.replace("/", "_")
            task_dir.mkdir(parents=True)
            for j, pred in enumerate(row["predictions"]):
                completion_id = pred["completion_id"]
                pred_file = task_dir / f"{completion_id}.py"
                pred_map[(task_id, completion_id)] = {
                    "completion_id": completion_id,
                    "row_id": i,
                    "pred_id": j,
                    "count": pred["count"],
                    "logprob": pred["cumulative_logprob"],
                }
                to_write = pred["prediction"]
                with open(pred_file, "w", encoding="utf-8") as f:
                    f.write(to_write)
                with open(
                    tmp_dir / "pred_map.json", "w", encoding="utf-8"
                ) as f:
                    ujson.dump(
                        {
                            f"{t.replace('/', '_')}/{i}": v
                            for (t, i), v in pred_map.items()
                        },
                        f,
                    )

        evalplus(
            dataset=dataset_name,
            samples=str(tmp_dir.absolute()),
            parallel=num_workers,
            disable_cache=False,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
            test_details=run_all_tests,
        )

        logger.info("Reading evalplus results.")
        with open(tmp_dir / "eval_results.json", encoding="utf-8") as f:
            eval_results = ujson.load(f)

        out = []
        for tid, res in eval_results["eval"].items():
            sol_list = []
            for sol in res:
                passed_base = sol["base_status"] == PASS

                if sol["plus_status"] == "NOT_RAN":
                    passed_plus = None
                    passed = passed_base
                else:
                    passed_plus = sol["plus_status"] == PASS
                    passed = passed_base and passed_plus

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
                        },
                        "passed": passed,
                        "passed_public": passed_base,
                        "passed_plus": passed_plus,
                        "plus_outcomes": sol["plus_details"],
                        "base_outcomes": sol["base_details"],
                        "count": sol["count"],
                        "cumulative_logprob": sol["logprob"],
                    }
                )
            out.append(
                {
                    "task_id": tid,
                    "predictions": sol_list,
                }
            )

    return out


def process_evalplus_problem(problem: Dict[str, Any], dataset: Dict) -> Dict:
    entry_point = dataset[problem["task_id"]]["entry_point"]
    with swallow_io():
        problem["predictions"] = dedup_predictions(
            problem["predictions"], lambda x: sanitize(x, entry_point)
        )
    return problem


def process_problem(problem: Dict[str, Any]) -> Dict:
    problem["predictions"] = dedup_predictions(
        problem["predictions"], lambda x: x
    )
    return problem


@click.command()
@click.argument(
    "pred_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output_dir",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    default=None,
)
@click.option("--output_name", "-out_name", type=str, default=None)
@click.option("--run_all_tests", "-full", is_flag=True)
@click.option("--num_workers", "-n", type=int, default=4)
@click.option("--debug_problems", "-d", type=int, default=None)
@click.option("--max_tests", "-m", type=int, default=None)
@click.option("--ce_timeout", "-cet", type=float, default=30)
@click.option("--ce_command_timeout", "-cect", type=float, default=10)
@click.option("--evalplus_min_timeout", "-etmin", type=float, default=1)
@click.option("--evalplus_gt_timeout_factor", "-etgt", type=float, default=4)
def execute_eval_file(
    pred_file: Path,
    output_dir: Optional[Path],
    output_name: Optional[str],
    run_all_tests: bool,
    num_workers: int,
    debug_problems: Optional[int],
    max_tests: Optional[int],
    ce_timeout: float,
    ce_command_timeout: float,
    evalplus_min_timeout: float,
    evalplus_gt_timeout_factor: float,
):

    if output_dir is None:
        output_dir = pred_file.parent
    else:
        output_dir = output_dir.absolute()
    dataset_name = pred_file.stem.split(".raw")[0]
    out_name = output_name if output_name is not None else dataset_name

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(console=CONSOLE, level=logging.INFO),
        ],
    )

    out_file = output_dir / (out_name + "_executed.jsonl.gz")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Executing eval file: {pred_file}")
    logger.info(f"Output File: {out_file}")
    logger.info(f"Run all tests: {run_all_tests}")
    logger.info(f"Dataset Name: {dataset_name}")
    logger.info(f"Output Name: {out_name}")

    has_printed = False
    problems = []

    with gzip.open(pred_file, "rt", encoding="utf-8") as f:
        for line in map(ujson.loads, f):
            if not has_printed:
                logger.info(
                    f"Line Keys: "
                    + ",".join(f"{k}:{type(v)}" for k, v in line.items())
                )
                has_printed = True
            if "query" in line:
                line.pop("query")
            if "prompt" in line:
                line.pop("prompt")
            problems.append(line)

    logger.info(f"Loaded {len(problems):,} problems")
    mbpp_plus = get_mbpp_plus()
    humaneval_plus = get_human_eval_plus()
    if debug_problems is not None:
        logger.info(f"Debugging {debug_problems} problems")
        problems = problems[:debug_problems]

    if dataset_name == "mbpp":
        processor = functools.partial(
            process_evalplus_problem, dataset=mbpp_plus
        )
    elif dataset_name == "humaneval":
        processor = functools.partial(
            process_evalplus_problem, dataset=humaneval_plus
        )
    else:
        processor = process_problem

    start_num_preds = sum(map(lambda x: len(x["predictions"]), problems))
    logger.info(f"Starting with {start_num_preds} predictions")
    problems = run_in_parallel(
        processor, problems, num_workers=num_workers, desc="Processing problems"
    )
    end_num_preds = sum(map(lambda x: len(x["predictions"]), problems))
    logger.info(
        f"Total of {end_num_preds:,} predictions ({end_num_preds/start_num_preds:0.2%} of total)"
    )
    logger.info(f"Deduplicated {start_num_preds - end_num_preds:,} predictions")

    if dataset_name == "code_contests":
        _, problems = code_contests.evaluate(
            predictions=problems,
            num_workers=num_workers,
            first_command_timeout=ce_timeout,
            command_timeout=ce_command_timeout,
            num_stdout_save=1,
            early_stopping=not run_all_tests,
            max_commands=max_tests,
            execution_kwargs={
                "log_freq": 2500,
                "buffer_size": 100,
            },
            force_command_timeout=False,
            solution_str_key="prediction",
            solution_list_key="predictions",
        )
        for p in problems:
            p.pop("inputs")
            p.pop("outputs")
    elif dataset_name in {"gsm8k"}:
        _, problems = gsm8k.evaluate(
            predictions=problems,
            num_workers=num_workers,
            timeout=ce_timeout,
            solution_str_key="prediction",
            solution_list_key="predictions",
        )
    elif dataset_name in {"mbpp", "humaneval"}:
        problems = execute_evalplus(
            dataset_name=dataset_name,
            predictions=problems,
            num_workers=num_workers,
            min_time_limit=evalplus_min_timeout,
            gt_time_limit_factor=evalplus_gt_timeout_factor,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_preds_per = [
        sum(s["count"] for s in p["predictions"]) for p in problems
    ]
    assert (
        len(set(num_preds_per)) == 1
    ), "All predictions should have the same number of solutions"
    logger.info(f"Writing to {out_file}")
    with gzip.open(out_file, "wt", encoding="utf-8") as f:
        for problem in problems:
            f.write(ujson.dumps(problem) + "\n")


if __name__ == "__main__":
    execute_eval_file()
