import gzip
import logging
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import ujson
from bigcode_eval.tasks import TASK_REGISTRY
from code_execution.eval_dataset import code_contests
from code_execution.eval_dataset import gsm8k
from evalplus.evaluate import evaluate as evalplus
from tqdm import tqdm

from src.metrics import estimate_pass_at_k
from src.utils import clean_solution

logger = logging.getLogger(__name__)


def _dedupe_predictions(
    predictions: List[Dict],
    process_fn: Optional[Callable[[str], str]] = None,
    prediction_key: str = "prediction",
) -> List[Dict]:
    """
    Deduplicates predictions while preserving metadata and aggregating statistics.

    Args:
        predictions: List of dictionaries containing prediction data.
            Each dict should have a "predictions" key containing a list of prediction dictionaries.
        process_fn: Optional function to process prediction strings before deduplication.
            Defaults to identity function.
        prediction_key: Key to use for accessing prediction text. Defaults to "prediction".
            Will fall back to "solution" if prediction_key is not found.

    Returns:
        List of dictionaries with deduplicated predictions and aggregated metadata.

    Raises:
        KeyError: If required keys are missing from the prediction dictionaries.
        ValueError: If predictions list is empty or malformed.
    """
    if not predictions:
        raise ValueError("Predictions list cannot be empty")

    if process_fn is None:
        process_fn = lambda x: x

    # Make a deep copy to avoid modifying the input
    predictions = deepcopy(predictions)
    total_removed = 0

    for i, problem in enumerate(predictions):
        if "predictions" not in problem:
            raise KeyError(f"Problem at index {i} missing 'predictions' key")

        deduped_preds: Dict[str, Dict] = {}

        for pred_dict in problem["predictions"]:
            # Get prediction text using fallback keys
            pred_text = pred_dict.get(prediction_key)
            if pred_text is None:
                pred_text = pred_dict.get("solution")
                if pred_text is None:
                    raise KeyError(
                        f"Prediction dict missing both '{prediction_key}' and 'solution' keys"
                    )

            # Process the prediction text
            processed_pred = process_fn(pred_text)

            # Extract metadata we want to aggregate
            metadata = {
                "cumulative_logprob": pred_dict.get("cumulative_logprob"),
                "completion_id": pred_dict.get("completion_id"),
            }

            if processed_pred not in deduped_preds:
                # Create new entry for unique prediction
                deduped_preds[processed_pred] = {
                    "cumulative_logprob": metadata["cumulative_logprob"],
                    "completion_id": metadata["completion_id"],
                    "count": 1,
                    # Copy remaining fields from original prediction
                    **{
                        k: v
                        for k, v in pred_dict.items()
                        if k
                        not in [
                            "cumulative_logprob",
                            "completion_id",
                            prediction_key,
                            "solution",
                        ]
                    },
                }
            else:
                # Update existing entry
                existing = deduped_preds[processed_pred]
                existing["count"] += 1
                if metadata["cumulative_logprob"] is not None:
                    existing["cumulative_logprob"] = max(
                        existing["cumulative_logprob"],
                        metadata["cumulative_logprob"],
                    )
                if metadata["completion_id"] is not None:
                    existing["completion_id"] = min(
                        existing["completion_id"], metadata["completion_id"]
                    )

        # Calculate how many duplicates were removed
        removed_count = len(problem["predictions"]) - len(deduped_preds)
        logger.debug(
            f"Problem {i}: Removed {removed_count:,} duplicate predictions"
        )
        total_removed += removed_count

        # Create final predictions list with computed averages
        final_predictions = []
        for pred_text, metadata in deduped_preds.items():
            final_pred = {
                "solution": pred_text,
                "count": metadata.pop("count"),
                "completion_id": metadata.pop("completion_id"),
            }

            # Add average logprob if available
            if metadata["cumulative_logprob"]:
                final_pred["cumulative_logprob"] = metadata.pop(
                    "cumulative_logprob"
                )

            # Add remaining metadata
            final_pred.update(
                {
                    k: v
                    for k, v in metadata.items()
                    if k not in ["cumulative_logprob", "completion_id", "count"]
                }
            )
            final_predictions.append(final_pred)

        predictions[i]["predictions"] = final_predictions

    logger.info(f"Total duplicates removed: {total_removed:,}")
    return predictions


def _process_evalplus(task_name: str, predictions: List, num_workers: int):
    """Processes evalplus results for a given task."""
    logger.info("Using evalplus to evaluate predictions.")

    def _postproc(p):
        p = p.rstrip().replace("\r", "")
        if "mbpp" in task_name:
            if "```python" in p:
                _, p = p.split("```python", 1)
                p = p.strip()
        if "\n```\n" in p:
            p, _ = p.split("\n```\n", 1)
        return p

    predictions = _dedupe_predictions(predictions, _postproc)
    pred_map = {}
    task_meta = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        logger.debug(f"Created temporary directory: {tmp_dir}")
        logger.debug("Writing predictions to temporary directories.")

        for idx, row in enumerate(predictions):
            if "mbpp" in task_name:
                task_id = f"Mbpp/{row['task_id']}"

                task_meta[task_id] = {
                    "prompt": row["prompt"],
                    "test_list": row["test_list"],
                }
            else:
                task_id = row["task_id"]
                task_meta[task_id] = {
                    "prompt": row["prompt"],
                    "fn_name": row["entry_point"],
                }

            task_dir = tmp_dir / task_id.replace("/", "_")
            task_dir.mkdir(parents=True)
            for pred in row["predictions"]:
                completion_id = pred["completion_id"]
                pred_file = task_dir / f"{completion_id}.py"
                pred_map[(task_id, completion_id)] = {
                    "cumulative_logprob": pred["cumulative_logprob"],
                    "count": pred["count"],
                }
                if "humaneval" in task_name:
                    if task_name.endswith("-unstripped"):
                        code = row["prompt"] + pred["solution"]
                    else:
                        code = row["prompt"].strip() + pred["solution"]
                else:
                    code = pred["solution"]
                with open(pred_file, "w", encoding="utf-8") as f:
                    f.write(code)
        with open(tmp_dir / "pred_map.json", "w", encoding="utf-8") as f:
            ujson.dump(
                {
                    f"{t.replace('/', '_')}/{i}": v
                    for (t, i), v in pred_map.items()
                },
                f,
            )
        logger.debug("Running evalplus on temporary directories.")
        evalplus(
            dataset="mbpp" if "mbpp" in task_name else "humaneval",
            samples=tmp_dir,
            parallel=num_workers,
        )

        logger.info("Reading evalplus results.")
        with open(tmp_dir / "eval_results.json", encoding="utf-8") as f:
            eval_results = ujson.load(f)
    out = []
    for tid, results in eval_results["eval"].items():
        sol_data = []
        for sol in results:
            base_pass = sol["base_status"] == "pass"
            plus_pass = sol["plus_status"] == "pass"
            key = (tid, sol["completion_id"])
            assert key in pred_map

            sol_data.append(
                {
                    "completion_id": sol["completion_id"],
                    "solution": sol["solution"],
                    "passed": base_pass,
                    "hidden_passed": plus_pass,
                    "elapsed": sol["elapsed"],
                    "cumulative_logprob": pred_map[key]["cumulative_logprob"],
                    "count": pred_map[key]["count"],
                    "base_details": sol["base_details"],
                    "plus_details": sol["plus_details"],
                }
            )

        out.append((tid, task_meta[tid], sol_data))

    return eval_results["elapsed"], out


def _process_gsm8k(
    predictions: List,
    num_workers: int,
    timeout: int,
) -> List[Tuple[str, Dict, List]]:
    """Process the predictions for GSM8K."""

    predictions = _dedupe_predictions(predictions)

    metrics, results = gsm8k.evaluate(
        predictions=predictions,
        num_workers=num_workers,
        timeout=timeout,
        solution_list_key="predictions",
        solution_str_key="solution",
    )

    logger.debug("Processing GSM8K results.")
    out = []
    for problem in results:
        sol_data = []
        tid = problem.pop("task_id")
        # Do this to remove the extra keys and data.
        for pred in problem.pop("predictions"):
            pred.pop("stdout")
            pred.pop("stderr")
            sol_data.append(pred)
        out.append((tid, problem, sol_data))

    return metrics["elapsed"], out


def _process_code_contests(
    predictions: List[Dict],
    num_workers: int,
    timeout: int,
    command_timeout: int,
):

    logger.debug("Evaluating Code Contests predictions.")
    predictions = _dedupe_predictions(predictions)
    metrics, predictions = code_contests.evaluate(
        predictions=predictions,
        num_workers=num_workers,
        first_command_timeout=max(30, timeout),
        command_timeout=command_timeout,
        solution_str_key="solution",
        solution_list_key="predictions",
        early_stopping=True,
        num_stdout_save=1,
    )
    out = []
    for problem in predictions:
        sol_data = []
        tid = problem.pop("task_id")
        for pred in problem.pop("predictions"):
            pred.pop("stderr")
            sol_data.append(pred)
        out.append((tid, problem, sol_data))
    logger.info(f"Evaluated {len(predictions):,} Code Contests problems.")

    return metrics["elapsed"], out


def _process_task_file(
    pred_file: Path,
    task_name: str,
    num_workers: int,
    timeout: int,
    command_timeout: int,
    debug_num: Optional[int] = None,
):
    """Postprocesses a list of predictions for a given task."""

    logger.info(f"Postprocessing predictions for '{pred_file.name}'")
    with gzip.open(pred_file, "rt", encoding="utf-8") as f:
        predictions = [ujson.loads(line) for line in f]

    if debug_num is not None:
        predictions = predictions[:debug_num]

    if task_name in {"mbpp", "humaneval"}:
        return _process_evalplus(
            task_name=task_name,
            predictions=predictions,
            num_workers=num_workers,
        )

    for i, r in enumerate(predictions):

        preds = []
        for p in r["predictions"]:
            p["solution"] = p.pop("prediction")
            preds.append(p)
        r["predictions"] = preds
        predictions[i] = r

    if "gsm" in task_name:
        logger.info(f"Evaluating GSM8K predictions for '{task_name}'")
        elapsed, out = _process_gsm8k(
            predictions=predictions,
            num_workers=num_workers,
            timeout=timeout,
        )
    elif task_name == "code_contests":
        logger.info(f"Evaluating Code Contests predictions for '{task_name}'")
        elapsed, out = _process_code_contests(
            predictions=predictions,
            num_workers=num_workers,
            timeout=timeout,
            command_timeout=command_timeout,
        )
    else:
        raise NotImplementedError(f"Postprocessing for '{task_name}'")
    return elapsed, out


def calculate_ranking_metrics(
    results: List[Tuple], task_name: str, num_samples: int, net_elapsed: float
):
    """Calculates ranking metrics for a given task."""
    logger.info(f"Calculating ranking metrics for '{task_name}'")
    out = []
    has_hidden = task_name in {"mbppplus", "humanevalplus"}
    num_oracle_pass = num_maj_pass = num_lp_pass = 0
    num_oracle_hidden = num_maj_hidden = num_lp_hidden = 0
    pass_counts = []
    mean_ranks = []
    hidden_ranks = []
    hidden_pass_counts = []
    all_elapsed_vals = []
    num_programs = 0
    for task_id, task_meta, sols in tqdm(results, desc="Calculating metrics"):
        # Rank of the first correct solution using logprob sorting. We do this
        # separately because we want to preserve generation order.
        lp_pass_rank = lp_hidden_rank = None
        has_passing = has_hidden_pass = False
        num_passed = num_hidden_passed = 0
        # Use default dict instead of counter because getting the max later is
        # not as bad compared to most_common.
        maj_counter = defaultdict(
            lambda: {
                "count": 0,
                "lowest_sol_idx": float("inf"),
                "passed": False,
                "hidden_passed": False,
            }
        )
        elapsed_vals = []
        for i, s in sorted(
            enumerate(sols, start=1),
            key=lambda x: x[1]["cumulative_logprob"],
            reverse=True,
        ):
            num_programs += 1
            cid = s["completion_id"]
            if isinstance(cid, list):
                cid = cid[0]
            cleaned_sol = clean_solution(s["solution"])
            maj_counter[cleaned_sol]["count"] += s["count"]
            maj_counter[cleaned_sol]["lowest_sol_idx"] = min(
                maj_counter[cleaned_sol]["lowest_sol_idx"], cid
            )
            maj_counter[cleaned_sol]["passed"] = s["passed"]
            elapsed_vals.append(s["elapsed"])
            if s["passed"]:
                num_passed += s["count"]
                has_passing = True
                if lp_pass_rank is None:
                    lp_pass_rank = i
            if has_hidden:

                maj_counter[cleaned_sol]["hidden_passed"] = s["hidden_passed"]
                if s["hidden_passed"]:

                    num_hidden_passed += s["count"]
                    has_hidden_pass = True
                    if lp_hidden_rank is None:
                        lp_hidden_rank = i
        all_elapsed_vals.extend(elapsed_vals)
        if lp_pass_rank is None:
            lp_pass_rank = len(sols) + 1
        if lp_hidden_rank is None:
            lp_hidden_rank = len(sols) + 1
        mean_ranks.append(1 / lp_pass_rank)
        pass_counts.append(num_passed)
        hidden_pass_counts.append(num_hidden_passed)

        if has_hidden:
            hidden_ranks.append(1 / lp_hidden_rank)

        majority_solution = max(
            maj_counter.values(),
            key=lambda x: (
                x["count"],
                -x["lowest_sol_idx"],
            ),
        )
        num_oracle_pass += has_passing
        num_maj_pass += majority_solution["passed"]
        num_lp_pass += lp_pass_rank == 1
        if has_hidden:
            num_oracle_hidden += has_hidden_pass
            num_maj_hidden += majority_solution["hidden_passed"]
            num_lp_hidden += lp_hidden_rank == 1
        prob_dict = {
            "task_id": task_id,
            "solutions": sols,
            **task_meta,
            "num_passed": num_passed,
            "lp_pass_rank": lp_pass_rank,
            "lp_passed": lp_pass_rank == 1,
            "oracle_passed": has_passing,
            "majority_passed": majority_solution["passed"],
            "majority_rank": majority_solution["lowest_sol_idx"] + 1,
        }
        if has_hidden:
            prob_dict.update(
                {
                    "num_hidden_passed": sum(x["hidden_passed"] for x in sols),
                    "lp_hidden_rank": lp_hidden_rank,
                    "lp_hidden_passed": lp_hidden_rank == 1,
                    "oracle_hidden_passed": has_hidden_pass,
                    "majority_hidden_passed": majority_solution[
                        "hidden_passed"
                    ],
                }
            )
        out.append(prob_dict)

    final_metrics = {
        "oracle_pass": num_oracle_pass / len(results),
        "majority_pass": num_maj_pass / len(results),
        "logprob_pass": num_lp_pass / len(results),
        "mean_rank": np.mean(mean_ranks),
        "mean_pass_count": np.mean(pass_counts),
        "mean_elapsed": np.mean(all_elapsed_vals),
        "median_elapsed": np.median(all_elapsed_vals),
        "std_elapsed": np.std(all_elapsed_vals),
        "programs_per_second": num_programs / net_elapsed,
    }
    if has_hidden:
        final_metrics.update(
            {
                "oracle_hidden_pass": num_oracle_hidden / len(results),
                "majority_hidden_pass": num_maj_hidden / len(results),
                "logprob_hidden_pass": num_lp_hidden / len(results),
                "mean_hidden_rank": np.mean(hidden_ranks),
            }
        )

    for k in [1, 10, 100]:
        if k > num_samples:
            break
        final_metrics[f"pass@{k}"] = np.mean(
            estimate_pass_at_k([num_samples] * len(pass_counts), pass_counts, k)
        )
        if has_hidden:
            final_metrics[f"hidden_pass@{k}"] = np.mean(
                estimate_pass_at_k(
                    [num_samples] * len(hidden_pass_counts),
                    hidden_pass_counts,
                    k,
                )
            )

    return final_metrics, out


def process_and_execute_raw_preds(
    raw_dir: Path,
    output_dir: Path,
    num_workers: int,
    timeout: int,
    command_timeout: int,
    debug_num: Optional[int] = None,
    execute_task: Optional[str] = None,
):
    logger.info(f"Reading raw predictions from {raw_dir}")
    metrics = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {output_dir}")
    for pred_file in raw_dir.glob("*.raw.jsonl.gz"):
        if execute_task is not None and not pred_file.stem.startswith(
            execute_task
        ):
            logger.info(f"Skipping {pred_file}")
            continue
        logger.debug(f"Processing {pred_file}")
        task_name = pred_file.stem.split(".")[0]

        elapsed, processed_preds = _process_task_file(
            pred_file=pred_file,
            task_name=task_name,
            num_workers=num_workers,
            timeout=timeout,
            command_timeout=command_timeout,
            debug_num=debug_num,
        )
        t_metrics, final_preds = calculate_ranking_metrics(
            results=processed_preds,
            task_name=task_name,
            num_samples=len(processed_preds[0][-1]),
            net_elapsed=elapsed,
        )
        t_metrics["net_elapsed"] = elapsed
        metrics[task_name] = t_metrics
        with gzip.open(
            output_dir / f"{task_name}.jsonl.gz", "wt", encoding="utf-8"
        ) as f:
            for row in tqdm(final_preds, desc=f"Saving {task_name}"):
                f.write(ujson.dumps(row) + "\n")

    for t, tv in metrics.items():
        logger.info(f"Metrics for {t}:")
        for k, v in tv.items():
            logger.info(f"{k:>32}={v:0.2f}")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        ujson.dump(
            {f"{t}/{k}": v for t, tv in metrics.items() for k, v in tv.items()},
            f,
            sort_keys=True,
            indent=2,
        )
