import dataclasses
import itertools
import logging
from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal
from typing import Dict, List, Tuple

import numpy as np
from scipy import special
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)

SUM_METRICS = {
    "num_collisions",
    "pred_collisions",
}


@dataclasses.dataclass
class ScoredSolution:
    sid: int
    completion_id: int
    score: float
    score_time: float
    passed: bool
    passed_public: int
    passed_plus: int
    logprob: float
    num_tests_passed: int
    count: int
    filtered_out: bool = False


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [
            estimator(int(n), int(c), k)
            for n, c in zip(num_samples_it, num_correct)
        ]
    )


def ranking_score(passed_vals):

    sum_counts = passed_vals.sum()
    if sum_counts == 0:
        return 0
    mask = np.arange(passed_vals.shape[0])
    mask[mask < sum_counts] = 1
    mask[mask >= sum_counts] = 0
    return (passed_vals * mask).sum() / sum_counts


def best_of_k(passing_vals, k):
    N = len(passing_vals)
    out = np.array(
        [
            special.comb(N - i - 1, k - 1) if passing_vals[i] else 0
            for i in range(len(passing_vals) - k)
        ]
    )
    if special.comb(N, k) == 0:
        return 0
    return out.sum() / special.comb(N, k)


def _get_pass_arrays(
    problem_scores: List[ScoredSolution],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    out = defaultdict(list)
    score_array = []
    test_passed_array = []
    num_filtered = sum(
        score.filtered_out * score.count for score in problem_scores
    )
    scores = [score.score for score in problem_scores if not score.filtered_out]
    if len(scores) == 0:
        min_score = -10.0
    else:
        min_score = float(min(scores))
    for score in sorted(problem_scores, key=lambda x: x.score, reverse=True):
        # Can't keep infinity as the score for other metrics, so we ensure it is
        # replaced with the minimum score.
        actual_score = float(
            score.score if not score.filtered_out else min_score - 1
        )
        for _ in range(score.count):

            out["public"].append(score.passed_public)
            out["plus"].append(score.passed_plus)
            out["all"].append(score.passed)
            score_array.append(actual_score)
            test_passed_array.append(score.num_tests_passed)
    out = {k: np.array(v) for k, v in out.items()}

    return np.array(score_array), out, np.array(test_passed_array)


def get_ranking_stats(pass_arrays: Dict[str, np.ndarray], k_vals: List[int]):
    def rank_stats(pass_array: np.ndarray, use_k_vals: List[int]):
        out = {
            f"b_of_{k_val}": best_of_k(pass_array, k_val)
            for k_val in use_k_vals
        }
        out["rs"] = ranking_score(pass_array)

        if pass_array.sum() == 0:
            out["mrr"] = 0
            out["passed"] = False
            return out

        first_pass = pass_array.argmax()
        out["mrr"] = 1 / (first_pass + 1)
        out["passed"] = first_pass == 0
        return out

    stats = {}
    for k, v in pass_arrays.items():
        prefix = f"{k}/"

        stats.update(
            {
                prefix + k: v
                for k, v in rank_stats(
                    v, k_vals if k == "all" else [k_vals[-1]]
                ).items()
            }
        )
    return stats


def get_timing_stats(problem_scores: List[ScoredSolution]):
    timings = [ps.score_time for ps in problem_scores if not ps.filtered_out]
    if not timings:
        return {}
    return {
        "timing/seconds_per_prog": np.mean(timings),
        "timing/seconds_per_prog_median": np.median(timings),
        "timing/seconds_per_prog_std": np.std(timings),
        "timing/seconds_per_problem": sum(timings),
    }


def calculate_problem_metrics(
    problem_scores: List[ScoredSolution],
    k_vals: List[int],
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Calculates scoring metrics for a single problem."""

    # First parse the scores in to arrays and get the arrays for pass counts.
    score_array, pass_arrays, test_passed_array = _get_pass_arrays(
        problem_scores
    )
    is_multi_label = len(score_array.shape) == 2

    coll_count = Counter(list(map(Decimal, score_array)))
    margins = pred_collisions = None
    ranking_stats = get_ranking_stats(pass_arrays, k_vals)
    num_collisions = sum([v > 1 for v in coll_count.values()])

    out = {
        **ranking_stats,
        **get_timing_stats(problem_scores),
        "num_collisions": num_collisions,
        "unique_preds": len(problem_scores),
    }

    if is_multi_label:
        out["pred_collisions"] = pred_collisions
        out["margins"] = margins.mean()
        out["margin_std"] = margins.std()

    return out, {
        "score_array": score_array,
        "test_passed_array": test_passed_array,
        "passed_array": pass_arrays["all"],
    }


def process_task_scores(
    scores: Dict[int, List[ScoredSolution]],
    timings: Dict[str, float],
    k_vals: List[int],
) -> Tuple[Dict, List[Dict]]:
    logger.info(f"Calculating metrics for {len(scores)} tasks")

    # Remove k_vals that are larger than the number of scores.
    longest_score = max(map(len, scores.values()))
    k_vals = [k_val for k_val in k_vals if k_val <= longest_score]
    if not k_vals:
        k_vals = [longest_score]
    logger.debug(
        f"Using k_vals: {k_vals} with longest task having {longest_score} scores"
    )

    final_metrics = defaultdict(list)
    total_programs = 0
    test_passed_array = []
    score_array = []
    passed_array = []
    for pid, task_scores in scores.items():
        total_programs += len(task_scores)

        task_scores, arrays = calculate_problem_metrics(
            task_scores, k_vals=k_vals
        )
        for k, v in task_scores.items():
            final_metrics[k].append(v)
        test_passed_array.append(arrays["test_passed_array"])
        score_array.append(arrays["score_array"])
        passed_array.append(arrays["passed_array"])
    test_passed_array = np.array(test_passed_array)
    score_array = np.array(score_array)
    passed_array = np.array(passed_array)
    out = {}
    for k, v in final_metrics.items():
        if k in SUM_METRICS:
            out[k] = int(sum(v))
            if k in {"num_collisions", "pred_collisions"}:
                total = sum(final_metrics["unique_preds"])
                if k == "num_collisions":
                    key = "pct_collisions"
                else:
                    key = "pct_pred_collisions"
                out[key] = out[k] / total
        elif v:
            out[k] = float(np.mean(v))
        else:
            out[k] = 0

    for k in k_vals:
        out[f"all/ndcg@{k}"] = (
            ndcg_score(test_passed_array, score_array, k=k) * 100
        )
    out[f"all/ndcg@all"] = ndcg_score(test_passed_array, score_array) * 100
    out["total_programs"] = total_programs
    for k, elap in timings.items():
        out[f"timing/{k}"] = elap
        if elap > 0:
            out[f"pps/{k}"] = total_programs / elap

    out["pps/filter_and_score"] = total_programs / (
        timings["filtering"] + timings["scoring_function"]
    )

    save_scores = [
        {"task_id": k, **asdict(s)} for k, v in scores.items() for s in v
    ]
    return out, save_scores
