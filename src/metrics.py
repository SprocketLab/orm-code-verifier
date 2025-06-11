"""Metrics calculation module for evaluating solution quality and performance.

This module provides functionality for calculating various evaluation metrics for solutions,
particularly focused on pass@k metrics, ranking statistics, and timing measurements. It is
designed to work with scored solutions that have been evaluated against test cases.

The module supports:
- Pass@k estimation
- Ranking metrics (MRR, NDCG)
- Timing statistics
- Collision detection in predictions
- Test case pass rate analysis

Main components:
- ScoredSolution: Data class representing a single evaluated solution
- calculate_problem_metrics: Core function for computing metrics for a single problem
- process_task_scores: Aggregates metrics across multiple tasks
"""

import dataclasses
import itertools
import logging
from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy import special
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)

# Set of metrics that should be summed rather than averaged when aggregating
# across multiple problems
SUM_METRICS = {
    "num_collisions",  # Total number of score collisions across all problems
    "pred_collisions",  # Total number of prediction collisions across all problems
}


@dataclasses.dataclass
class ScoredSolution:
    """Represents a single evaluated solution with its scores and metadata.

    Attributes:
        sid (int): Solution identifier
        completion_id (int): Completion identifier for the solution
        score (float): Numerical score assigned to the solution
        score_time (float): Time taken to score the solution
        passed (bool): Whether the solution passed all tests
        passed_public (int): Number of public test cases passed
        passed_plus (int): Number of additional test cases passed
        logprob (float): Log probability of the solution
        num_tests_passed (int): Total number of tests passed
        count (int): Number of times this solution appears
        filtered_out (bool, optional): Whether this solution was filtered out. Defaults to False.
    """

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
    """Estimates the pass@k metric for each problem.

    The pass@k metric estimates the probability of solving a problem if k independent attempts
    are allowed. This implementation uses the unbiased estimator for pass@k described in
    "Evaluating Large Language Models Trained on Code" (Chen et al., 2021).

    Args:
        num_samples (Union[int, List[int]]): Number of samples per problem. Can be either a single
            integer applied to all problems, or a list of integers for each problem.
        num_correct (List[int]): Number of correct solutions for each problem.
        k (int): Number of allowed attempts.

    Returns:
        np.ndarray: Array of pass@k estimates for each problem.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates the unbiased pass@k estimator for a single problem.

        The estimator is calculated as: 1 - C(n-c,k)/C(n,k)
        where C(n,k) is the binomial coefficient.

        Args:
            n (int): Total number of samples
            c (int): Number of correct samples
            k (int): Number of allowed attempts

        Returns:
            float: Estimated probability of solving the problem with k attempts
        """
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
    """Calculates a ranking score based on the position of passed solutions.

    The ranking score measures how well the passed solutions are ranked at the top.
    It is calculated as the sum of passed solutions that appear before any failed solution,
    divided by the total number of passed solutions.

    Args:
        passed_vals (np.ndarray): Binary array indicating whether each solution passed (1) or failed (0)

    Returns:
        float: Ranking score between 0 and 1, where 1 indicates all passed solutions are ranked at the top
    """
    sum_counts = passed_vals.sum()
    if sum_counts == 0:
        return 0
    mask = np.arange(passed_vals.shape[0])
    mask[mask < sum_counts] = 1
    mask[mask >= sum_counts] = 0
    return (passed_vals * mask).sum() / sum_counts


def best_of_k(passing_vals, k):
    """Calculates the probability of having at least one passing solution in k random draws.

    This metric differs from pass@k in that it considers the actual distribution of passing
    solutions in the ranked list, rather than using an estimator.

    Args:
        passing_vals (np.ndarray): Binary array indicating whether each solution passed (1) or failed (0)
        k (int): Number of random draws to consider

    Returns:
        float: Probability between 0 and 1 of getting at least one passing solution in k draws
    """
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
    """Processes problem scores into arrays of passing tests and scores.

    This function converts a list of ScoredSolution objects into arrays that track:
    1. The scores of each solution
    2. Arrays indicating which solutions passed different test categories:
       - public: Tests visible in the problem statement
       - plus: Additional/hidden test cases
       - all: Combined results of both public and plus tests

    Note: Infinity scores are replaced with (min_score - 1) to ensure numerical stability
    in downstream calculations.

    Args:
        problem_scores (List[ScoredSolution]): List of scored solutions to process

    Returns:
        Tuple containing:
        - np.ndarray: Array of scores for each solution
        - Dict[str, np.ndarray]: Dictionary mapping test categories (public/plus/all) to
          boolean arrays indicating which solutions passed that category
        - np.ndarray: Array indicating the number of tests passed for each solution
    """
    out = defaultdict(list)
    score_array = []
    test_passed_array = []
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
    """Calculates various ranking statistics for different test categories.

    This function computes multiple ranking metrics for each test category (public/plus/all):
    - Mean Reciprocal Rank (MRR): 1/rank of first passing solution
    - Best of k: Probability of passing with k random draws
    - Ranking Score (rs): Measure of how well passing solutions are ranked
    - Passed: Whether the first solution passed

    The 'all' category (combined public+plus tests) is evaluated against all k values,
    while other categories only use the largest k value for efficiency.

    Args:
        pass_arrays (Dict[str, np.ndarray]): Dictionary mapping test categories to boolean
            arrays indicating which solutions passed
        k_vals (List[int]): List of k values to evaluate best_of_k metrics

    Returns:
        Dict[str, float]: Dictionary containing metrics for each test category, with keys
            formatted as "{category}/{metric_name}"
    """

    def rank_stats(pass_array: np.ndarray, use_k_vals: List[int]):
        """Calculates ranking statistics for a single pass array.

        Args:
            pass_array (np.ndarray): Boolean array indicating which solutions passed
            use_k_vals (List[int]): k values to use for best_of_k calculation

        Returns:
            Dict[str, Union[float, bool]]: Dictionary containing:
                - b_of_k: Best of k values for each k
                - rs: Ranking score
                - mrr: Mean reciprocal rank
                - passed: Whether first solution passed
        """
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
    """Calculates timing statistics for a set of problem solutions.

    This function computes various timing metrics including mean, median, and standard deviation
    of scoring times per program, as well as the total time spent on the problem.

    Args:
        problem_scores (List[ScoredSolution]): List of scored solutions to analyze

    Returns:
        Dict[str, float]: Dictionary containing the following timing metrics:
            - timing/seconds_per_prog: Mean seconds per program
            - timing/seconds_per_prog_median: Median seconds per program
            - timing/seconds_per_prog_std: Standard deviation of seconds per program
            - timing/seconds_per_problem: Total seconds spent on the problem
            Returns empty dict if no valid timings are found.
    """
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
    """Processes and aggregates metrics across multiple tasks.

    This function calculates comprehensive metrics across all tasks, including:
    - Pass@k metrics for various k values
    - NDCG (Normalized Discounted Cumulative Gain) scores
    - Timing statistics
    - Programs Per Second (PPS) metrics for different processing stages
    - Collision statistics

    Some metrics (defined in SUM_METRICS) are summed across tasks, while others
    are averaged.

    Args:
        scores (Dict[int, List[ScoredSolution]]): Dictionary mapping task IDs to lists
            of scored solutions
        timings (Dict[str, float]): Dictionary of timing measurements for different
            processing stages
        k_vals (List[int]): List of k values to evaluate pass@k metrics

    Returns:
        Tuple containing:
        - Dict: Aggregated metrics across all tasks
        - List[Dict]: List of individual solution scores with task IDs
    """
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
