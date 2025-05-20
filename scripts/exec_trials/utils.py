import ast
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from code_execution.code_trees import safe_ast_parse
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


PROG_RESULTS_FILE = "raw_program_results.csv.gz"
OVERALL_FILE = "overall.json"
FINAL_RESULTS_FILE = "results.parquet"
HOMOGENEOUS_COLUMNS = {
    "num_ran",
    "num_passed",
    "passed",
    "return_code",
    "timeout",
}

TIMING_COLUMNS = {
    "writing_time",
    "execution_time",
    "cleanup_time",
    "preprocess_time",
    "postprocess_time",
    "base_execution_time",
    "plus_execution_time",
}


@dataclass
class Solution:
    """Base class for storing solution information.

    Attributes:
        solution (str): The solution code or content
        sid (str): Unique solution identifier
    """

    solution: str
    sid: str


@dataclass
class Problem:
    """Base class for representing a problem.

    Attributes:
        task_id (str): Unique identifier for the task/problem
        model (str): Model that generated these solutions.
        solutions (List[Solution]): List of solutions for this problem
        sampling_setup (str): Sampling setup used to generate these solutions
    Class Methods:
        from_dict: Creates a Problem instance from a dictionary
    """

    task_id: str
    model: str
    sampling_setup: str
    solutions: List[Solution]

    @classmethod
    def from_dict(cls, data: Dict) -> "Problem":
        data["solutions"] = [Solution(**s) for s in data["solutions"]]
        return cls(**data)


@dataclass
class EvalPlusProblem(Problem):
    """Problem class specific to EvalPlus."""

    source: str


@dataclass
class CodeContestsProblem(Problem):
    """Problem class specific to Code Contests.

    Attributes:
        name (str): Name of the problem
        public_tests (Dict): Dictionary containing public test cases
        private_tests (Dict): Dictionary containing private test cases
        generated_tests (Dict): Dictionary containing generated test cases
        time_limit (int): Time limit for solution execution in seconds
        memory_limit_bytes (int): Memory limit for solution execution in bytes
    """

    name: str
    public_tests: Dict
    private_tests: Dict
    generated_tests: Dict
    time_limit: int
    memory_limit_bytes: int


@dataclass
class GSM8KProblem(Problem):
    """Problem class specific to GSM8K math problems.

    Attributes:
        answer (str): The expected answer for the math problem
    """

    answer: str


def get_builtin_modules() -> Set[str]:
    """Get set of Python's built-in modules."""
    import pkgutil
    import sys

    builtin_modules = {
        mod.name
        for mod in pkgutil.iter_modules()
        if mod.name in sys.stdlib_module_names
    }
    # Add some common built-ins that might be missed
    builtin_modules.update(
        {
            "os",
            "sys",
            "math",
            "random",
            "time",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "re",
        }
    )
    return builtin_modules


def display_import_counts(
    import_stats: Dict[str, Tuple[int, int]], min_count: int = 0
):
    """Display import counts and pass rates in a formatted table using Rich.

    Args:
        import_stats: Dictionary mapping package names to (total_count, pass_count)
        min_count: Minimum count threshold for displaying a package
    """
    console = Console()
    table = Table(title="Package Import Statistics")

    table.add_column("Package", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Passed", justify="right", style="yellow")

    # Sort by count (descending) then by name
    sorted_imports = sorted(
        [
            (pkg, passed)
            for pkg, passed in import_stats.items()
            if len(passed) >= min_count
        ],
        key=lambda x: len(x[1]),
    )

    for package, passed in sorted_imports:
        table.add_row(package, f"{len(passed):,}", f"{sum(passed):,}")

    console.print(table)


def extract_imports(code: str) -> Set[str]:
    """Extract import names from Python code.

    Args:
        code: Python source code as string

    Returns:
        Set of package names that are imported
    """
    imports = set()
    tree = safe_ast_parse(code)
    if tree is None:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                # Get the actual package name, ignoring any alias
                base_package = name.name.split(".")[0]
                imports.add(base_package)
        elif isinstance(node, ast.ImportFrom):
            if node.module:  # Handles "from x import y"
                base_package = node.module.split(".")[0]
                imports.add(base_package)
            # Skip relative imports starting with '.'

    return imports


def check_homogeneous(series):
    """Check if all values in a series are identical.

    Returns:
        tuple: (is_homogeneous, value if homogeneous else None)
    """
    first_val = series.iloc[0]
    is_homogeneous = (series == first_val).all()
    counts = Counter(series)
    return (
        None
        if is_homogeneous
        else "|".join(f"{k}:{v}" for k, v in counts.items())
    )


def get_mean_trial_test_time(series: pd.Series) -> pd.Series:
    """Get mean test time for a given trial.

    Args:
        series: Series of test times

    Returns:
        Mean test time
    """
    return series.mean()


def get_test_passed_at_n(series: pd.Series) -> pd.Series:
    """Get test passed at n.

    Args:
        series: Series of test results

    Returns:
        Test passed at n
    """
    return series.all()


def percentile(n):
    def percentile_(x):
        return x.quantile(n)

    percentile_.__name__ = "percentile_{:02.0f}".format(n * 100)
    return percentile_


def process_raw_results(output_dir: Path) -> pd.DataFrame:
    """Process raw results dataframe to compute statistics.

    Args:
        raw_results_df: DataFrame containing raw execution results

    Returns:
        DataFrame with aggregated statistics
    """
    parquet_files = [
        p for p in output_dir.glob("*.parquet") if p.name != "results.parquet"
    ]
    if not parquet_files:
        raise ValueError(f"No parquet files found in {output_dir}")

    raw_results_df = pd.concat(
        [pd.read_parquet(f) for f in parquet_files], ignore_index=True
    )

    logger.info(f"Value columns for statistics: {raw_results_df.columns}")

    first_columns = {
        c: "first"
        for c in raw_results_df.columns
        if c
        not in TIMING_COLUMNS.union(HOMOGENEOUS_COLUMNS).union(
            {"trial", "task_id", "sid"}
        )
    }

    # Define aggregation dictionary
    agg_dict = {
        # For homogeneity check columns
        **{
            col: ["first", check_homogeneous]
            for col in HOMOGENEOUS_COLUMNS
            if col in raw_results_df.columns
        },
        **first_columns,
        # For numeric columns, calculate statistics
        **{
            col: ["mean", "std"]
            for col in TIMING_COLUMNS
            if col in raw_results_df.columns
        },
    }
    logger.info(f"Aggregating {agg_dict}")
    if "n_test_time" in raw_results_df.columns:
        agg_dict["n_test_time"] = "mean"

    # Group and aggregate
    processed_df = raw_results_df.groupby(["task_id", "sid"]).agg(agg_dict)

    new_cols = []
    for col, agg_func in processed_df.columns:
        if agg_func in {"first", "mean"}:
            new_cols.append(col)
        elif agg_func == "check_homogeneous":
            new_cols.append(f"{col}.vals")
        else:
            new_cols.append(f"{col}.{agg_func}")

    processed_df.columns = new_cols

    # Reset index to make task_id and sid regular columns
    return processed_df.reset_index()
