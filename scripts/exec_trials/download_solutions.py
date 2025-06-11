"""This"""

import gzip
import logging
import tempfile
from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import click
import ujson
from datasets import load_dataset
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

from utils import CodeContestsProblem
from utils import EvalPlusProblem
from utils import GSM8KProblem
from utils import Solution

# Set up logging similar to run_trials.py
CONSOLE = Console(width=156)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("black").setLevel(logging.WARNING)
logging.getLogger("blib2to3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def process_code_contests(
    row: Dict, original_dataset: Dict
) -> CodeContestsProblem:
    original_row = original_dataset[row["task_id"]]

    solutions = [
        Solution(
            solution=s["solution"],
            sid=s["sid"],
        )
        for s in row["solutions"]
    ]

    problem = CodeContestsProblem(
        task_id=row["task_id"],
        name=original_row["name"],
        solutions=solutions,
        model=row["model"],
        sampling_setup=row["setup"],
        public_tests=original_row["public_tests"],
        private_tests=original_row["private_tests"],
        generated_tests=original_row["generated_tests"],
        time_limit=original_row["time_limit"],
        memory_limit_bytes=original_row["memory_limit_bytes"],
    )

    return problem


def process_gsm8k(row: Dict, gsm8k_dataset: Dict) -> GSM8KProblem:

    original_row = gsm8k_dataset[row["task_id"]]
    solutions = [
        Solution(
            solution=s["solution"],
            sid=s["sid"],
        )
        for s in row["solutions"]
    ]

    return GSM8KProblem(
        task_id=row["task_id"],
        model=row["model"],
        solutions=solutions,
        sampling_setup=row["setup"],
        answer=original_row["answer"],
    )


def process_evalplus(row: Dict) -> EvalPlusProblem:

    solutions = [
        Solution(
            solution=s["solution"],
            sid=s["sid"],
        )
        for s in row["solutions"]
    ]

    return EvalPlusProblem(
        task_id=row["task_id"],
        model=row["model"],
        solutions=solutions,
        source=row["dataset"],
        sampling_setup=row["setup"],
    )


def process_dataset(row: Dict, cc_dataset: Dict, gsm8k_dataset: Dict) -> Dict:

    if row["dataset"] == "gsm8k":
        prob = process_gsm8k(row, gsm8k_dataset)
    elif row["dataset"] == "code_contests":
        prob = process_code_contests(row, cc_dataset)
    elif row["dataset"] in {"humaneval", "mbpp"}:
        prob = process_evalplus(row)
    else:
        raise ValueError(f"Unknown dataset: {row['dataset']}")

    return {
        "dataset": row["dataset"],
        # Need to convert to string because of huggingface datasets.
        "data": ujson.dumps(asdict(prob)),
    }


@click.command()
@click.option(
    "--output_dir",
    type=click.Path(dir_okay=True, path_type=Path, file_okay=False),
    default=Path(__file__, "..", "..", "..", "data", "exec_trial_problems"),
)
def cli(
    output_dir: Path,
):
    """Download and process solutions for later evaluation.

    Args:
        output_dir: Directory to save processed solutions
    """
    output_dir = output_dir.resolve().absolute()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(console=CONSOLE, level=logging.INFO),
        ],
    )
    logger.info(f"Downloading and processing solutions for {output_dir}")
    dataset = load_dataset(
        "anon/eval-corm_black_comments",
        split="test",
    )

    original_code_contests = {
        v["name"]: v
        for v in load_dataset("deepmind/code_contests", split="test")
    }

    gsm8k_dataset = {
        f"gsm8k/{i}": l
        for i, l in enumerate(load_dataset("gsm8k", "main", split="test"))
    }

    processed = dataset.map(
        process_dataset,
        fn_kwargs={
            "cc_dataset": original_code_contests,
            "gsm8k_dataset": gsm8k_dataset,
        },
        num_proc=8,
        remove_columns=dataset.column_names,
    )

    processed_problems = defaultdict(list)

    for row in tqdm(processed, desc="Processing problems"):

        if row["dataset"] == "gsm8k":
            problem_cls = GSM8KProblem
        elif row["dataset"] == "code_contests":
            problem_cls = CodeContestsProblem
        else:
            problem_cls = EvalPlusProblem
        problem = problem_cls(**ujson.loads(row["data"]))
        processed_problems[
            (row["dataset"], problem.model, problem.sampling_setup)
        ].append(problem)

    logger.info(f"Finished processing {len(dataset):,} problems")

    for (
        dataset,
        model,
        sampling_setup,
    ), problems in processed_problems.items():
        logger.info(
            f"Saving {len(problems):,} problems for {dataset}_{model}_{sampling_setup}"
        )
        save_dir = output_dir.resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        with gzip.open(
            save_dir / f"{dataset}_{model}_{sampling_setup}.jsonl.gz", "wt"
        ) as f:
            for problem in tqdm(problems, desc=f"Saving {dataset}_{model}"):
                f.write(ujson.dumps(asdict(problem)) + "\n")


if __name__ == "__main__":
    cli()
