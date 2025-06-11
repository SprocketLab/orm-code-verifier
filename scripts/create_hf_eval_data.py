import functools
import gzip
import logging
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict, List, Optional

import click
import ujson
import yaml
from code_execution.utils import run_in_parallel
from datasets import load_dataset
from evalplus.data import get_human_eval_plus
from evalplus.data import get_mbpp_plus
from rich.logging import RichHandler
from tqdm import tqdm

ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT.resolve().absolute()))
from src.evaluation.preproc import get_eval_ds_name
from src.utils import format_with_black

logger = logging.getLogger(__name__)
logging.getLogger("parso").setLevel(logging.WARNING)
logging.getLogger("black").setLevel(logging.WARNING)
logging.getLogger("blib2to3.pgen2.driver").setLevel(logging.WARNING)

HE_DATASET = get_human_eval_plus()
MBPP_DATASET = get_mbpp_plus()

GSM8K_DATASET = {
    f"gsm8k/{i}": l
    for i, l in enumerate(load_dataset("gsm8k", "main", split="test"))
}
DEFAULT_CONFIG = {"license": "apache-2.0"}
DEFAULT_README = "# Ranking Evaluation Dataset"
DEFAULT_CONFIG_LIST = [
    {
        "config_name": "default",
        "data_files": [{"split": "test", "path": ["**/*.jsonl.gz"]}],
    },
]

EXPECTED_TASKS = {"humaneval", "mbpp", "gsm8k", "code_contests"}

SETUP_NAME_TO_CLEAN = {
    "t1.0_n128": "t10_n128",
    "t1.0_n256": "t10_n256",
    "t0.2_n128": "t02_n128",
    "t0.4_n128": "t04_n128",
    "t0.6_n128": "t06_n128",
    "t0.8_n128": "t08_n128",
}


@dataclass
class Solution:
    sid: int
    completion_id: int
    solution: str
    passed: bool
    passed_public: bool
    passed_plus: bool
    timed_out: bool
    logprob: float
    count: int
    elapsed: float
    num_tests_passed: int = 0
    outcomes: List[bool] = None
    plus_outcomes: List[bool] = None

    def __post_init__(self):
        if not self.outcomes:
            self.outcomes = [self.passed]
        self.num_tests_passed = sum(self.outcomes)
        if not self.plus_outcomes:
            self.plus_outcomes = [self.passed_plus]
        else:
            self.num_tests_passed += sum(self.plus_outcomes)


@dataclass
class Problem:
    task_id: str
    num_passed: int
    num_pub_tests: int
    num_plus_tests: int
    num_passed_plus: int
    num_passed_public: int
    solutions: List[Solution]


def process_evalplus_line(line: dict, source) -> Problem:
    num_passed = 0
    num_passed_plus = 0
    num_passed_pub = 0
    solutions = []
    row = (
        HE_DATASET[line["task_id"]]
        if source == "humaneval"
        else MBPP_DATASET[line["task_id"]]
    )

    num_pub_tests = len(row["base_input"])
    num_plus_tests = len(row["plus_input"])
    for sid, p in enumerate(line["predictions"]):
        num_passed += p["passed"]
        num_passed_pub += p["passed_public"]
        num_passed_plus += p["passed_plus"]
        solution = Solution(
            sid=sid,
            completion_id=p["sid"],
            solution=p["solution"],
            passed=p["passed"],
            passed_public=p["passed_public"],
            passed_plus=p["passed_plus"],
            timed_out=p["timeout"],
            count=p["count"],
            logprob=p.get("cumulative_logprob", 0.0),
            elapsed=p["timing"]["execution"],
            outcomes=p["base_outcomes"],
            plus_outcomes=p["plus_outcomes"],
        )
        solutions.append(solution)
    return Problem(
        task_id=line["task_id"],
        num_passed=num_passed,
        num_pub_tests=num_pub_tests,
        num_plus_tests=num_plus_tests,
        num_passed_plus=num_passed_plus,
        num_passed_public=num_passed_pub,
        solutions=solutions,
    )


def process_gsm8k_line(line: dict):
    num_passed = 0

    solutions = []
    for sid, p in enumerate(line["predictions"]):
        num_passed += p["passed"]

        solution = Solution(
            sid=sid,
            completion_id=p["completion_id"],
            solution=p["prediction"],
            passed=p["passed"],
            passed_public=p["passed"],
            passed_plus=p["passed"],
            timed_out=p["timeout"],
            count=p["count"],
            logprob=p["cumulative_logprob"],
            elapsed=p["timing"]["execution"]
            + p["timing"]["preprocess"]
            + p["timing"]["postprocess"],
        )
        solutions.append(solution)
    return Problem(
        task_id=line["task_id"],
        num_passed=num_passed,
        num_pub_tests=1,
        num_plus_tests=0,
        num_passed_plus=0,
        num_passed_public=num_passed,
        solutions=solutions,
    )


def process_code_contests_line(line: Dict) -> Problem:

    test_counts = Counter(line["test_types"])
    num_pub_tests = test_counts[0]
    plus_tests = test_counts[2] + test_counts[1]

    num_passed = 0
    num_passed_plus = 0
    num_passed_pub = 0
    solutions = []
    for sid, p in enumerate(line["predictions"]):
        num_passed += p["passed"]

        passed_hidden = p.get("passed_private", None)
        passed_generated = p.get("passed_generated", None)

        # One of these must exist.
        if passed_hidden is None:
            passed_hidden = passed_generated
        if passed_generated is None:
            passed_generated = passed_hidden
        if passed_hidden is None and passed_generated is None:
            raise ValueError("No passed values found")
        if passed_generated and passed_hidden:
            passed_plus = True
            num_passed_plus += 1
        else:
            passed_plus = False
        num_passed += p["passed"]
        num_passed_pub += p["passed_public"]

        solution = Solution(
            sid=sid,
            completion_id=p["completion_id"],
            solution=p["prediction"],
            passed=p["passed"],
            passed_public=p["passed_public"],
            passed_plus=passed_plus,
            timed_out=p["timeout"],
            count=p["count"],
            logprob=p["cumulative_logprob"],
            elapsed=p["timing"]["execution"]
            + p["timing"]["preprocess"]
            + p["timing"]["postprocess"],
            outcomes=p["outcomes"],
        )
        solutions.append(solution)
    out_dict = Problem(
        task_id=line["name"],
        num_passed=num_passed,
        num_pub_tests=num_pub_tests,
        num_plus_tests=plus_tests,
        num_passed_plus=num_passed_plus,
        num_passed_public=num_passed_pub,
        solutions=solutions,
    )
    return out_dict


TASK_PROCESS = {
    "humaneval": lambda x: process_evalplus_line(x, "humaneval"),
    "mbpp": lambda x: process_evalplus_line(x, "mbpp"),
    "gsm8k": process_gsm8k_line,
    "code_contests": process_code_contests_line,
}


def clean_predictions(
    problem: Problem, remove_comments: bool, black_format: bool
) -> Problem:

    for i, sol in enumerate(problem.solutions):
        if remove_comments:
            sol.solution = remove_comments(sol.solution)
        if black_format:
            sol.solution = format_with_black(sol.solution)
        problem.solutions[i] = sol
    return problem


def process_pred_file(
    model_name: str,
    pred_file: Path,
    remove_comments: bool,
    black_format: bool,
    num_workers: int = 1,
    debug_num: Optional[int] = None,
):
    logger.info(f"Processing predictions from {pred_file}")
    logger.info(f"Model name: {model_name}")

    task_name = pred_file.name.split("_exec")[0]
    logger.info(f"Task name: {task_name}")

    process_fn = TASK_PROCESS.get(task_name)
    if process_fn is None:
        logger.error(f"Missing process function for {task_name}")
        exit(1)

    logger.info(f"Processing predictions from {pred_file}")
    problems = []
    with gzip.open(pred_file, "rt", encoding="utf-8") as f:
        for read_num, line in enumerate(map(ujson.loads, f), start=1):
            problems.append(process_fn(line))
            if read_num % 1000 == 0:
                logger.info(f"Processed {read_num} lines")
            if debug_num is not None and read_num >= debug_num:
                break

    num_preds_w_count = [
        sum(p.count for p in problem.solutions) for problem in problems
    ]
    if len(set(num_preds_w_count)) > 1:
        logger.error(
            "All problems must have the same number of predictions with counts"
        )
        exit(1)

    if remove_comments or black_format:
        logger.info(
            f"Cleaning predictions with {remove_comments=}, {black_format=}"
        )
        problems = run_in_parallel(
            functools.partial(
                clean_predictions,
                remove_comments=remove_comments,
                black_format=black_format,
            ),
            problems,
            num_workers=num_workers,
            desc=f"Cleaning {task_name} predictions",
        )
    return task_name, problems


def get_eval_datasets(input_dir: Path):
    logger.info(f"Getting eval datasets from {input_dir}")
    for eval_dir in input_dir.iterdir():
        if not eval_dir.is_dir():
            continue

        gen_model = eval_dir.name
        for sampling_setup_dir in eval_dir.iterdir():
            if not sampling_setup_dir.is_dir():
                continue
            logger.info(f"Processing {gen_model} with {sampling_setup_dir}")
            datasets = list(sampling_setup_dir.glob("*exec.jsonl.gz"))
            if len(datasets) == 0:
                logger.error(f"No datasets found for {sampling_setup_dir}")
                continue

            sampling_setup = sampling_setup_dir.name
            for ds_file in datasets:
                yield gen_model, sampling_setup, ds_file


@click.command()
@click.option(
    "--input_dir",
    type=click.Path(exists=True, path_type=Path, file_okay=False),
    required=True,
    help="Input file with problems",
)
@click.option(
    "--output_dir",
    type=click.Path(path_type=Path),
    help="Output directory",
    default=Path(os.getenv("DEFAULT_OUTPUT_DIR", "outputs"), "eval_dataset"),
)
@click.option(
    "--hf_ds_dir",
    type=click.Path(exists=True, path_type=Path),
    help="HF dataset directory",
    default=None,
)
@click.option("--black_format", is_flag=True, help="Black format the programs")
@click.option(
    "--remove_comments", is_flag=True, help="Remove comments from the programs"
)
@click.option("--num_workers", type=int, default=1, help="Number of workers")
@click.option(
    "--debug_specific", type=str, default=None, help="Debug specific model"
)
@click.option(
    "--debug_num", type=int, default=None, help="Debug number of tasks"
)
def create_eval_data(
    input_dir: Path,
    output_dir: Path,
    black_format: bool,
    remove_comments: bool,
    num_workers: int,
    debug_specific: Optional[str],
    hf_ds_dir: Optional[Path],
    debug_num: Optional[int],
):

    logger.info("Creating evaluation data")
    ds_name = get_eval_ds_name(black_format, remove_comments)
    output_dir = output_dir / ds_name
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Creating output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Task directory: {output_dir}")
    shutil.rmtree(output_dir, ignore_errors=True)
    model_rows = defaultdict(list)
    found_tasks = defaultdict(set)

    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, setup_name, ds_file in get_eval_datasets(input_dir):

        if debug_specific and debug_specific not in ds_file.name:
            logger.info(f"Skipping {ds_file} because {debug_specific} is set")
            continue

        task_name, processed = process_pred_file(
            model_name,
            ds_file,
            remove_comments,
            black_format,
            num_workers,
            debug_num,
        )
        if not processed:
            logger.error(f"No processed data for {ds_file}")
            continue
        found_tasks[(model_name, setup_name)].add(task_name)
        for line in tqdm(
            map(asdict, processed),
            total=len(processed),
        ):
            line["model"] = model_name
            line["setup"] = setup_name
            line["dataset"] = task_name

            model_rows[(model_name, setup_name, task_name)].append(line)

    for (model_name, setup_name), tasks in found_tasks.items():
        missing_tasks = EXPECTED_TASKS - tasks
        if missing_tasks:
            logger.warning(f"Missing tasks for {model_name}: {missing_tasks}")
            for t in tasks:
                logger.info(f"Removing rows for {t}")
                del model_rows[(model_name, setup_name, t)]
        else:
            for t in tasks:
                with gzip.open(
                    output_dir / f"{model_name}_{setup_name}_{t}.jsonl.gz", "wt"
                ) as f:
                    for line in model_rows[(model_name, setup_name, t)]:
                        f.write(ujson.dumps(line) + "\n")

    if hf_ds_dir is None:
        return

    task_dir = hf_ds_dir / f"eval-{ds_name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(task_dir / "test", ignore_errors=True)
    config_list = {}
    expected_total_rows = 0
    expected_cfg_rows = defaultdict(lambda: 0)
    for (model_name, setup_name, task_name), rows in model_rows.items():
        use_setup_name = SETUP_NAME_TO_CLEAN.get(setup_name, setup_name)
        save_name = f"{task_name}_{use_setup_name}.jsonl.gz"
        model_dir = task_dir / "test" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        cfg_name = f"{model_name}_{use_setup_name}_{task_name}"
        config_list[cfg_name] = {
            "config_name": cfg_name,
            "data_files": [
                {
                    "split": "test",
                    "path": [f"test/{model_name}/{save_name}"],
                }
            ],
        }
        expected_total_rows += len(rows)
        expected_cfg_rows[cfg_name] += len(rows)
        with gzip.open(model_dir / save_name, "wt") as f:
            for row in rows:
                f.write(ujson.dumps(row) + "\n")
    readme_file = task_dir / "README.md"
    logger.info(f"Updating README {readme_file}")
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write("---\n")
        config_strs = yaml.dump(
            {
                **DEFAULT_CONFIG,
                "configs": DEFAULT_CONFIG_LIST + list(config_list.values()),
            }
        )
        f.write(config_strs)
        f.write("---\n")
        f.write(DEFAULT_README)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = load_dataset(
            str(task_dir.resolve().absolute()),
            cache_dir=tmp_dir,
            split="test",
        )
        logger.info(f"Default Dataset: {ds}")
        assert len(ds) == expected_total_rows
        for subset_name in config_list.keys():
            ds = load_dataset(
                str(task_dir.resolve().absolute()),
                cache_dir=tmp_dir,
                name=subset_name,
                split="test",
            )
            logger.info(f"{subset_name} Dataset: {ds}")
            assert len(ds) == expected_cfg_rows[subset_name]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    create_eval_data()
