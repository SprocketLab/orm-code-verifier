"""Executes a single shard of synthetic data."""

import functools
import gzip
import logging
import sys
import tempfile
from collections import OrderedDict
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional

import click
import code_execution
import numpy as np
import ujson
from code_execution.eval_dataset import code_contests
from code_execution.eval_dataset import gsm8k
from code_execution.execution import safe_execute
from datasets import Dataset
from datasets import concatenate_datasets
from datasets import load_dataset
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

logger = logging.getLogger(__name__)


def dedup_and_remove_syntax_err(problem_dict: Dict, postproc_fn: Callable):
    unique_predictions = OrderedDict()

    preds = problem_dict.pop("predictions")
    logprobs = problem_dict.pop("cum_log_probs")
    num_tokens = problem_dict.pop("num_tokens")

    for cid, (pred, lp, nt) in enumerate(zip(preds, logprobs, num_tokens)):
        clean_pred = postproc_fn(pred)
        clean_pred = clean_pred.strip()
        if clean_pred not in unique_predictions:
            unique_predictions[clean_pred] = {
                "count": 0,
                "valid_syntax": False,
                "cum_logprob": [],
                "num_tokens": [],
                "completion_ids": [],
            }
        unique_predictions[clean_pred]["count"] += 1
        is_valid_syntax = code_execution.is_valid_python(clean_pred)
        unique_predictions[clean_pred]["valid_syntax"] = is_valid_syntax
        unique_predictions[clean_pred]["cum_logprob"].append(lp)
        unique_predictions[clean_pred]["num_tokens"].append(nt)
        unique_predictions[clean_pred]["completion_ids"].append(cid)
    assert sum(data["count"] for data in unique_predictions.values()) == len(
        preds
    )
    new_preds = []
    invalid_syntax = []
    for pred, data in unique_predictions.items():
        data["cum_logprob"] = min(data["cum_logprob"])
        data["num_tokens"] = min(data["num_tokens"])
        data["completion_ids"] = min(data["completion_ids"])

        if data.pop("valid_syntax"):
            new_preds.append({"code": pred, **data})
        else:
            invalid_syntax.append(
                {
                    "code": pred,
                    "passed": False,
                    "stderr": [],
                    "stdout": [],
                    **data,
                }
            )
    problem_dict["predictions"] = new_preds
    problem_dict["invalid_syntax"] = invalid_syntax
    return problem_dict


@dataclass
class Config:
    num_workers: int = 1
    output_dir: Path = Path("outputs")
    debug: bool = False
    timeout: float = 10.0
    command_timeout: float = 10.0
    debug_num: int = 100
    debug_per: int = None
    seed: int = 1
    mutations_per_program: int = 1
    output_name: str = None


pass_context = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option(
    "--num_workers",
    type=click.INT,
    default=Config.num_workers,
    help="Number of workers to use for data loading.",
)
@click.option(
    "--output_dir",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    help="Output directory.",
    default=Config.output_dir,
)
@click.option(
    "--log_dir",
    type=click.Path(exists=False, file_okay=True, path_type=Path),
    default=Path("logs"),
    help="Log file.",
)
@click.option("--verbose", is_flag=True, help="Verbose logging.")
@click.option("--debug", is_flag=True, help="Debug mode.")
@click.option(
    "--timeout",
    type=click.FLOAT,
    default=Config.timeout,
    help="Timeout for each program execution.",
)
@click.option(
    "--command_timeout",
    type=click.FLOAT,
    default=Config.command_timeout,
    help="Timeout for each command execution.",
)
@click.option(
    "--debug_num",
    type=click.INT,
    default=Config.debug_num,
    help="Debug number.",
)
@click.option(
    "--seed", type=click.INT, default=Config.seed, help="Random seed."
)
@click.option(
    "--debug_per",
    type=click.INT,
    default=Config.debug_per,
    help="Debug per.",
)
@click.option(
    "--output_name", type=click.STRING, help="Output name.", default=None
)
@pass_context
def execute_shard(
    ctx: Config,
    log_dir: Path,
    verbose: bool,
    **kwargs,
):
    log_dir.mkdir(parents=True, exist_ok=True)
    console = Console(width=128)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=console,
                level=logging.INFO if not verbose else logging.DEBUG,
            ),
            logging.FileHandler(
                log_dir / "execute_shard.log",
            ),
        ],
    )
    if len(logging.getLogger().handlers) > 1:
        logging.getLogger().handlers[1].setLevel(logging.DEBUG)
        logging.getLogger().handlers[1].setFormatter(
            logging.Formatter(
                "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("black").setLevel(logging.WARNING)
    logging.getLogger("blib2to3").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)

    ctx.obj = Config(
        **kwargs,
    )
    logger.info("Starting execution")


def _process_file(
    line_reader: Generator[Dict, None, None],
    dataset: Dict,
    validate_fn: Callable,
    postproc_fn: Callable,
    cfg: Config,
):
    total_programs = 0
    problems_w_preds = []
    for line in map(ujson.loads, line_reader):
        meta = dataset[int(line["idx"])]
        if not validate_fn(line, meta):
            logger.error(f"{line['idx']} failed validation.")
            exit(1)

        # Do this to make the keys easier to deal with later. Otherwise very nested keys.
        completions = line.pop("predictions")
        cum_log_probs = line.pop("cum_log_probs")
        num_tokens = line.pop("num_tokens")
        if (
            cfg.debug
            and cfg.debug_per is not None
            and len(completions) >= cfg.debug_per
        ):
            completions = completions[: cfg.debug_per]
            cum_log_probs = cum_log_probs[: cfg.debug_per]
            num_tokens = num_tokens[: cfg.debug_per]

        line.update(
            {
                "predictions": completions,
                "cum_log_probs": cum_log_probs,
                "num_tokens": num_tokens,
                **meta,
            }
        )
        line = dedup_and_remove_syntax_err(line, postproc_fn)

        line.pop("query")
        line["_idx"] = len(problems_w_preds)
        problems_w_preds.append(line)
        total_programs += len(line["predictions"])
        if len(problems_w_preds) % 100 == 0:
            logger.info(
                f"Read {total_programs:,} programs for {len(problems_w_preds):,} predictions"
            )
        if cfg.debug and len(problems_w_preds) >= cfg.debug_num:
            logger.warning(
                f"Debug limit reached, stopping at {len(problems_w_preds)}"
            )
            break
    return total_programs, problems_w_preds


def read_shard(
    cfg: Config,
    shard_path: Path,
    dataset: Dict,
    validate_fn: Callable,
    postproc_fn: Callable,
):
    logger.info(f"Reading '{shard_path}'")

    if shard_path.suffix == ".gz":
        logger.info("Reading gzipped shard")
        with gzip.open(shard_path, "rt") as f:
            total_programs, problems_w_preds = _process_file(
                f, dataset, validate_fn, postproc_fn, cfg
            )
    else:
        logger.info("Reading uncompressed shard")
        with shard_path.open("r", encoding="utf-8") as f:
            total_programs, problems_w_preds = _process_file(
                f, dataset, validate_fn, postproc_fn, cfg
            )

    logger.info(
        f"Found {total_programs:,} programs for {len(problems_w_preds):,} problems"
    )

    mean_unique = np.mean(
        [
            len(p["predictions"]) + len(p["invalid_syntax"])
            for p in problems_w_preds
        ]
    )
    logger.info(f"Average of {mean_unique:.2f} unique predictions per problem")
    return problems_w_preds


def save_executed_shard(cfg: Config, save_file_name: str, results: List[dict]):

    logger.info(f"Saving to '{cfg.output_dir}'")
    logger.info(f"Saving to '{save_file_name}.jsonl.gz'")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_file = cfg.output_dir / f"{save_file_name}.jsonl.gz"

    metrics = defaultdict(list)

    with gzip.open(save_file, "wt", encoding="utf-8") as f:
        for prob in results:
            num_passed = num_failed = 0
            for p in prob["predictions"]:
                if p["passed"]:
                    num_passed += 1

                else:
                    num_failed += 1

            metrics["Passed"].append(num_passed)
            metrics["Pairs"].append(min(num_passed, num_failed))

            f.write(ujson.dumps(prob) + "\n")

    metrics = {k: np.array(v) for k, v in metrics.items()}
    logger.info("Num Problems With:")
    for k, v in sorted(metrics.items(), key=lambda x: x[0]):
        m = v > 0
        logger.info(f"\t{k}: {m.sum():,} ({m.mean():.2%})")

    logger.info("Averages:")
    for k, v in sorted(metrics.items(), key=lambda x: x[0]):
        logger.info(f"\t{k}: {v.mean():.2f} ({v.std():.2f})")


@execute_shard.command("code_contests")
@click.argument(
    "shard_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "dataset_info_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@pass_context
def execute_code_contests_shard(
    ctx: Config,
    shard_path: Path,
    dataset_info_path: Path,
):
    cfg = ctx.obj
    if cfg.output_dir != Path():
        cfg.output_dir = cfg.output_dir / "code_contests"
    logger.info(f"Saving to '{cfg.output_dir}'")

    logger.info("Executing code contests shard")
    with gzip.open(dataset_info_path, "rt", encoding="utf-8") as f:
        ds_info = {l["idx"]: l for l in map(ujson.loads, f)}

    logger.debug(f"Loaded {len(ds_info)} problems")
    data = read_shard(
        cfg,
        shard_path,
        ds_info,
        lambda l, m: all(
            l.get(k, m[k]) == m[k] for k in ["name", "task_id", "split"]
        ),
        lambda p: p.split("\n```")[0],
    )

    execute_fn = functools.partial(
        code_contests.evaluate,
        num_workers=cfg.num_workers,
        first_command_timeout=cfg.timeout,
        command_timeout=cfg.command_timeout,
        solution_list_key="predictions",
        solution_str_key="code",
        early_stopping=True,
        num_stdout_save=1,
    )

    _, results = execute_fn(
        predictions=data,
    )

    save_executed_shard(cfg, cfg.output_name or shard_path.stem, results)
    num_passed_all = 0
    num_passed_public = 0
    num_passed_public_private = 0
    for r in results:
        has_passed_public = False
        has_passed_private = False
        has_passed_all = False
        for p in r["predictions"]:
            if p["passed"]:
                has_passed_all = True
            if p["passed_public"]:
                has_passed_public = True
                if p.get("passed_private", False):
                    has_passed_private = True
        num_passed_public += has_passed_public
        num_passed_public_private += has_passed_private
        num_passed_all += has_passed_all
    logger.info(
        f"{num_passed_public:,}/{len(results):,} have at least one program that passes the public tests"
    )
    logger.info(
        f"{num_passed_public_private:,}/{len(results):,} problems have at least one program that passes both the public and private tests"
    )
    logger.info(
        f"{num_passed_all:,}/{len(results):,} problems have at least one program that passes all tests"
    )


@execute_shard.command("gsm8k")
@click.argument(
    "shard_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "dataset_info_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@pass_context
def execute_gsm8k_shard(
    ctx: Config,
    shard_path: Path,
    dataset_info_path: Path,
):
    cfg = ctx.obj
    if cfg.output_dir != Path():
        cfg.output_dir = cfg.output_dir / "gsm8k"
    logger.info(f"Saving to '{cfg.output_dir}'")

    logger.info("Executing gsm8k shard")
    with gzip.open(dataset_info_path, "rt", encoding="utf-8") as f:
        ds_info = {l["idx"]: l for l in map(ujson.loads, f)}

    logger.debug(f"Loaded {len(ds_info)} problems")
    data = read_shard(
        cfg,
        shard_path,
        ds_info,
        lambda l, m: l["question"] == m["question"]
        and l["answer"] == m["answer"],
        lambda p: p.split("\n```")[0],
    )

    execute_fn = functools.partial(
        gsm8k.evaluate,
        num_workers=cfg.num_workers,
        timeout=cfg.timeout,
        solution_list_key="predictions",
        solution_str_key="code",
    )

    metrics, results = execute_fn(
        predictions=data,
    )
    for k, v in sorted(metrics.items(), key=lambda x: x[0]):
        p_val = f"{v:0.2f}" if isinstance(v, float) else v
        logger.info(f"{k:>24}: {p_val}")

    save_executed_shard(cfg, cfg.output_name or shard_path.stem, results)


if __name__ == "__main__":
    execute_shard()  # pylint: disable=no-value-for-parameter
