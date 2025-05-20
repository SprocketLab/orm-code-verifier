import gzip
import logging
import math
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import ujson
from datasets import load_dataset
from rich.logging import RichHandler
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.getLogger("parso").setLevel(logging.WARNING)
logging.getLogger("black").setLevel(logging.WARNING)
logging.getLogger("blib2to3.pgen2.driver").setLevel(logging.WARNING)


README = """---
license: apache-2.0
configs:
  - config_name: default
    data_files:
      - split: train
        path:
          - data/train/*.jsonl.gz
      - split: validation
        path:
          - data/validation/*.jsonl.gz
---"""


def make_cc_train_data(data_dir: Path, debug: bool = False):

    logger.info(f"Making code contests data from {data_dir}")
    validation_problems = load_dataset("deepmind/code_contests", split="valid")
    validation_problems = set(validation_problems["name"])
    logger.info(f"Loaded {len(validation_problems)} validation problems")
    num_shards = 0
    num_train_found = 0
    validation = []
    train = []
    for file in sorted(
        data_dir.glob("*.jsonl.gz"),
        key=lambda x: int(x.stem.split(".")[0].split("_")[-1]),
    ):
        logger.info(f"Reading {file}")
        with gzip.open(file, "rt") as f:
            for i, line in enumerate(map(ujson.loads, f), start=1):
                new_line = {
                    "task_id": line["task_id"],
                    "description": line["description"],
                    "predictions": [],
                    "invalid_syntax": [],
                    "source": "code_contests",
                }
                has_private = 1 in set(line["test_types"])
                has_generated = 2 in set(line["test_types"])
                num_tests = len(line["inputs"])

                num_passed = 0
                num_passed_public = 0
                num_passed_private = 0
                for sol_group in ["predictions", "invalid_syntax"]:
                    for v in line[sol_group]:
                        new_prd = {
                            k: v.get(k, False if "passed" in k else None)
                            for k in {
                                "code",
                                "passed",
                                "passed_public",
                                "count",
                            }
                        }
                        new_line[sol_group].append(new_prd)
                        new_prd["pct_passed"] = (
                            sum(v.get("outcomes", [])) / num_tests
                        )
                        if has_private:
                            passed_private = v.get("passed_private", 0) == 1
                        elif has_generated:
                            passed_private = v.get("passed_generated", 0) == 1
                        else:
                            passed_private = v["passed"]
                        new_prd["passed_private"] = passed_private
                        if new_prd["passed"]:
                            num_passed += 1
                        if new_prd["passed_public"]:
                            num_passed_public += 1
                        if new_prd["passed_private"]:
                            num_passed_private += 1
                new_line["num_passed"] = num_passed
                new_line["num_passed_public"] = num_passed_public
                new_line["num_passed_private"] = num_passed_private
                if (
                    line["name"] in validation_problems
                    or line["split"] == "validation"
                ):
                    validation.append(new_line)
                    validation_problems.remove(line["name"])
                else:
                    num_train_found += 1
                    train.append(new_line)

                if i % 100 == 0:
                    logger.info(
                        f"Read {i} lines. Num Train={num_train_found:,}. {len(validation):,} validation examples"
                    )

        if debug:
            break
        num_shards += 1
    logger.info(
        f"Found {num_train_found:,} train and {len(validation):,} validation problems"
    )
    return train, validation


def make_gsm_train_data(
    data_dir: Path,
    seed: int,
    pct_validation: float = 0.1,
    max_validation: int = 250,
    debug: bool = False,
):

    logger.info(f"Making gsm8k data from {data_dir}")
    train = []
    for file in data_dir.glob("*.jsonl.gz"):
        logger.info(f"Reading {file}")
        with gzip.open(file, "rt") as f:
            for i, line in enumerate(map(ujson.loads, f), start=1):
                new_line = {
                    "task_id": line["task_id"],
                    "description": line["question"],
                    "predictions": [],
                    "invalid_syntax": [],
                    "source": "gsm8k",
                }

                num_passed = 0
                num_passed_public = 0
                for sol_group in ["predictions", "invalid_syntax"]:
                    for v in line[sol_group]:
                        new_prd = {
                            k: v.get(k, False if "passed" in k else None)
                            for k in {
                                "code",
                                "passed",
                                "count",
                            }
                        }
                        new_prd["passed_public"] = new_prd["passed"]
                        new_prd["pct_passed"] = 1 if new_prd["passed"] else 0
                        new_line[sol_group].append(new_prd)
                        if new_prd["passed"]:
                            num_passed += 1
                            num_passed_public += 1
                        new_prd["passed_private"] = new_prd["passed"]
                new_line["num_passed"] = num_passed
                new_line["num_passed_public"] = num_passed_public
                new_line["num_passed_private"] = num_passed_public
                train.append(new_line)

                if i % 100 == 0:
                    logger.info(f"Read {i} lines. Num Train={len(train):,}")
        if debug:
            break

    in_val = min(math.ceil(len(train) * pct_validation), max_validation)
    logger.info(
        f"Found {len(train):,} train problems, using {in_val}({in_val/len(train):0.2%}) problems for validation"
    )
    rng = np.random.default_rng(seed)
    pool = list(
        filter(
            lambda x: 0
            < sum(map(lambda y: y["passed"], train[x]["predictions"]))
            < len(train[x]["predictions"]),
            range(len(train)),
        )
    )

    validation_idx = set(rng.choice(pool, in_val, replace=False))

    validation = [train[i] for i in validation_idx]
    train = [train[i] for i in range(len(train)) if i not in validation_idx]
    return train, validation


@click.command()
@click.argument("data_dir", type=click.Path(path_type=Path))
@click.argument("out_dir", type=click.Path(path_type=Path))
@click.option("--seed", default=1, help="Random seed", type=click.INT)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing data",
)
@click.option(
    "--pct_validation", type=click.FLOAT, default=0.1, show_default=True
)
@click.option("--max_validation", type=click.INT, default=100)
@click.option("--debug", is_flag=True, help="Debug mode")
def cli(
    data_dir: Path,
    out_dir: Path,
    seed: int,
    overwrite: bool,
    debug: bool,
    pct_validation: float,
    max_validation: int,
):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(
                level=logging.DEBUG,
            )
        ],
    )

    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("black").setLevel(logging.WARNING)
    logging.getLogger("blib2to3").setLevel(logging.WARNING)
    logging.getLogger("code_execution").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = out_dir / "data" / "train"
    validation_dir = out_dir / "data" / "validation"
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(validation_dir, ignore_errors=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    cc_train, cc_validation = make_cc_train_data(
        data_dir=data_dir / "code_contests",
        debug=debug,
    )
    gsm_train, gsm_validation = make_gsm_train_data(
        data_dir=data_dir / "gsm8k",
        seed=seed,
        pct_validation=pct_validation,
        max_validation=max_validation,
        debug=debug,
    )
    train_data = cc_train + gsm_train
    validation_data = cc_validation + gsm_validation
    for save_dir, data in [
        (train_dir, train_data),
        (validation_dir, validation_data),
    ]:
        per_shard = 1000
        num_shards = math.ceil(len(data) / per_shard)
        for i in range(num_shards):
            start = i * per_shard
            end = min(start + per_shard, len(data))
            logger.info(f"Writing examples {(start,end)} to shard {i}")
            with gzip.open(save_dir / f"{i}.jsonl.gz", "wt") as f:
                for line in data[start:end]:
                    f.write(ujson.dumps(line) + "\n")

    with (out_dir / "README.md").open("w") as f:
        f.write(f"{README}\n\n# CORM Train Data")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = load_dataset(str(out_dir.resolve().absolute()), cache_dir=tmp_dir)

    train_sources = Counter(ds["train"]["source"])
    validation_sources = Counter(ds["validation"]["source"])
    logger.info(f"Train: {len(ds['train'])}, expecting {len(train_data)}")
    logger.info(
        f"Validation: {len(ds['validation'])}, expecting {len(validation_data)}"
    )

    logger.info(f"Train sources: {train_sources}")
    logger.info(f"Validation sources: {validation_sources}")
    logger.info(f"Train columns: {ds['train'].column_names}")
    logger.info(f"Validation columns: {ds['validation'].column_names}")


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
