"""Script to download execution data for code contests so that we dont need to download 30GB every time."""

import argparse
import functools
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

import ujson
from code_execution.eval_dataset.code_contests import REQUIRED_EXECUTION_KEYS
from code_execution.eval_dataset.code_contests import process_problem
from datasets import concatenate_datasets
from datasets import load_dataset
from tqdm import tqdm


def make_cc_problem(ex, i):
    return {
        **ex,
        **process_problem(ex),
        "idx": i,
    }


def cc_filter(ex):
    return len(ex["public_tests"].get("input", [])) > 0


def add_split(ex, i, split):
    return {"idx": i, "split": split, "task_id": f"code_contests/{i}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    train_dataset = load_dataset("deepmind/code_contests", split="train").map(
        functools.partial(add_split, split="train"),
        with_indices=True,
        num_proc=12,
    )
    val_dataset = load_dataset("deepmind/code_contests", split="valid").map(
        functools.partial(add_split, split="valid"),
        with_indices=True,
        num_proc=12,
    )
    dataset = (
        concatenate_datasets([train_dataset, val_dataset])
        .filter(cc_filter, num_proc=12)
        .map(
            functools.partial(make_cc_problem),
            with_indices=True,
            num_proc=12,
        )
    )
    dataset = dataset.remove_columns(
        [
            c
            for c in dataset.column_names
            if c
            not in REQUIRED_EXECUTION_KEYS.union(
                {"task_id", "split", "name", "description", "idx"}
            )
        ]
    )
    with gzip.open(args.output_dir / "code_contests.jsonl.gz", "wt") as f:
        for ex in tqdm(dataset, desc="Writing to file"):
            f.write(ujson.dumps(ex))
            f.write("\n")

    print(
        f"Saved {len(dataset):,} examples to {args.output_dir / 'code_contests.jsonl'}"
    )
    train_dataset = load_dataset("openai/gsm8k", "main", split="train").map(
        lambda ex, i: {"task_id": f"gsm8k/{i}", "split": "train", "idx": i},
        with_indices=True,
    )
    with gzip.open(args.output_dir / "gsm8k.jsonl.gz", "wt") as f:
        for ex in tqdm(train_dataset, desc="Writing to file"):
            f.write(ujson.dumps(ex))
            f.write("\n")


if __name__ == "__main__":
    main()
