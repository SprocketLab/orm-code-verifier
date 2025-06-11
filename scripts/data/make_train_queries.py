import functools
import gzip
import json
import sys
from pathlib import Path

import click
from bigcode_eval.tasks import TASK_REGISTRY
from datasets import concatenate_datasets
from datasets import load_dataset
from rich.console import Console
from transformers import AutoTokenizer

print(Path(__file__).parents[2].absolute())
sys.path.append(str(Path(__file__).parents[2].absolute()))
from src.evaluation import prompts
from src.modeling import MODEL_NAMES
from src.utils import CONSOLE
from src.utils import DEFAULT_OUT_DIR

CONSOLE = Console()


@click.command()
@click.option("--model_name", type=str, default="qc-inst-7b")
@click.option(
    "--output_dir", type=Path, default=Path(DEFAULT_OUT_DIR, "train_queries")
)
@click.option("--is_chat", is_flag=True)
@click.option("--shard_size", type=int, default=1000)
def main(model_name: str, output_dir: Path, is_chat: bool, shard_size: int):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAMES.get(model_name, model_name)
    )

    prompt_fn = functools.partial(
        prompts.get_prompt,
        is_chat=is_chat,
        model_name=MODEL_NAMES.get(model_name, model_name),
        tokenizer=tokenizer,
    )

    CONSOLE.print("Loading Code Contests dataset")
    train_dataset = load_dataset("deepmind/code_contests", split="train").map(
        lambda ex, i: {
            "split": "train",
            "task_id": f"code_contests/{i}",
        },
        with_indices=True,
        num_proc=8,
    )
    val_dataset = load_dataset("deepmind/code_contests", split="valid").map(
        lambda ex, i: {
            "split": "valid",
            "task_id": f"code_contests/{i}",
        },
        with_indices=True,
        num_proc=8,
    )
    dataset = concatenate_datasets([train_dataset, val_dataset]).filter(
        lambda ex: len(ex["public_tests"].get("input", [])) > 0, num_proc=8
    )
    dataset = dataset.map(
        lambda ex, i: {
            "query": prompt_fn(prompts.get_cc_prompt(ex, is_chat=is_chat)),
            "idx": i,
            "name": ex["name"],
            "split": ex["split"],
        },
        with_indices=True,
        remove_columns=[
            c
            for c in dataset.column_names
            if c not in {"task_id", "split", "name"}
        ],
        load_from_cache_file=False,
        desc="Preparing Code Contests",
        num_proc=8,
    )
    CONSOLE.print(f"Example CodeContests query:\n{dataset[0]['query']}")
    CONSOLE.print(f"{len(dataset):,} examples in CodeContests dataset")

    # Write shards
    cc_output_dir = (
        output_dir / f"{model_name.replace('/', '_')}" / "code_contests"
    )
    cc_output_dir.mkdir(parents=True, exist_ok=True)

    num_shards = (len(dataset) + shard_size - 1) // shard_size
    CONSOLE.print(f"Writing {num_shards} shards of size {shard_size}")

    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, len(dataset))
        shard = dataset.select(range(start_idx, end_idx))
        CONSOLE.print(
            f"Writing shard {shard_idx + 1}/{num_shards} to {shard_size} examples"
        )
        output_file = cc_output_dir / f"shard_{shard_idx:05d}.jsonl.gz"
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            for item in shard:
                f.write(json.dumps(item) + "\n")

        CONSOLE.print(
            f"Wrote shard {shard_idx + 1}/{num_shards} to {output_file}"
        )

    CONSOLE.print("Loading GSM8K dataset")
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    CONSOLE.print("GSM8K dataset loaded, processing")
    gsm_task = TASK_REGISTRY["pal-gsm8k-majority_voting"]()
    icl = prompts.make_gsm_icl(
        gsm_task.fewshot_examples(), question_is_doc=False, is_chat=is_chat
    )

    dataset = dataset.map(
        lambda e, i: {
            "idx": i,
            "query": prompt_fn(prompts.get_gsm_prompt(e, icl)),
        },
        with_indices=True,
        remove_columns=[
            c for c in dataset.column_names if c in {"task_id", "split", "name"}
        ],
        desc="Making GSM8K Query",
        num_proc=8,
    )
    CONSOLE.print(f"Example GSM8K query:\n{dataset[0]['query']}")
    CONSOLE.print(f"{len(dataset):,} examples in GSM8K dataset")

    # Write shards
    gsm_output_dir = output_dir / f"{model_name.replace('/', '_')}" / "gsm8k"
    gsm_output_dir.mkdir(parents=True, exist_ok=True)

    num_shards = (len(dataset) + shard_size - 1) // shard_size
    CONSOLE.print(f"Writing {num_shards} shards of size {shard_size}")

    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, len(dataset))
        shard = dataset.select(range(start_idx, end_idx))
        CONSOLE.print(
            f"Writing shard {shard_idx + 1}/{num_shards} to {shard_size} examples"
        )
        output_file = gsm_output_dir / f"shard_{shard_idx:05d}.jsonl.gz"
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            for item in shard:
                f.write(json.dumps(item) + "\n")

        CONSOLE.print(
            f"Wrote shard {shard_idx + 1}/{num_shards} to {output_file}"
        )


if __name__ == "__main__":
    main()
