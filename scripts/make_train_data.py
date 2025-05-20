import functools
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import click
from datasets import concatenate_datasets
from datasets import load_dataset
from rich.logging import RichHandler

ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT.resolve().absolute()))
from src.training.build_dataset import make_dataset_dir_name
from src.utils import CONSOLE
from src.utils import DEFAULT_OUT_DIR
from src.utils import format_with_black

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[
        RichHandler(
            console=CONSOLE,
            level=logging.DEBUG,
        ),
    ],
)

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("black").setLevel(logging.WARNING)
logging.getLogger("blib2to3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)


def should_keep_problem(
    problem: Dict,
    require_pf: bool,
    include_syntax: bool,
    pass_level: str,
    only_dataset: Optional[str] = None,
) -> bool:

    valid_sols = [s for s in problem["predictions"] if len(s["code"]) < 10000]
    syntax_sols = [
        s for s in problem["invalid_syntax"] if len(s["code"]) < 10000
    ]
    if len(valid_sols) == 0:
        return False

    num_passed = (
        sum(s["passed"] for s in valid_sols)
        if pass_level == "all"
        else sum(s["passed_public"] for s in valid_sols)
    )
    num_failed = len(valid_sols) - num_passed
    if include_syntax:
        num_failed += len(syntax_sols)
    if num_passed == 0:
        return False
    if require_pf and (num_passed == 0 or num_failed == 0):
        return False
    if only_dataset is not None:
        return problem["source"] == only_dataset
    return True


def process_predictions(
    problem: Dict,
    black_format: bool,
    remove_comments: bool,
) -> Dict:
    predictions = problem["predictions"]
    for i, p in enumerate(predictions):
        if len(p["code"]) > 10000:
            continue
        if black_format:
            p["code"] = format_with_black(p["code"])
        if remove_comments:
            p["code"] = remove_comments(p["code"])
        predictions[i] = p
    problem["predictions"] = predictions
    return problem


@click.command()
@click.option("--num_proc", type=int, default=4)
@click.option("--black_format", is_flag=True)
@click.option("--remove_comments", is_flag=True)
@click.option(
    "--pass_level", type=click.Choice(["all", "public"]), default="all"
)
@click.option("--require_pf", is_flag=True)
@click.option("--include_syntax", is_flag=True)
@click.option(
    "--only_dataset",
    type=click.Choice(["code_contests", "gsm8k", None]),
    default=None,
)
@click.option("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
def cli(
    num_proc: int,
    black_format: bool,
    remove_comments: bool,
    pass_level: str,
    require_pf: bool,
    include_syntax: bool,
    only_dataset: Optional[str],
    output_dir: Path,
):
    logger.info("Making train dataset")

    logger.info("Loading Code Contests dataset")
    ds = load_dataset("anon/synth-train-prog")
    logger.info(f"Loaded original dataset: {ds}")
    ds = ds.filter(
        lambda x: should_keep_problem(
            x,
            require_pf=require_pf,
            include_syntax=include_syntax,
            pass_level=pass_level,
            only_dataset=only_dataset,
        ),
        desc="Filtering",
    )
    ds = ds.map(
        functools.partial(
            process_predictions,
            black_format=black_format,
            remove_comments=remove_comments,
        ),
        num_proc=num_proc,
    )
    logger.info(f"Finished processing training examples: {ds}")

    use_name = make_dataset_dir_name(
        pass_level=pass_level,
        include_syntax=include_syntax,
        black_format=black_format,
        remove_comments=remove_comments,
        only_dataset=only_dataset,
        require_pf=require_pf,
    )
    logger.info(f"Source Counts: {Counter(ds['train']['source'])}")
    logger.info(f"Source Counts: {Counter(ds['validation']['source'])}")
    logger.info(f"Saving dataset to {use_name}")
    ds.save_to_disk(str(Path(output_dir, "train_data", use_name)))


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
