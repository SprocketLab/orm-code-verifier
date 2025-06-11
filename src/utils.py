import ast
import difflib
import functools
import itertools
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import black
import code_execution
import code_execution.utils
import git
import numpy as np
import torch
import transformers
import wandb
from datasets import Dataset
from jinja2 import Environment
from omegaconf import DictConfig
from omegaconf import OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)
PROJECT_NAME = "corm-project-final"
CONSOLE = Console(width=128)
BLACK_MODE = black.FileMode(
    target_versions={black.TargetVersion.PY311},
    line_length=80,
    is_pyi=False,
    string_normalization=True,
)
JINJA_ENV = Environment()
DEFAULT_OUT_DIR = os.environ.get("DEFAULT_OUTPUT_DIR", "outputs")

if "DEFAULT_OUTPUT_DIR" in os.environ and "f24-project" not in DEFAULT_OUT_DIR:
    DEFAULT_OUT_DIR = os.path.join(DEFAULT_OUT_DIR, "f24-project")

PREPROC_DATA_PATH = Path(DEFAULT_OUT_DIR, "preprocessed_datasets")


DEBUG_OS_KEY = "DEBUGGING_MODE"
CURRENT_OUT_DIR_KEY = "CURRENT_OUTPUT_DIR"
DEBUG_PRINTING_ENABLED = False


def is_debugging_enabled() -> bool:
    return DEBUG_PRINTING_ENABLED


def get_current_out_dir() -> Path:
    return Path(os.environ.get(CURRENT_OUT_DIR_KEY, DEFAULT_OUT_DIR))


class PaddingCollator:
    def __init__(
        self,
        pad_token,
        padding_side="right",
        special_pad_token_map=None,
        cast_label_float=False,
    ):
        self.pad_token = pad_token
        self.padding_side = padding_side
        self.special_pad_token_map = special_pad_token_map or {}
        self.cast_label_float = cast_label_float

    def _pad(self, seq, longest, pad_token):
        to_pad = longest - len(seq)

        if self.padding_side == "right":
            return seq + [pad_token] * to_pad
        else:
            return [pad_token] * to_pad + seq

    def __call__(self, features):
        batch = {}
        for k in features[0].keys():
            values = [f[k] for f in features]
            if not isinstance(values[0], list):
                if k == "idx":
                    batch[k] = torch.tensor(values, dtype=torch.long)
                else:
                    batch[k] = torch.tensor(values, dtype=torch.float)
            else:
                longest = max(map(len, values))
                if k == "input_ids":
                    pad_token = self.pad_token
                else:
                    pad_token = self.special_pad_token_map.get(k, 0)
                batch[k] = torch.tensor(
                    [
                        self._pad(
                            v,
                            longest,
                            pad_token=pad_token,
                        )
                        for v in values
                    ]
                )
            if self.cast_label_float and k == "labels":
                batch[k] = batch[k].float()
        return batch


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.
    Args:
        seed (`int`): The seed to set.
    """
    logger.info("Setting seed to {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    return np.random.default_rng(seed)


def get_diff(a, b):
    return list(difflib.unified_diff(a.split("\n"), b.split("\n"), lineterm=""))


def format_with_black(code):
    try:
        with code_execution.utils.swallow_io():
            code = black.format_file_contents(code, fast=False, mode=BLACK_MODE)
    except (
        black.NothingChanged,
        black.InvalidInput,
        black.parsing.ASTSafetyError,
        IndentationError,
        KeyError,
    ):
        pass
    finally:
        # Make sure there's a newline after the content
        if code and code[-1] != "\n":
            code += "\n"
    return code


def remove_comments(code):

    tree = code_execution.safe_ast_parse(code)
    if tree is not None:
        # Removes the comments.
        try:
            sol = ast.unparse(tree)
        except RecursionError:
            sol = code
    else:
        sol = code
    return sol


def clean_solution(sol):
    sol = remove_comments(sol)
    return format_with_black(sol).strip()


def get_exp_dir(
    dir_path: Path,
    model_name: str,
    exp_name: str,
    sub_dir: str = None,
    overwrite: bool = False,
    resume: bool = False,
    include_date: bool = False,
    delete_existing: bool = False,
    group_name=None,
) -> Path:
    leaf_dir_name = model_name
    if group_name is not None:
        leaf_dir_name = f"{leaf_dir_name}-{group_name}"
    out_dir = dir_path / leaf_dir_name
    if include_date:
        current_date = datetime.now().strftime("%Y%m%d")
        out_dir = out_dir / current_date / exp_name
    else:
        out_dir = out_dir / exp_name
    if sub_dir:
        out_dir = out_dir / sub_dir
    if out_dir.exists():
        if resume:
            logger.info(f"Resuming from {out_dir}")
            return out_dir
        elif not overwrite:
            raise FileExistsError(f"Directory {out_dir} already exists")
        elif delete_existing:

            CONSOLE.print(f"WARNING: Deleting existing directory {out_dir}")
            shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_current_repo_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def seconds_to_human(seconds):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


def setup_env(
    job_type: str,
    cfg,
    out_dir: Path,
    run_name,
    overwrite: bool = False,
    debug: bool = False,
    verbose: bool = False,
    include_date: bool = False,
    resume: bool = False,
    config_save_name: str = "config.yaml",
    delete_existing: bool = False,
    group_name: str = None,
) -> Tuple[str, Path]:

    global DEBUG_PRINTING_ENABLED
    if debug:
        if not run_name.startswith("debug-"):
            run_name = f"debug-{run_name}"
        DEBUG_PRINTING_ENABLED = True
    else:
        DEBUG_PRINTING_ENABLED = False

    out_dir = get_exp_dir(
        dir_path=out_dir,
        exp_name=run_name,
        model_name=cfg.model.name,
        overwrite=overwrite,
        include_date=include_date,
        delete_existing=delete_existing,
        resume=resume,
        group_name=group_name,
    )

    file_handler = logging.FileHandler(out_dir / f"{job_type}.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=CONSOLE,
                level=logging.DEBUG if verbose else logging.INFO,
            ),
            file_handler,
        ],
    )
    if len(logging.getLogger().handlers) > 1:
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

    logger.info("Starting %s", run_name)
    logger.info(
        "Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    )
    logger.info("Output directory: %s", out_dir)
    with (out_dir / config_save_name).open("w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True))
    os.environ[CURRENT_OUT_DIR_KEY] = str(out_dir.absolute())
    return run_name, out_dir


def sanitize_value(value):
    if isinstance(value, Path):
        return str(value)
    return value


def flatten(nested_dict, parent=None, sep=".", sanitize: bool = False):
    out = {}
    for k, v in nested_dict.items():
        try:

            new_key = str(parent) + sep + str(k) if parent else str(k)
        except ValueError as e:
            print(parent, sep, k)
            print(f"{type(parent)=}, {type(sep)=}, {type(k)=}")
            raise e
        if isinstance(v, dict):
            out.update(flatten(v, new_key, sep=sep).items())
        else:
            if sanitize:
                v = sanitize_value(v)

            out[new_key] = v
    return out


def setup_wandb(
    cfg: DictConfig,
    job_type: str,
    run_name: str,
    group: str,
    run_dir: Path,
    tags: List[str] = None,
):
    logger.debug(f"Setting up wandb for {group}/{run_name}({job_type=})")
    flattened_cfg = flatten(OmegaConf.to_container(cfg, resolve=True))
    run = wandb.init(
        project=cfg.get("project_name", PROJECT_NAME) or PROJECT_NAME,
        entity=cfg.get("entity_name", None),
        name=run_name,
        config=flattened_cfg,
        job_type=job_type,
        group=group,
        dir=run_dir,
        tags=tags or [],
    )
    return run


class TokenBatchSampler(Sampler[List[int]]):
    """
    Yields batches of indices where the total number of tokens (including padding)
    in each batch is approximately constant.
    """

    def __init__(
        self,
        lengths: List[int],
        max_tokens_per_batch: int,
        drop_last: bool = False,
        pad_to_multiple_of: int = 1,
    ):
        """
        Args:
            rng: Random number generator
            lengths: List of sequence lengths for each item in the dataset
            max_tokens_per_batch: Maximum number of tokens (including padding) per batch
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
            pad_to_multiple_of: If specified, pads to the next multiple of this value
        """
        self.lengths = lengths
        self.max_tokens_per_batch = max_tokens_per_batch
        self.drop_last = drop_last
        self.pad_to_multiple_of = pad_to_multiple_of

    def __iter__(self) -> Iterator[List[int]]:

        batch = []
        # Store the current longest sequence in the batch. We are operating
        # with the assumption the batch is being padded, so whatever the
        # longest sequence is, we will pad to that length.
        longest_seq = 0
        total_batch_tokens = 0

        for idx, sequence_length in sorted(
            enumerate(self.lengths),
            key=lambda x: x[1],
            reverse=True,
        ):

            # Determine what the new longest will be. If we need to pad the
            # current sequence to the next multiple of pad_to_multiple_of, do
            # that.
            use_seq_length = (
                sequence_length
                + self.pad_to_multiple_of
                - sequence_length % self.pad_to_multiple_of
            )
            new_longest = max(
                longest_seq,
                use_seq_length,
            )

            # Calculate new total tokens with the current sequence included
            new_batch_tokens = (len(batch) + 1) * new_longest

            if new_batch_tokens > self.max_tokens_per_batch:

                if batch:
                    # Current sequence would exceed token limit
                    # Yield current batch and start a new one
                    yield batch

                    batch = [idx]
                    longest_seq = use_seq_length

                else:
                    # Current sequence is too long to fit in a batch, yield
                    # just itself.
                    yield [idx]
                    batch = []
                    longest_seq = 0
            elif new_batch_tokens == self.max_tokens_per_batch:
                # Current sequence would fill the batch exactly
                batch.append(idx)
                yield batch
                batch = []
                longest_seq = 0

            else:

                # Add to current batch
                batch.append(idx)
                # Update batch statistics
                total_batch_tokens += sequence_length
                longest_seq = new_longest

        # Handle last batch
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # This is an approximation
        total_tokens = sum(self.lengths)
        if self.drop_last:
            return total_tokens // self.max_tokens_per_batch
        return (
            total_tokens + self.max_tokens_per_batch - 1
        ) // self.max_tokens_per_batch


def get_dynamic_token_dataloader(
    dataset: Dataset,
    max_tokens_per_batch: int,
    collate_fn: Callable,
    input_ids_key: str = "input_ids",
    pad_to_multiple_of: int = 1,
) -> DataLoader:
    """Creates a dataloader that dynamically batches based on token count."""
    if "length" in dataset.column_names:
        lengths = np.array(dataset["length"])
    else:
        lengths = np.array([len(x[input_ids_key]) for x in dataset])
    logger.debug(f"Finished getting lengths for {len(lengths):,} examples")
    sampler = TokenBatchSampler(
        lengths=lengths,
        max_tokens_per_batch=max_tokens_per_batch,
        drop_last=False,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )
