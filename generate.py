import functools
import gzip
import json
import logging
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import bigcode_eval.tasks
import click
import ray
import ujson
from bigcode_eval.tasks import TASK_REGISTRY
from code_execution.eval_dataset import apps
from code_execution.eval_dataset.code_contests import REQUIRED_EXECUTION_KEYS
from code_execution.eval_dataset.code_contests import process_problem
from datasets import concatenate_datasets
from datasets import load_dataset
from rich.logging import RichHandler
from transformers import AutoTokenizer
from vllm import SamplingParams

from src import generation
from src.evaluation import prompts
from src.evaluation.utils import EVAL_TASKS
from src.evaluation.utils import STOP_WORDS
from src.evaluation.utils import generate_eval_ds
from src.modeling import MODEL_NAMES
from src.utils import CONSOLE
from src.utils import DEFAULT_OUT_DIR
from src.utils import set_seed

logger = logging.getLogger(__name__)


@click.group()
@click.option("--visible_devices", type=str, default=None)
@click.option("--debug", is_flag=True)
@click.option("--verbose", is_flag=True)
@click.option("--max_examples", type=int, default=None)
@click.option("--overwrite", is_flag=True)
@click.pass_context
def cli(
    ctx,
    debug: bool,
    verbose: bool,
    visible_devices: str,
    max_examples: int,
    overwrite: bool,
):
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug
    ctx.obj["OVERWRITE"] = overwrite

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=CONSOLE,
                level=logging.INFO if not verbose else logging.DEBUG,
            ),
            logging.FileHandler(
                Path("logs", "generate.log"),
                mode="w",
            ),
        ],
    )
    if len(logging.getLogger().handlers) > 1:
        logging.getLogger().handlers[1].setFormatter(
            logging.Formatter(
                "[%(asctime)s - %(levelname)s - %(name)s:%(lineno)d] %(message)s",
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
    logger.info("Logging initialized")

    if visible_devices is not None:
        logger.info(f"Setting CUDA_VISIBLE_DEVICES to '{visible_devices}'")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        ctx.obj["CUDA_VISIBLE_DEVICES"] = visible_devices

    if max_examples is not None:
        logger.warning(f"Limiting the number of examples to '{max_examples}'")
        ctx.obj["MAX_EXAMPLES"] = max_examples


@cli.command(
    name="eval-ds",
)
@click.argument("model_name", type=str)
@click.option(
    "--num_samples",
    "-n",
    type=int,
    default=200,
    help="The number of samples to generate for each example in the task dataset",
)
@click.option("--temperature", "-t", type=float, default=1.0)
@click.option("--top_k", "-k", type=int, default=-1)
@click.option("--top_p", "-p", type=float, default=0.95)
@click.option("--max_new_tokens", "-toks", type=int, default=1024)
@click.option("--swap_space", "-swap", type=int, default=16)
@click.option("--max_num_seqs", "-seqs", type=int, default=256)
@click.option("--seed", type=int, default=42)
@click.option("--greedy", is_flag=True)
@click.option("--debug_num", type=int, default=None)
@click.option("--max_model_len", type=int, default=8192)
@click.option("--task_name", type=str, default=None)
@click.option("--raw_pred_dir", "-raw", type=Path, default=None)
@click.option("--dir_suffix", type=str, default=None)
@click.option("--is_chat", "-chat", is_flag=True)
@click.option("--num_repeat", type=int, default=1)
@click.option("--use_chat_stop_words", is_flag=True)
@click.option("--no_py_md", is_flag=True)
@click.pass_context
def eval_ds(
    ctx,
    model_name: str,
    num_samples: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
    swap_space: int,
    max_num_seqs: int,
    seed: int,
    greedy: bool = False,
    debug_num: int = None,
    max_model_len: int = 8192,
    task_name: str = None,
    raw_pred_dir: Optional[Path] = None,
    dir_suffix: Optional[str] = None,
    is_chat: bool = False,
    num_repeat: int = 1,
    use_chat_stop_words: bool = False,
    no_py_md: bool = False,
):

    if greedy and num_samples != 1:
        raise ValueError("Greedy sampling only supports num_samples=1")

    ctx.ensure_object(dict)
    logger.debug(f"Context object: {ctx.obj}")

    suffix = f"t{temperature}_n{num_samples}"

    if greedy:
        suffix = "greedy"
    if dir_suffix is not None:
        suffix += f"_{dir_suffix}"
    if raw_pred_dir is None:
        raw_pred_dir = Path(
            DEFAULT_OUT_DIR,
            "eval_ds_raw_preds",
            model_name.replace("/", "_"),
            suffix,
        )
        raw_pred_dir.mkdir(parents=True, exist_ok=True)
    sampling_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_new_tokens,
        "logprobs": 1,
    }
    if greedy:
        sampling_kwargs["temperature"] = 0.0
        sampling_kwargs["top_p"] = 1
        sampling_kwargs["top_k"] = 1
        num_samples = 1

    logger.info("Sampling parameters:")
    for k, v in sampling_kwargs.items():
        logger.info(f"{k:>24}={v}")
    gen_fn = functools.partial(
        generate_eval_ds,
        model_name=model_name,
        seed=seed,
        max_num_seqs=max_num_seqs,
        swap_space=swap_space,
        sampling_kwargs=sampling_kwargs,
        overwrite=ctx.obj["OVERWRITE"],
        max_model_len=max_model_len,
        raw_pred_dir=raw_pred_dir,
        tokenizer=AutoTokenizer.from_pretrained(
            MODEL_NAMES.get(model_name, model_name)
        ),
        is_chat=is_chat,
        debug_num=debug_num,
        use_chat_stop_words=use_chat_stop_words,
        add_py_md=not no_py_md,
    )
    tasks_to_gen = [task_name] if task_name else list(sorted(EVAL_TASKS))

    for task in tasks_to_gen:
        gen_fn(
            task_name=task,
            num_samples=num_samples,
            num_repeat=num_repeat,
        )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
