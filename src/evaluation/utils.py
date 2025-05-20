import functools
import gzip
import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import ujson
from code_execution.eval_dataset import code_contests
from code_execution.eval_dataset import gsm8k
from datasets import Dataset
from datasets import load_dataset
from evalplus.data import get_human_eval_plus
from evalplus.data import get_mbpp_plus
from evalplus.evaluate import evaluate as evalplus
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from vllm import LLM
from vllm import SamplingParams

from src.evaluation.prompts import get_cc_prompt
from src.evaluation.prompts import get_evalplus_prompt
from src.evaluation.prompts import get_gsm_prompt
from src.evaluation.prompts import get_prompt
from src.evaluation.prompts import make_gsm_icl
from src.generation import serialize_raw_predictions
from src.modeling import MODEL_NAMES

logger = logging.getLogger(__name__)
BASE_DATA_PATH = Path("data", "ranking_datasets")

with Path(__file__).parents[2].joinpath(
    "data", "gsm8k_few_shot_prompts.json"
).open("r") as f:
    GSM8K_ICL = ujson.load(f)

DS_WITH_HIDDEN = {"humaneval", "mbpp"}


EVAL_TASKS = {"mbpp", "humaneval", "gsm8k", "code_contests"}

STOP_WORDS = {
    "gsm8k": {
        "\n```",
        "\nif",
        "\n#",
        "\n@",
        "\nprint",
        "<file_sep>",
        "\nclass",
    },
    "code_contests": set(["\n```"]),
    "humaneval": {
        "\nclass",
        "\nassert",
        '\n"""',
        "\nprint",
        "\nif __name__",
        "\n<|/",
        "\n```",
    },
    "mbpp": {
        "\nclass",
        "\nassert",
        '\n"""',
        "\nprint",
        "\nif __name__",
        "\n<|/",
        "\n```",
    },
}

CHAT_STOP_WORDS = {
    "qwen": {"<|im_end|>", "<|endoftext|>", "<|im_start|>"},
    "deepseek-r1": {
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "</answer>",
    },
}


def prep_code_contests(
    ex,
    idx,
    prompt_fn: Callable,
    is_chat: bool = False,
    add_py_md: bool = True,
):
    example = code_contests.process_problem(ex)
    example.pop("solutions")
    example.pop("incorrect_solutions")
    example.pop("untranslated_description")
    example.pop("input_file")
    example.pop("output_file")

    prompt = prompt_fn(
        get_cc_prompt(example, is_chat, add_py_md), is_chat=is_chat
    )
    return {"task_id": f"code_contests/{idx}", "query": prompt, **example}


def load_cc_dataset(
    prompt_fn: Callable,
    is_chat: bool = False,
    split: str = "test",
    add_py_md: bool = True,
) -> Dataset:
    dataset = load_dataset(
        "deepmind/code_contests", split=split, trust_remote_code=True
    ).map(
        functools.partial(
            prep_code_contests,
            prompt_fn=prompt_fn,
            is_chat=is_chat,
            add_py_md=add_py_md,
        ),
        with_indices=True,
        remove_columns=[
            "solutions",
            "incorrect_solutions",
            "untranslated_description",
            "input_file",
            "output_file",
        ],
        load_from_cache_file=False,
        desc="Preparing Code Contests",
    )
    return dataset


def load_evalplus_dataset(
    prompt_fn: Callable,
    is_chat: bool,
    ds_name: str,
    add_py_md: bool = True,
) -> Dataset:
    if ds_name == "mbpp":
        dataset = get_mbpp_plus()
    elif ds_name == "humaneval":
        dataset = get_human_eval_plus()
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    out = []
    for k, v in dataset.items():
        prompt = prompt_fn(
            get_evalplus_prompt(
                doc=v,
                is_chat=is_chat,
                add_py_md=add_py_md,
            ),
            is_chat=is_chat,
        )
        out.append(
            {
                "task_id": k,
                "query": prompt,
            }
        )

    return Dataset.from_list(out)


def load_gsm8k_dataset(
    prompt_fn: Callable,
    is_chat: bool = False,
    split: str = "test",
    add_py_md: bool = True,
) -> Dataset:
    logger.info("Loading GSM8K dataset")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    logger.debug("GSM8K dataset loaded, processing")
    icl = make_gsm_icl(GSM8K_ICL, question_is_doc=False, is_chat=is_chat)
    dataset = dataset.map(
        lambda e, i: {
            "task_id": f"gsm8k/{i}",
            "question": e["question"],
            "answer": e["answer"],
            "query": prompt_fn(
                get_gsm_prompt(e, icl, is_chat=is_chat, add_py_md=add_py_md),
                is_chat=is_chat,
            ),
        },
        with_indices=True,
        remove_columns=dataset.column_names,
        desc="Making GSM8K Query",
    )

    return dataset


def _load_eval_dataset(
    model_name: str,
    ds_name: str,
    is_chat: bool = False,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    add_py_md: bool = True,
) -> Dataset:

    if ds_name == "code_contests":
        dataset_fn = load_cc_dataset
    elif ds_name == "gsm8k":
        dataset_fn = load_gsm8k_dataset
    elif ds_name in {"mbpp", "humaneval"}:
        dataset_fn = functools.partial(load_evalplus_dataset, ds_name=ds_name)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    make_prompt_fn = functools.partial(
        get_prompt,
        is_chat=is_chat,
        model_name=model_name,
        tokenizer=tokenizer,
        add_generation_prefix=not add_py_md,
    )
    dataset = dataset_fn(
        prompt_fn=make_prompt_fn, is_chat=is_chat, add_py_md=add_py_md
    )

    logger.debug(f"Columns for {ds_name}: {dataset.column_names}")
    return dataset


COLS_KEEP_TASK = {
    "mbpp": [
        "task_id",
    ],
    "humaneval": [
        "task_id",
    ],
    "gsm8k": ["question", "answer", "task_id"],
    "code_contests": [
        "task_id",
        "name",
    ]
    + list(code_contests.REQUIRED_EXECUTION_KEYS),
}


def generate_eval_ds(
    task_name: str,
    num_samples: int,
    num_repeat: int,
    model_name: str,
    seed: int,
    max_num_seqs: int,
    swap_space: int,
    sampling_kwargs: Dict,
    overwrite: bool,
    max_model_len: int,
    tokenizer: PreTrainedTokenizer,
    is_chat: bool,
    raw_pred_dir: Optional[Path] = None,
    debug_num: Optional[int] = None,
    use_chat_stop_words: bool = False,
    add_py_md: bool = True,
):
    task_ds = _load_eval_dataset(
        model_name=MODEL_NAMES.get(model_name, model_name),
        ds_name=task_name,
        is_chat=is_chat,
        tokenizer=tokenizer,
        add_py_md=add_py_md,
    )
    logger.debug(f"{task_name}: {len(task_ds)} examples")
    if debug_num is not None and len(task_ds) > debug_num:
        indices = list(range(len(task_ds)))
        task_ds = task_ds.select(
            indices[: debug_num // 2] + indices[-debug_num // 2 :]
        )
    logger.debug(
        f"Stop words for {task_name}: {STOP_WORDS.get(task_name, set())}"
    )
    stop_words = STOP_WORDS.get(task_name, set())

    raw_pred_file = raw_pred_dir / f"{task_name}.raw.jsonl.gz"

    if raw_pred_file.exists() and not overwrite:
        logger.warning(f"Raw predictions already exist at {raw_pred_file}")
        return
    logger.info(
        f"Example query for {task_name}:\n\n{task_ds['query'][0].rstrip()}"
    )

    logger.info(
        f"Generating {num_samples} per for {len(task_ds):,} problems for task '{task_name}'"
    )
    logger.info(f"Saving to {raw_pred_file}")

    full_model_name = MODEL_NAMES.get(model_name, model_name)
    if use_chat_stop_words:
        if full_model_name.startswith("Qwen/Qwen2.5-Coder"):
            stop_words = CHAT_STOP_WORDS["qwen"]
        elif full_model_name.startswith("Qwen/QwQ"):
            stop_words = CHAT_STOP_WORDS["qwen"]
        elif full_model_name.startswith("deepseek-ai/DeepSeek-R1"):
            stop_words = CHAT_STOP_WORDS["deepseek-r1"]
        else:
            raise ValueError(f"Unknown model: {full_model_name}")
    sampling_parameters = SamplingParams(
        **sampling_kwargs,
        n=num_samples,
        stop=list(stop_words),
    )
    logger.info(f"Stop Words: {stop_words}")
    logger.info(f"Loading model '{full_model_name}'")
    llm = LLM(
        model=full_model_name,
        seed=seed,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        swap_space=swap_space,
    )
    predictions = [[] for _ in range(len(task_ds))]
    for _ in range(num_repeat):
        preds = llm.generate(task_ds["query"], sampling_parameters)
        for i, p in enumerate(preds):
            predictions[i].append(p)
    predictions = serialize_raw_predictions(
        task_name=task_name,
        raw_predictions=predictions,
        dataset=task_ds,
        prompt_key="query",
        expected_num_preds=num_samples,
        row_keys_add=COLS_KEEP_TASK.get(
            task_name,
            [
                c
                for c in task_ds.column_names
                if c
                not in {
                    "query",
                }
            ],
        ),
    )
    logger.debug(f"Writing raw predictions to {raw_pred_file}")

    with gzip.open(raw_pred_file, "wt") as f:
        for row in tqdm(predictions, desc="Writing raw predictions"):
            f.write(ujson.dumps(row) + "\n")

    logger.info(
        f"First prediction (ticks for clarity):\n```python\n{predictions[0]['predictions'][0]['prediction']}\n```"
    )
