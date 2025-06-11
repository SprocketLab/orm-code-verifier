import functools
import logging
import os
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from datasets import Dataset
from datasets import load_from_disk
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.preprocessing import Preprocessor
from src.preprocessing import PreprocessorConfig
from src.training.config import DataConfig
from src.training.config import TrainerConfig
from src.utils import DEFAULT_OUT_DIR
from src.utils import clean_solution
from src.utils import remove_comments

logger = logging.getLogger(__name__)


def make_dataset_dir_name(
    pass_level: str,
    require_pf: bool,
    include_syntax: bool,
    black_format: bool,
    remove_comments: bool,
    only_dataset: Optional[str] = None,
) -> str:
    parts = []
    parts.append("p" + pass_level[0])
    if include_syntax:
        parts.append("syntax")
    if black_format:
        parts.append("black")
    else:
        parts.append("no-format")
    if require_pf:
        parts.append("require-pf")
    if remove_comments:
        parts.append("no-comments")
    if only_dataset is not None:
        parts.append(only_dataset)
    return f"{'_'.join(parts)}"


def truncate_problem(
    problem: str, max_prob_length: int, tokenizer: PreTrainedTokenizer
) -> str:
    """Truncates a problem on the right side prior to any processing so we keep the contents somewhat intact."""
    tokens = tokenizer.encode(problem)
    if len(tokens) > max_prob_length:
        return tokenizer.decode(tokens[:max_prob_length])
    return problem


def filter_solutions(
    batch: Dict[str, List], max_prog_chars: int
) -> Dict[str, List]:
    def filter_prob_sols(
        programs: List[Dict],
        syntax: List[Dict],
    ):
        out = {
            "syntax_errors": [],
            "programs": [],
            "original_pred": [],
            "mutants": [],
        }
        pass_counts = {
            "all": 0,
            "public": 0,
            "private": 0,
            "generated": 0,
        }
        valid_idx = set()
        for i, p in enumerate(programs):
            if len(p["code"]) > max_prog_chars:
                continue
            out["programs"].append(p)
            valid_idx.add(i)
            pass_counts["all"] += p["passed"]
            pass_counts["public"] += p["passed_public"]
            pass_counts["private"] += p["passed_private"]
            pass_counts["generated"] += p["passed_generated"]

        for p in syntax:
            if len(p["code"]) > max_prog_chars:
                continue
            out["syntax_errors"].append(p)

        out["pass_counts"] = pass_counts
        return out

    new_batch = defaultdict(list)
    for i in range(len(batch["description"])):
        filtered = filter_prob_sols(
            programs=batch["old_programs"][i],
            syntax=batch["old_syntax_errors"][i],
        )
        new_batch["programs"].append(filtered["programs"])
        new_batch["syntax_errors"].append(filtered["syntax_errors"])
        new_batch["original_pred"].append(filtered["original_pred"])
        new_batch["pass_counts"].append(filtered["pass_counts"])
    return new_batch


def prepare_problem_batch(
    batch: Dict[str, List],
    cfg: DataConfig,
    preprocess_cfg: PreprocessorConfig,
    tokenizer: PreTrainedTokenizer,
    max_prob_length: int,
):

    preprocessor = Preprocessor(preprocess_cfg)

    def prepare_problem(
        problem,
    ):
        """Prepares a problem for model training by processing its description and associated programs.

        Args:
            problem (dict): The problem data containing its description and programs.
            cfg (DataConfig): Configuration for data processing, including flags for
                including syntax errors, mutations, and failed programs.
            preprocess_cfg (PreprocessorConfig): Configuration for preprocessing templates
                and rendering options.

        Returns:
            dict: A dictionary with the processed problem description and programs, including:
                - "problem": The rendered problem text.
                - "solutions": A list of processed program solutions.
                - "passed_idx": Indices of programs that passed the tests.
                - "failed_idx": Indices of programs that failed the tests.
                - "sol_ids": Unique identifiers for each solution.
        """

        pass_key = "passed" + (
            "" if cfg.pass_level == "all" else "_" + cfg.pass_level
        )
        problem_text = truncate_problem(
            problem["description"], max_prob_length, tokenizer
        )

        problem_text = preprocessor.render_problem(problem_text)
        new_programs = []
        to_use = ["predictions"]
        if cfg.include_syntax:
            to_use.append("invalid_syntax")

        # Keep the list of pas and fail indices for creating pair data.

        sol_ids = []
        outcomes = []
        valid_idx_to_use = set()
        for k in to_use:
            for j, prog in enumerate(problem[k]):
                solution = prog["code"]

                solution = preprocessor.render_program(solution)

                if k == "predictions":
                    passed = prog[pass_key]
                    if not cfg.include_failed and not passed:
                        continue
                else:
                    passed = False

                if passed and k == "predictions":
                    valid_idx_to_use.add(j)
                outcomes.append(passed)
                new_programs.append(solution)
                sol_ids.append(f"{problem['task_id']}/{k}/{j}")
        return {
            "problem": problem_text,
            "solutions": new_programs,
            "outcomes": outcomes,
            "sol_ids": sol_ids,
            "source": problem["source"],
        }

    out = defaultdict(list)
    for i in range(len(batch["description"])):
        problem = prepare_problem({k: v[i] for k, v in batch.items()})
        for k, v in problem.items():
            out[k].append(v)
    return out


def load_training_dataset(
    cfg: TrainerConfig,
    tokenizer: PreTrainedTokenizer,
    debug_num_problems: Optional[int] = None,
) -> Dataset:
    """Loads the training dataset from disk.

    First it loads the raw dataset from disk, then filters it according to the
    data config. Next it applys the preprocessor to the problems and solutions.
    This does NOT do any selection of examples.

    Args:
        cfg: The training config.
        tokenizer: The tokenizer to use for truncation and cleaning.
        debug_num_problems: The number of problems to use for debugging.

    Returns:
        The raw training dataset with columns:
            - problem: The problem text.
            - solutions: A list of the processed solutions.
            - outcomes: A list of the outcomes of the solutions.
            - sol_ids: The unique identifiers for each solution.
            - source: The source of the problem.
    """
    data_dir_name = make_dataset_dir_name(
        pass_level=cfg.data.pass_level,
        require_pf=cfg.data.require_pf,
        include_syntax=cfg.data.include_syntax,
        black_format=cfg.preprocessing.black_format,
        remove_comments=cfg.preprocessing.remove_comments,
        only_dataset=cfg.data.only_dataset,
    )

    # Load raw dataset from disk
    raw_dataset = load_from_disk(
        os.path.join(DEFAULT_OUT_DIR, "train_data", data_dir_name)
    )["train"]

    logger.info(f"Loaded {len(raw_dataset):,} training problems")

    # Apply debug limit if specified
    if debug_num_problems is not None:
        logger.warning(f"Debugging mode: Using {debug_num_problems:,} problems")
        rng = np.random.default_rng(cfg.seed)
        raw_dataset = raw_dataset.select(
            rng.choice(
                len(raw_dataset),
                min(len(raw_dataset), debug_num_problems),
                replace=False,
            )
        )

    # Process problems and solutions
    processed_dataset = raw_dataset.map(
        functools.partial(
            prepare_problem_batch,
            cfg=cfg.data,
            preprocess_cfg=cfg.preprocessing,
            tokenizer=tokenizer,
            max_prob_length=cfg.max_problem_tokens,
        ),
        desc="Preparing Train Dataset",
        num_proc=cfg.num_workers,
        remove_columns=[c for c in raw_dataset.column_names if c != "source"],
        batch_size=cfg.map_batch_size,
        batched=True,
    )

    logger.info(f"{len(processed_dataset):,} training problems after filtering")
    return processed_dataset


def _add_special(
    input_ids: List[int],
    attention_mask: List[int],
    tokenizer: PreTrainedTokenizer,
) -> Tuple[List[int], List[int]]:

    add_bos = (
        tokenizer.bos_token is not None
        and tokenizer.bos_token != tokenizer.eos_token
    )
    input_ids = [
        [tokenizer.bos_token_id] * add_bos + ids + [tokenizer.eos_token_id]
        for ids in input_ids
    ]
    attention_mask = [[1] * add_bos + mask + [1] for mask in attention_mask]
    return input_ids, attention_mask


def clm_tokenize_batch(
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    preprocess_cfg: PreprocessorConfig,
    solution_in_label: bool = False,
    max_length: int = None,
) -> Dict[str, List]:

    preprocessor = Preprocessor(preprocess_cfg)

    queries = []
    sequences = []
    for problem, solution, passed in zip(
        batch["problem"], batch["solution"], batch["passed"]
    ):
        prompt = preprocessor.render_prompt(problem=problem, program=solution)
        if solution_in_label:
            mask_seq = prompt[: prompt.index(solution)]
        else:
            mask_seq = prompt

        completion = preprocessor.render_completion(outcome=passed)
        queries.append(mask_seq)
        sequences.append(prompt + completion)
    out = tokenizer(
        sequences,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )

    out["input_ids"], out["attention_mask"] = _add_special(
        out["input_ids"], out["attention_mask"], tokenizer
    )
    query_toked = tokenizer(
        queries,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]
    add_bos = (
        tokenizer.bos_token is not None
        and tokenizer.bos_token != tokenizer.eos_token
    )
    out["passed"] = batch["passed"]
    out["labels"] = []
    for i, (q_toks, ids, attn) in enumerate(
        zip(query_toked, out["input_ids"], out["attention_mask"])
    ):
        to_trunc = len(q_toks) + add_bos
        label = [-100] * to_trunc + ids[to_trunc:]
        if len(ids) > max_length:
            label = label[-max_length:]
            out["input_ids"][i] = ids[-max_length:]
            out["attention_mask"][i] = attn[-max_length:]

        out["labels"].append(label)

    return out


def cls_tokenize_batch(
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    preprocess_cfg: PreprocessorConfig,
    max_length: int,
    positive_label: int = 1,
    negative_label: int = 0,
) -> Dict[str, List]:
    """Tokenizes a batch of sequences for classification.

    Args:
        batch (Dict[str, List]): A batch of data with keys "problem" and "solution".
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        preprocess_cfg (PreprocessorConfig): The configuration for preprocessing.
        max_length (int): The maximum length of the tokenized sequences.
        positive_label (float): The label for positive examples.
        negative_label (float): The label for negative examples.

    Returns:
        Dict[str, List]: A dictionary with the following keys:
            - "input_ids": The tokenized sequences.
            - "attention_mask": The attention masks for the sequences.
            - "labels": The labels for the sequences.
            - "passed": The passed status for the sequences.
    """
    preprocessor = Preprocessor(preprocess_cfg)

    sequences = []
    for problem, solution in zip(batch["problem"], batch["solution"]):
        prompt = preprocessor.render_prompt(problem=problem, program=solution)

        sequences.append(prompt.strip())

    out = tokenizer(
        sequences,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    out["input_ids"], out["attention_mask"] = _add_special(
        out["input_ids"], out["attention_mask"], tokenizer
    )
    out["input_ids"] = [i[-max_length:] for i in out["input_ids"]]
    out["attention_mask"] = [a[-max_length:] for a in out["attention_mask"]]
    out["labels"] = [
        positive_label if p else negative_label for p in batch["passed"]
    ]
    out["passed"] = batch["passed"]

    return out


def rm_tokenize_batch(
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    preprocess_cfg: PreprocessorConfig,
    max_length: int,
) -> Dict[str, List]:

    preprocessor = Preprocessor(preprocess_cfg)
    chosen_seqs = []
    rejected_seqs = []
    for problem, chosen, rejected in zip(
        batch["problem"], batch["chosen"], batch["rejected"]
    ):
        chosen_seq = preprocessor.render_prompt(problem=problem, program=chosen)
        rejected_seq = preprocessor.render_prompt(
            problem=problem, program=rejected
        )
        chosen_seqs.append(chosen_seq.strip())
        rejected_seqs.append(rejected_seq.strip())
    chosen_toks = tokenizer(
        chosen_seqs,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    chosen_toks["input_ids"], chosen_toks["attention_mask"] = _add_special(
        chosen_toks["input_ids"], chosen_toks["attention_mask"], tokenizer
    )
    rejected_toks = tokenizer(
        rejected_seqs,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    rejected_toks["input_ids"], rejected_toks["attention_mask"] = _add_special(
        rejected_toks["input_ids"], rejected_toks["attention_mask"], tokenizer
    )

    out = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }

    for c_toks, r_toks, c_attn, r_attn in zip(
        chosen_toks["input_ids"],
        rejected_toks["input_ids"],
        chosen_toks["attention_mask"],
        rejected_toks["attention_mask"],
    ):
        if len(c_toks) > 32000 or len(r_toks) > 32000:
            logger.warning(
                f"Skipping example because it's too long: {len(c_toks)} {len(r_toks)}"
            )
            continue
        out["input_ids_chosen"].append(c_toks[-max_length:])
        out["attention_mask_chosen"].append(c_attn[-max_length:])
        out["input_ids_rejected"].append(r_toks[-max_length:])
        out["attention_mask_rejected"].append(r_attn[-max_length:])

    return out


def make_clm_dataset(
    train_dataset: Dataset,
    rng: np.random.Generator,
    cfg: TrainerConfig,
    tokenizer: PreTrainedTokenizer,
    debug_num_problems: Optional[int] = None,
) -> Tuple[List, Dataset]:
    logger.info("Selecting CLM examples")
    selected, train_dataset = select_pointwise_examples(
        rng=rng,
        data=train_dataset,
        examples_per_problem=cfg.examples_per_problem,
    )
    logger.info(f"Selected {len(selected[-1]):,} training examples")
    logger.info(
        f"Max length: {min(cfg.max_input_length, tokenizer.model_max_length)}"
    )
    train_dataset = train_dataset.map(
        functools.partial(
            clm_tokenize_batch,
            tokenizer=tokenizer,
            preprocess_cfg=cfg.preprocessing,
            solution_in_label=cfg.solution_in_label,
            max_length=min(cfg.max_input_length, tokenizer.model_max_length),
        ),
        batched=True,
        batch_size=cfg.map_batch_size,
        num_proc=cfg.num_workers,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    if debug_num_problems is not None:
        train_dataset = train_dataset.select(
            rng.choice(
                len(train_dataset),
                min(len(train_dataset), debug_num_problems),
                replace=False,
            )
        )

        logger.warning(f"Only using {len(train_dataset):,} training examples")

    example = train_dataset[0]
    logger.info("Example query:")
    logger.info(
        tokenizer.decode(example["input_ids"], skip_special_tokens=False)
    )
    logger.info("Example Label")
    l = tokenizer.decode(
        [
            t
            for t, l in zip(example["input_ids"], example["labels"])
            if l != -100
        ],
        skip_special_tokens=False,
    )
    logger.info(l)
    return selected, train_dataset


def make_cls_dataset(
    train_dataset: Dataset,
    rng: np.random.Generator,
    cfg: TrainerConfig,
    tokenizer: PreTrainedTokenizer,
    debug_num_problems: Optional[int] = None,
) -> Tuple[List, Dataset]:
    logger.info("Selecting CLM examples")
    selected, train_dataset = select_pointwise_examples(
        rng=rng,
        data=train_dataset,
        examples_per_problem=cfg.examples_per_problem,
    )

    logger.info(f"Selected {len(selected[-1]):,} training examples")

    train_dataset = train_dataset.map(
        functools.partial(
            cls_tokenize_batch,
            tokenizer=tokenizer,
            preprocess_cfg=cfg.preprocessing,
            max_length=min(cfg.max_input_length, tokenizer.model_max_length),
            positive_label=cfg.positive_label,
            negative_label=cfg.negative_label,
        ),
        batched=True,
        batch_size=cfg.map_batch_size,
        num_proc=cfg.num_workers,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    if debug_num_problems is not None:
        train_dataset = train_dataset.select(
            rng.choice(
                len(train_dataset),
                min(len(train_dataset), debug_num_problems),
                replace=False,
            )
        )
        logger.warning(f"Only using {len(train_dataset):,} training examples")
    example = train_dataset[0]
    logger.info("Example query:")
    logger.info(
        tokenizer.decode(example["input_ids"], skip_special_tokens=False)
    )

    return selected, train_dataset


def make_rm_dataset(
    train_dataset: Dataset,
    rng: np.random.Generator,
    cfg: TrainerConfig,
    tokenizer: PreTrainedTokenizer,
    debug_num_problems: Optional[int] = None,
) -> Tuple[List, Dataset]:
    logger.info("Selecting RM examples")
    selected, train_dataset = select_pairwise_examples(
        rng=rng,
        data=train_dataset,
        examples_per_problem=cfg.examples_per_problem,
    )
    logger.info(f"Selected {len(selected[-1]):,} training examples")

    if debug_num_problems is not None:
        train_dataset = train_dataset.select(
            rng.choice(
                len(train_dataset),
                min(len(train_dataset), debug_num_problems),
                replace=False,
            )
        )
        logger.warning(f"Only using {len(train_dataset):,} training examples")

    train_dataset = train_dataset.map(
        functools.partial(
            rm_tokenize_batch,
            tokenizer=tokenizer,
            preprocess_cfg=cfg.preprocessing,
            max_length=min(cfg.max_input_length, tokenizer.model_max_length),
        ),
        batched=True,
        batch_size=cfg.map_batch_size,
        num_proc=cfg.num_workers,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
        load_from_cache_file=False,
    )

    example = train_dataset[0]
    logger.info("Example chosen:")
    logger.info(
        tokenizer.decode(example["input_ids_chosen"], skip_special_tokens=False)
    )
    logger.info("Example rejected:")
    logger.info(
        tokenizer.decode(
            example["input_ids_rejected"], skip_special_tokens=False
        )
    )

    return selected, train_dataset


def make_dpo_dataset(
    train_dataset: Dataset,
    rng: np.random.Generator,
    cfg: TrainerConfig,
    tokenizer: PreTrainedTokenizer,
    debug_num_problems: Optional[int] = None,
) -> Tuple[List, Dataset]:
    logger.info("Selecting RM examples")
    selected, train_dataset = select_pairwise_examples(
        rng=rng,
        data=train_dataset,
        examples_per_problem=cfg.examples_per_problem,
    )
    logger.info(f"Selected {len(selected[-1]):,} training examples")

    if debug_num_problems is not None:
        train_dataset = train_dataset.select(
            rng.choice(
                len(train_dataset),
                min(len(train_dataset), debug_num_problems),
                replace=False,
            )
        )
        logger.warning(f"Only using {len(train_dataset):,} training examples")

    def process_dpo_example(example):
        preprocessor = Preprocessor(cfg.preprocessing)
        dummy_query = preprocessor.render_prompt(
            problem=example["problem"],
            program=preprocessor.render_program("__DUMMY__"),
        ).split("__DUMMY__")[0]
        chosen = preprocessor.render_prompt(
            problem=example["problem"], program=example["chosen"]
        )
        chosen = chosen[len(dummy_query) :]
        rejected = preprocessor.render_prompt(
            problem=example["problem"], program=example["rejected"]
        )
        rejected = rejected[len(dummy_query) :]

        return {
            "prompt": dummy_query,
            "chosen": chosen,
            "rejected": rejected,
        }

    train_dataset = train_dataset.map(
        process_dpo_example,
        batched=False,
        num_proc=cfg.num_workers,
    )
    example = train_dataset[0]
    logger.info("Example query:")
    logger.info(example["prompt"])
    logger.info("Example chosen:")
    logger.info(example["chosen"])
    logger.info("Example rejected:")
    logger.info(example["rejected"])

    return selected, train_dataset


def select_pointwise_examples(
    rng: np.random.Generator,
    data: Dataset,
    examples_per_problem: int,
) -> Tuple[Tuple[Counter, List], Dataset]:
    """
    Selects pointwise examples from the given dataset.

    Args:
        rng: The random number generator to use for selecting examples.
        data: The dataset to select examples from.
        examples_per_problem: The number of examples to select for each problem.

    Returns:
        - The ids selected and if they passed or not.
        - The dataset with the selected examples It has columns:
            - problem: The prompt for the example.
            - solution: The solution for the example.
            - passed: Whether the example passed or not.
    """
    logger.info(
        f"Selecting pointwise examples with {examples_per_problem} examples per problem"
    )

    def _sample_pointwise_examples(
        batch: Dict[str, List],
        rng: np.random.Generator,
        examples_per_problem: int,
    ) -> List[Tuple[int, bool]]:

        out = {
            "idx": [],
            "problem": [],
            "solution": [],
            "passed": [],
            "source": [],
        }
        for i, problem in enumerate(batch["problem"]):
            solutions = batch["solutions"][i]
            outcomes = batch["outcomes"][i]
            if examples_per_problem == -1:
                to_sample = len(solutions)
            else:
                to_sample = min(examples_per_problem, len(solutions))
            selected = rng.choice(len(solutions), to_sample, replace=False)
            for idx in selected:
                out["idx"].append(batch["sol_ids"][i][idx])
                out["problem"].append(problem)
                out["solution"].append(solutions[idx])
                out["passed"].append(outcomes[idx])
                out["source"].append(batch["source"][i])
        return out

    selected = data.map(
        functools.partial(
            _sample_pointwise_examples,
            rng=rng,
            examples_per_problem=examples_per_problem,
        ),
        batched=True,
        batch_size=100,
        num_proc=1,
        remove_columns=data.column_names,
        desc="Selecting pointwise examples",
    )
    selected_info = [p for p in zip(selected["idx"], selected["passed"])]

    by_source = Counter(selected["source"])
    selected_info = (by_source, selected_info)
    selected = selected.remove_columns(["idx", "source"])

    return selected_info, selected


def select_pairwise_examples(
    rng: np.random.Generator,
    data: Dataset,
    examples_per_problem: int,
) -> Tuple[Tuple[Counter, List], Dataset]:
    """
    Selects pointwise examples from the given dataset.

    Args:
        rng: The random number generator to use for selecting examples.
        data: The dataset to select examples from.
        examples_per_problem: The number of examples to select for each problem.

    Returns:
        - The ids selected and if they passed or not.
        - The dataset with the selected examples It has columns:
            - problem: The prompt for the example.
            - chosen: The chosen solution.
            - rejected: The rejected solution.
    """
    logger.info(
        f"Selecting pairwise examples with {examples_per_problem} examples per problem"
    )

    def _sample_pairwise_examples(
        batch: Dict[str, List],
        rng: np.random.Generator,
        examples_per_problem: int,
    ) -> List[Tuple[int, bool]]:

        out = {
            "problem": [],
            "chosen": [],
            "rejected": [],
            "chosen_idx": [],
            "rejected_idx": [],
            "source": [],
        }
        for i, problem in enumerate(batch["problem"]):
            solutions = batch["solutions"][i]
            outcomes = batch["outcomes"][i]
            passing_idx = []
            failing_idx = []
            for j, o in enumerate(outcomes):
                if o:
                    passing_idx.append(j)
                else:
                    failing_idx.append(j)
            to_sample = min(len(passing_idx), len(failing_idx))
            if examples_per_problem != -1:
                to_sample = min(examples_per_problem, to_sample)
            selected_chosen = rng.choice(passing_idx, to_sample, replace=False)
            selected_failing = rng.choice(failing_idx, to_sample, replace=False)

            for cidx, ridx in zip(selected_chosen, selected_failing):
                out["chosen_idx"].append(batch["sol_ids"][i][cidx])
                out["rejected_idx"].append(batch["sol_ids"][i][ridx])
                out["problem"].append(problem)
                out["chosen"].append(solutions[cidx])
                out["rejected"].append(solutions[ridx])
                out["source"].append(batch["source"][i])
        return out

    selected = data.map(
        functools.partial(
            _sample_pairwise_examples,
            rng=rng,
            examples_per_problem=examples_per_problem,
        ),
        batched=True,
        batch_size=100,
        num_proc=1,
        remove_columns=data.column_names,
        desc="Selecting Pairwise examples",
    )
    selected_info = [
        p for p in zip(selected["chosen_idx"], selected["rejected_idx"])
    ]
    by_source = Counter(selected["source"])
    selected_info = (by_source, selected_info)
    selected = selected.remove_columns(["source", "chosen_idx", "rejected_idx"])
    return selected_info, selected


def _process_validation_problem(
    example: Dict,
    prob_text: str,
    preprocessor: Preprocessor,
    per_prob: Optional[int],
) -> Generator[Dict, None, None]:
    problem = preprocessor.render_problem(prob_text)
    dummy_query = preprocessor.render_prompt(
        problem=problem, program=preprocessor.render_program("__DUMMY__")
    ).split("__DUMMY__")[0]
    programs = example["predictions"]
    if per_prob is not None:
        programs = programs[:per_prob]

    for sid, sol in enumerate(programs):
        solution = preprocessor.render_program(sol["code"])
        query = preprocessor.render_prompt(problem=problem, program=solution)

        completion = preprocessor.render_completion(outcome=True)
        yield {
            "query": query,
            "solution": solution,
            "problem": problem,
            "problem_dummy": dummy_query,
            **{
                k: sol[k]
                for k in [
                    "passed",
                    "passed_public",
                    "passed_private",
                ]
            },
            "completion": completion,
            "sid": sid,
            "source": example["source"],
        }


def _process_validation_batch(
    batch: Dict[str, List],
    indices: List[int],
    preprocess_cfg: PreprocessorConfig,
    per_prob: Optional[int],
    tokenizer: PreTrainedTokenizer,
    max_prob_length: int,
) -> Dict:
    """Process a batch of validation problems from a HF dataset and returns the
    individual examples for each problem."""

    preprocessor = Preprocessor(preprocess_cfg)

    out = defaultdict(list)

    for i, idx in enumerate(indices):
        problem_text = truncate_problem(
            problem=batch["description"][i],
            tokenizer=tokenizer,
            max_prob_length=max_prob_length,
        )
        processed_problem = _process_validation_problem(
            example={k: v[i] for k, v in batch.items()},
            prob_text=problem_text,
            preprocessor=preprocessor,
            per_prob=per_prob,
        )
        for ex in processed_problem:
            out["task_id"].append(f"{batch['task_id'][i]}/{idx}")
            for k, v in ex.items():
                out[k].append(v)
    return out


def make_validation_dataset(
    cfg: TrainerConfig,
    per_prob: Optional[int],
    tokenizer: PreTrainedTokenizer,
):
    """Creates the validation dataset.

    Loads the validation dataset from disk and filters out any problems that
    do not have at least one passing program. Then it processes the validation
    dataset by rendering the prompts and extracting the necessary information
    for ranking.

    Args:
        cfg: The TrainerConfig for the model being trained.
        per_prob: The number of programs to select for each problem. If None,
            all programs are selected.
        tokenizer: The tokenizer to use for truncation.

    Returns:
        A Dataset with the processed validation data.
    """
    data_dir_name = make_dataset_dir_name(
        pass_level=cfg.data.pass_level,
        require_pf=cfg.data.require_pf,
        include_syntax=cfg.data.include_syntax,
        black_format=cfg.preprocessing.black_format,
        remove_comments=cfg.preprocessing.remove_comments,
        only_dataset=cfg.data.only_dataset,
    )

    # Load raw dataset from disk
    raw_dataset = load_from_disk(
        os.path.join(DEFAULT_OUT_DIR, "train_data", data_dir_name)
    )["validation"]
    dataset = raw_dataset.map(
        functools.partial(
            _process_validation_batch,
            preprocess_cfg=cfg.preprocessing,
            per_prob=per_prob,
            tokenizer=tokenizer,
            max_prob_length=cfg.max_problem_tokens,
        ),
        batched=True,
        with_indices=True,
        batch_size=cfg.map_batch_size,
        num_proc=cfg.num_workers,
        remove_columns=raw_dataset.column_names,
        desc="Validation Preprocessing",
        load_from_cache_file=False,
    )
    logger.info(f"{len(dataset):,} validation examples.")

    return dataset
