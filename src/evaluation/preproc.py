import ast
import functools
import gzip
import logging
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import ujson
from datasets import Dataset
from datasets import load_dataset
from evalplus.data import get_human_eval_plus
from evalplus.data import get_mbpp_plus

from src import utils
from src.evaluation.configs import EvalSuite
from src.evaluation.configs import HumanEvalConfig
from src.evaluation.configs import MBPPConfig
from src.evaluation.configs import TaskConfig
from src.evaluation.filter_functions import run_filter_on_dataset
from src.preprocessing import Preprocessor
from src.preprocessing import PreprocessorConfig

ROOT_DIR = Path(__file__).parents[2]

logger = logging.getLogger(__name__)

CC_DATASET = {
    l["name"]: l for l in load_dataset("deepmind/code_contests", split="test")
}
HE_DATASET = get_human_eval_plus()
MBPP_DATASET = get_mbpp_plus()

GSM8K_DATASET = {
    f"gsm8k/{i}": l
    for i, l in enumerate(load_dataset("gsm8k", "main", split="test"))
}


SETUP_NAME_TO_CLEAN = {
    "t1.0_n128": "t10_n128",
    "t1.0_n256": "t10_n256",
    "t0.2_n128": "t02_n128",
    "t0.4_n128": "t04_n128",
    "t0.6_n128": "t06_n128",
    "t0.8_n128": "t08_n128",
}


def get_subset_name(
    generator_model: str, sampling_setup: str, task_dataset: str
) -> str:
    sampling_setup = SETUP_NAME_TO_CLEAN.get(sampling_setup, sampling_setup)
    return f"{generator_model}_{sampling_setup}_{task_dataset}"


def _preproc_humaneval(
    solution: str,
    example: Dict,
    preprocessor: Preprocessor,
    include_problem_str: bool = False,
) -> Tuple[str, str, str]:
    if include_problem_str:
        final_def = None
        for n in ast.parse(example["problem"]).body:
            if isinstance(n, ast.FunctionDef):
                final_def = n
        problem = ast.get_docstring(final_def).strip()

    else:
        problem = None

    problem = preprocessor.render_problem(
        problem=problem,
    )
    program = preprocessor.render_program(
        program=solution,
    )

    return problem, program


def _generic_preproc(
    solution: str,
    example: Dict,
    preprocessor: Preprocessor,
) -> Tuple[str, str]:
    problem = preprocessor.render_problem(
        problem=example["problem"],
    )
    program = preprocessor.render_program(
        program=solution,
    )

    return problem, program


def get_task_processor(
    task_cfg: TaskConfig,
) -> Callable[[str, Dict, Preprocessor], Tuple[str, str]]:
    if isinstance(task_cfg, HumanEvalConfig):
        logger.debug("Using humaneval preprocessor")
        return functools.partial(
            _preproc_humaneval,
            include_problem_str=task_cfg.include_problem_str,
        )
    logger.debug(
        f"Using generic preprocessor with {task_cfg.__class__.__name__}"
    )
    return _generic_preproc


def process_batch(
    batch: Dict[str, List],
    indices: List[int],
    preprocessor_cfg: PreprocessorConfig,
    task_cfg: TaskConfig,
    icl_examples: List[str],
    rstrip_completion: bool = False,
) -> Dict[str, List]:
    # Need to initialize here otherwise jinja breaks
    preprocessor = Preprocessor(preprocessor_cfg)
    task_preprocessor = get_task_processor(task_cfg)

    def process_problem(problem_dict: Dict):
        raw_predictions = problem_dict["solutions"]

        for _, sol_dict in enumerate(raw_predictions):
            solution = sol_dict["solution"]
            problem, program = task_preprocessor(
                solution=solution,
                example=problem_dict,
                preprocessor=preprocessor,
            )
            prompt = preprocessor.render_prompt(
                problem=problem,
                program=program,
                icl_examples=icl_examples,
            )
            dummy_query = preprocessor.render_prompt(
                problem=problem,
                program=preprocessor.render_program("__DUMMY__"),
                icl_examples=icl_examples,
            ).split("__DUMMY__")[0]
            completion = preprocessor.render_completion(outcome=True)
            if rstrip_completion:
                prompt = prompt.rstrip()
                completion = completion.rstrip()

            yield {
                "problem": problem,
                "problem_dummy": dummy_query,
                "query": prompt,
                "completion": completion,
                **sol_dict,
            }

    out = defaultdict(list)
    for i, idx in enumerate(indices):

        for sol in process_problem({k: v[i] for k, v in batch.items()}):
            for k, v in sol.items():
                out[k].append(v)
            out["_identifier"].append(batch["_identifier"][i])

    return out


def get_eval_ds_name(
    black_format: bool,
    remove_comments: bool,
) -> str:
    ds_name = "corm"
    if black_format:
        ds_name += "_black"
    else:
        ds_name += "_no_fmt"
    if remove_comments:
        ds_name += "_no_comments"
    else:
        ds_name += "_comments"
    return ds_name


def make_icl_examples(
    suite: EvalSuite, preproc_cfg: PreprocessorConfig
) -> Dict[str, List[str]]:
    logger.info("Creating ICL examples")

    out = {}
    preproc = Preprocessor(preproc_cfg)
    for name, examples in suite.icl_examples.items():
        logger.info(f"'{name}' has {len(examples)} ICL examples")
        out[name] = []
        for ex in examples:
            program = preproc.render_program(program=ex["solution"])
            problem = preproc.render_problem(
                problem=ex["problem"],
            )
            outcome = preproc.render_outcome(ex["passed"])
            out[name].append(
                preproc.render_icl_example(
                    problem=problem, program=program, outcome=outcome
                )
            )

    return out


def process_raw_dataset(
    raw_ds: Dataset,
    generator_model: str,
    sampling_setup: str,
    dataset_name: str,
    task_cfg: TaskConfig,
    preproc_cfg: PreprocessorConfig,
    num_proc: int,
    filter_function: Optional[str] = None,
    icl_examples: List[str] = None,
    rstrip_completion: bool = False,
) -> Tuple[Dataset, Dataset, Optional[Dataset], float]:

    if filter_function is not None:
        ds, filtered, filter_elapsed = run_filter_on_dataset(
            dataset=raw_ds,
            dataset_name=dataset_name,
            filter_function=filter_function,
            num_procs=num_proc,
            model=generator_model,
            sampling_setup=sampling_setup,
        )
    else:
        filtered = None
        filter_elapsed = 0.0
        ds = raw_ds

    ds = ds.map(
        functools.partial(
            process_batch,
            preprocessor_cfg=preproc_cfg,
            task_cfg=task_cfg,
            icl_examples=icl_examples,
            rstrip_completion=rstrip_completion,
        ),
        batched=True,
        num_proc=num_proc,
        remove_columns=[c for c in raw_ds.column_names if c != "_identifier"],
        with_indices=True,
        load_from_cache_file=False,
        desc="Applying Prompts",
    )
    logger.debug("Example query:")
    logger.debug(ds[0]["query"])

    return ds, filtered, filter_elapsed


def _add_id_and_problem_str(problem: Dataset, dataset_name: str) -> Dataset:

    if dataset_name == "humaneval":
        problem["problem"] = HE_DATASET[problem["task_id"]]["prompt"]
    elif dataset_name == "mbpp":
        problem["problem"] = MBPP_DATASET[problem["task_id"]]["prompt"]
    elif dataset_name == "code_contests":
        problem["problem"] = CC_DATASET[problem["task_id"]]["description"]
    elif dataset_name == "gsm8k":
        problem["problem"] = GSM8K_DATASET[problem["task_id"]]["question"]
    problem["_identifier"] = problem["task_id"]
    return problem


def load_raw_eval_dataset(
    generator_model: str,
    sampling_setup: str,
    task_dataset: str,
    preproc_cfg: PreprocessorConfig,
    debug_num_probs: Optional[int] = None,
):
    ds_name = get_eval_ds_name(
        black_format=preproc_cfg.black_format,
        remove_comments=preproc_cfg.remove_comments,
    )
    use_subset = f"{generator_model}_{sampling_setup}_{task_dataset}"
    ds_path = (
        ROOT_DIR
        / "data"
        / "eval_datasets"
        / ds_name
        / f"{generator_model}_{sampling_setup}_{task_dataset}.jsonl.gz"
    )
    if ds_path.exists():
        logger.info(f"Loading dataset from {ds_path}")
        with gzip.open(ds_path, "rt") as f:
            raw_ds = []
            for line in map(ujson.loads, f):
                raw_ds.append(_add_id_and_problem_str(line, task_dataset))

        raw_ds = Dataset.from_list(raw_ds)
    else:
        raw_ds = load_dataset(
            f"anon/eval-{ds_name}",
            split="test",
            name=get_subset_name(generator_model, sampling_setup, task_dataset),
        )
    raw_ds = raw_ds.map(
        _add_id_and_problem_str, fn_kwargs={"dataset_name": task_dataset}
    )

    assert all(sum(s["count"] for s in ss) == 128 for ss in raw_ds["solutions"])
    logger.info(
        f"{generator_model}.{sampling_setup} has {len(raw_ds):,} problems"
    )

    if debug_num_probs is not None:
        logger.warning(
            f"Limiting {use_subset} to {debug_num_probs} examples per problem"
        )
        raw_ds = raw_ds.select(range(debug_num_probs))
    return raw_ds
