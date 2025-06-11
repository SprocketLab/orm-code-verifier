import enum
import gzip
import logging
import shutil
from typing import Dict, List, Optional

import ujson
from transformers import PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from src.utils import get_current_out_dir


class TrainerType(enum.Enum):
    HUGGING_FACE = "hf"
    REWARD_MODEL = "rm"
    DPO = "dpo"


logger = logging.getLogger(__name__)


class DebuggingCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_in_memory: int,
    ) -> None:
        logger.warning(
            "Debugging callback initialized, this will use a lot of memory"
        )
        self.tokenizer = tokenizer
        self.max_in_memory = max_in_memory
        self.output_dir = get_current_out_dir() / "debugging"
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_examples = []
        self.step_examples = []
        self.steps_in_memory = 0

    def add_substep_examples(self, **kwargs):
        self.step_examples.append(kwargs)

    def _write_examples(self, step_num: int, epoch_num: int):
        substep = 1
        while self.step_examples:
            ex = self.step_examples.pop()
            self.current_examples.append(
                {
                    "step": step_num,
                    "epoch": epoch_num,
                    "substep": substep,
                    **ex,
                }
            )
            substep += 1
        self.steps_in_memory += 1
        if self.steps_in_memory >= self.max_in_memory:
            out_file = self.output_dir / f"{step_num}.jsonl.gz"
            logger.debug(f"Writing debugging examples to {out_file}")
            with gzip.open(out_file, "wt") as f:
                while self.current_examples:
                    ex = self.current_examples.pop()
                    f.write(ujson.dumps(ex) + "\n")
            self.steps_in_memory = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._write_examples(state.global_step, state.epoch)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._write_examples(state.global_step, state.epoch)


def _make_label(
    input_ids: List[int],
    passed: bool,
    scoring_method: str,
    query_length: int,
    problem_length: int,
    label_masking_token: Optional[int] = None,
    mask_problem: bool = False,
):
    if scoring_method == "classification":
        return 1 if passed else 0
    if label_masking_token is None:
        return input_ids

    if mask_problem:
        len_use = problem_length
    else:
        len_use = query_length
    return [label_masking_token] * len_use + input_ids[len_use:]


def make_label(
    batch: Dict[str, List],
    scoring_method: str,
    label_masking_token: Optional[int] = None,
    query_len_key: str = "query_length",
    problem_len_key: str = "problem_length",
    input_id_key: str = "input_ids",
    mask_problem: bool = False,
) -> Dict[str, List[int | List[int]]]:
    out = {"labels": []}
    num_ex = len(batch[input_id_key])
    for i in range(num_ex):
        label = _make_label(
            input_ids=batch[input_id_key][i],
            passed=batch["passed"][i],
            scoring_method=scoring_method,
            query_length=batch[query_len_key][i],
            problem_length=batch[problem_len_key][i],
            label_masking_token=label_masking_token,
            mask_problem=mask_problem,
        )
        out["labels"].append(label)
    return out


class FlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if (
            args.logging_strategy == IntervalStrategy.STEPS
            and state.global_step % state.logging_steps == 0
        ):
            control.should_log = True

        # Evaluate
        if (
            args.eval_strategy == IntervalStrategy.STEPS
            and state.global_step % state.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # Save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and state.save_steps > 0
            and state.global_step % state.save_steps == 0
        ):
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True
            # Save the model at the end if we have a save strategy
            if args.save_strategy != IntervalStrategy.NO:
                control.should_save = True

        return control

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Log
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if (
            args.eval_strategy == IntervalStrategy.EPOCH
            and args.eval_delay <= state.epoch
        ):
            control.should_evaluate = True

        elif args.eval_strategy == IntervalStrategy.STEPS:
            if state.global_step >= state.max_steps:
                control.should_evaluate = True
            elif state.epoch == args.num_train_epochs:
                control.should_evaluate = True

        # Save
        if args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True

        return control
