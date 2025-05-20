import dataclasses
import functools
import logging
from copy import deepcopy
from typing import Dict, List

import numpy as np
from datasets import Dataset
from transformers import DefaultFlowCallback
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments
from unidecode import unidecode

from src import scoring
from src.training.config import TrainerConfig
from src.training.utils import FlowCallback
from src.training.utils import make_label
from src.utils import PaddingCollator

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class HFTrainerConfig(TrainerConfig):
    solution_in_label: bool = False

    def __post_init__(self):
        super().__post_init__()


class HFScoringTrainer(HFTrainer):
    def __init__(
        self,
        *args,
        scoring_fn: scoring.ScoringFunction,
        preprocessed_eval_dataset: Dataset,
        eval_batch_tokens: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scoring_fn = scoring_fn
        logger.info(
            f"Actual batch size: "
            f"{self.args.per_device_train_batch_size*self.args.gradient_accumulation_steps} "
            f"(bs={self.args.per_device_train_batch_size}, grad_acc={self.args.gradient_accumulation_steps})"
        )
        self.preprocessed_eval_dataset = preprocessed_eval_dataset
        self.step_results = {}
        self.eval_batch_tokens = eval_batch_tokens
        self.callback_handler.pop_callback(DefaultFlowCallback)
        self.callback_handler.add_callback(FlowCallback)

    def evaluate(
        self,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        ignore_keys: List[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        _ = ignore_keys
        if eval_dataset is None:
            preproc_ds = self.preprocessed_eval_dataset
            eval_dataset = self.eval_dataset
        else:
            preproc_ds = (
                self.preprocessed_eval_dataset
                if metric_key_prefix == "eval"
                else None
            )
        self.model.eval()
        final_metrics = scoring.evaluate_dataset(
            accelerator=self.accelerator,
            dataset=eval_dataset,
            scoring_fn=self.scoring_fn,
            model=self.model,
            max_tokens_per_batch=self.eval_batch_tokens,
            preprocessed_ds=preproc_ds,
        )
        final_metrics = {k: float(np.mean(v)) for k, v in final_metrics.items()}

        self.step_results[self.state.global_step] = deepcopy(final_metrics)
        self.step_results[self.state.global_step]["epoch"] = self.state.epoch

        out = {
            f"{metric_key_prefix}_{k}": float(v)
            for k, v in final_metrics.items()
        }

        self.log(out)
        self._memory_tracker.stop_and_update_metrics(out)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, out
        )
        return out


def make_hf_trainer(
    cfg: HFTrainerConfig,
    args: TrainingArguments,
    model: PreTrainedModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    preprocessed_eval_dataset: Dataset,
    scoring_fn: scoring.ScoringFunction,
    sort_by_length: bool = False,
) -> HFScoringTrainer:
    _ = cfg

    collator = PaddingCollator(
        pad_token=tokenizer.pad_token_id, padding_side="right"
    )
    logger.info(
        f"Example Query:\n"
        f"{unidecode(tokenizer.decode(train_dataset['input_ids'][0],skip_special_tokens=False))}"
    )

    train_dataset = train_dataset.map(
        lambda ex: {"length": len(ex["input_ids"])},
        num_proc=cfg.num_workers,
    )
    if sort_by_length:
        train_dataset = train_dataset.sort("length", reverse=True)
    total_tokens = np.array(train_dataset["length"])

    logger.info(f"Total Tokens: {total_tokens.sum():,}")
    logger.info(f"Longest example: {total_tokens.max():,} tokens")
    logger.info(
        f"Average example length: {total_tokens.mean():.2f} "
        f"(+/- {total_tokens.std():0.2f}) tokens"
    )

    if not cfg.group_by_length:
        train_dataset = train_dataset.remove_columns(["length"])

    if "passed" in train_dataset.column_names:
        logger.info(
            f"Train dataset has {sum(train_dataset['passed'])/len(train_dataset):0.2%} passed"
        )

    return HFScoringTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        preprocessed_eval_dataset=preprocessed_eval_dataset,
        scoring_fn=scoring_fn,
        eval_batch_tokens=cfg.eval_batch_tokens,
    )
