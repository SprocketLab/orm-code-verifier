import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from trl import DPOConfig as TRLDPOConfig
from accelerate import Accelerator
from trl import DPOTrainer
from trl.trainer.dpo_trainer import PreferenceCollator

from src import scoring
from src.training.config import TrainerConfig
from src.training.utils import DebuggingCallback
from src.utils import PaddingCollator
from src.utils import is_debugging_enabled

logger = logging.getLogger(__name__)


@dataclass
class DPOTrainingConfig(TrainerConfig):
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    use_weighting: bool = False
    disable_dropout: bool = True
    precompute_ref_log_probs: bool = True
    is_choice_trainer: bool = True
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    reference_batch_size: Optional[int] = None
    solution_in_label: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.is_choice_trainer

    def make_args(self, **kwargs):
        return TRLDPOConfig(
            **kwargs,
            beta=self.beta,
            label_smoothing=self.label_smoothing,
            loss_type=self.loss_type,
            use_weighting=self.use_weighting,
            disable_dropout=self.disable_dropout,
            precompute_ref_log_probs=self.precompute_ref_log_probs,
            max_length=self.max_input_length,
            max_prompt_length=self.max_prompt_length,
            max_completion_length=self.max_completion_length,
            precompute_ref_batch_size=self.reference_batch_size,
        )


class DPOScoringTrainer(DPOTrainer):
    def __init__(
        self,
        *args,
        cfg: DPOTrainingConfig,
        scoring_fn: scoring.ScoringFunction,
        preprocessed_eval_dataset: Dataset,
        use_reversed_fail_rewards: bool = False,
        eval_dataset: Dataset = None,
        eval_batch_tokens: int = 4096,
        **kwargs,
    ):

        super().__init__(
            *args,
            eval_dataset=Dataset.from_dict(
                {"prompt": ["N/A"], "chosen": ["bleh"], "rejected": ["bleh"]}
            ),
            **kwargs,
        )

        self.model = self.accelerator.prepare_model(kwargs["model"])
        self.eval_dataset = eval_dataset
        self.scoring_fn = scoring_fn
        logger.info(
            f"Actual batch size: "
            f"{self.args.per_device_train_batch_size*self.args.gradient_accumulation_steps} "
            f"(bs={self.args.per_device_train_batch_size}, grad_acc={self.args.gradient_accumulation_steps})"
        )
        self.preprocessed_eval_dataset = preprocessed_eval_dataset
        self.use_reversed_fail_rewards = use_reversed_fail_rewards
        self.step_results = {}
        self.eval_batch_tokens = eval_batch_tokens
        self._debugging = is_debugging_enabled()
        self._debug_output_dir = None
        if self._debugging:
            self._debug_callback = DebuggingCallback(
                tokenizer=self.processing_class,
                max_in_memory=10,
            )
            self.callback_handler.add_callback(self._debug_callback)
        else:
            self._debug_callback = None
        self.eval_dataset = eval_dataset
        self.eval_collator = PaddingCollator(
            pad_token=self.processing_class.pad_token_id,
            padding_side="left",
            special_pad_token_map={
                "input_ids": self.processing_class.pad_token_id,
                "attention_mask": 0,
            },
        )
        self.cfg = cfg

        chosen_lengths = np.array(
            [len(x) for x in self.train_dataset["chosen_input_ids"]]
        )
        rejected_lengths = np.array(
            [len(x) for x in self.train_dataset["rejected_input_ids"]]
        )
        prompt_input_ids = np.array(
            [len(x) for x in self.train_dataset["prompt_input_ids"]]
        )
        logger.info(f"Longest Prompt: {prompt_input_ids.max()}")
        logger.info(f"Longest chosen: {chosen_lengths.max()}")
        logger.info(f"Longest rejected: {rejected_lengths.max()}")
        logger.debug(
            f"Example prompt:\n{self.tokenizer.decode(self.train_dataset['prompt_input_ids'][0], skip_special_tokens=False)}"
        )
        logger.debug(
            f"Example chosen:\n{self.tokenizer.decode(self.train_dataset['chosen_input_ids'][0], skip_special_tokens=False)}"
        )
        logger.debug(
            f"Example rejected:\n{self.tokenizer.decode(self.train_dataset['rejected_input_ids'][0], skip_special_tokens=False)}"
        )

    def _get_current_epoch_file(self) -> Optional[Path]:
        if not self._debugging:
            return None
        return self._debug_output_dir / f"epoch_{self.state.epoch}.jsonl.gz"

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
        metrics = scoring.evaluate_dataset(
            accelerator=self.accelerator,
            dataset=eval_dataset,
            scoring_fn=self.scoring_fn,
            model=self.model,
            max_tokens_per_batch=self.eval_batch_tokens,
            preprocessed_ds=preproc_ds,
        )
        self.step_results[self.state.global_step] = deepcopy(metrics)
        self.step_results[self.state.global_step]["epoch"] = self.state.epoch

        out = {f"{metric_key_prefix}_{k}": float(v) for k, v in metrics.items()}

        self.log(out)
        self._memory_tracker.stop_and_update_metrics(out)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, out
        )
        return out

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = (
            eval_dataset if isinstance(eval_dataset, str) else "eval"
        )

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.eval_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "sampler": self._get_eval_sampler(eval_dataset),
            "drop_last": False,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[
                    dataloader_key
                ] = (  # pylint: disable=attribute-defined-outside-init
                    eval_dataloader
                )
            else:
                self._eval_dataloaders = (
                    {  # pylint: disable=attribute-defined-outside-init
                        dataloader_key: eval_dataloader
                    }
                )

        return self.accelerator.prepare(eval_dataloader)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """
        batch_size = (
            self.args.precompute_ref_batch_size
            or self.args.per_device_train_batch_size
        )

        logger.info(
            f"Precomputing reference log probs for training dataset with batch size {batch_size}"
        )
        if self._debugging:
            logger.info("Debugging enabled, sorting by length")
            self.train_dataset: Dataset = self.train_dataset.map(
                lambda x: {
                    "len": len(
                        x["prompt_input_ids"]
                        + x["chosen_input_ids"]
                        + x["rejected_input_ids"]
                    ),
                    **x,
                }
            )
            self.train_dataset = self.train_dataset.sort("len", reverse=True)
            self.train_dataset = self.train_dataset.remove_columns("len")

        return super().get_train_dataloader()


def make_dpo_trainer(
    cfg: DPOTrainingConfig,
    args: TRLDPOConfig,
    model: PreTrainedModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    preprocessed_eval_dataset: Dataset,
    scoring_fn: scoring.ScoringFunction,
    sort_by_length: bool = False,
) -> DPOScoringTrainer:
    _ = cfg

    if sort_by_length:
        train_dataset = train_dataset.map(
            lambda ex: {
                "length": len(ex["prompt"] + ex["chosen"] + ex["rejected"]),
                **ex,
            },
            num_proc=cfg.num_workers,
        )
        train_dataset = train_dataset.sort("length")
        train_dataset = train_dataset.remove_columns("length")

    return DPOScoringTrainer(
        model=model,
        args=args,
        data_collator=PreferenceCollator(pad_token_id=tokenizer.pad_token_id),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        preprocessed_eval_dataset=preprocessed_eval_dataset,
        scoring_fn=scoring_fn,
        eval_batch_tokens=cfg.eval_batch_tokens,
        cfg=cfg,
    )
