import dataclasses
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import DefaultFlowCallback
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from trl import RewardConfig as TRLRewardConfig
from trl import RewardTrainer
from trl.trainer.utils import RewardDataCollatorWithPadding
from unidecode import unidecode

from src import scoring
from src.training.config import TrainerConfig
from src.training.utils import DebuggingCallback
from src.training.utils import FlowCallback
from src.utils import PaddingCollator
from src.utils import is_debugging_enabled

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RewardTrainingConfig(TrainerConfig):

    use_reversed_fail_rewards: bool = False
    rm_only_pass: bool = False
    add_chosen_loss: bool = False
    add_rejected_loss: bool = False
    regular_loss_coef: float = 0.5

    def make_args(self, **kwargs):
        return TRLRewardConfig(**kwargs, remove_unused_columns=False)

    def __post_init__(self):
        super().__post_init__()
        if self.rm_only_pass and self.use_reversed_fail_rewards:
            raise ValueError(
                "cls_rm_only_pass and cls_uses_reversed_fail cannot both be true"
            )


class RewardScoringTrainer(RewardTrainer):
    def __init__(
        self,
        *args,
        cfg: RewardTrainingConfig,
        scoring_fn: scoring.ScoringFunction,
        preprocessed_eval_dataset: Dataset,
        eval_batch_tokens: int = 4096,
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
        self.use_reversed_fail_rewards = cfg.use_reversed_fail_rewards
        self.step_results = {}
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
        self.eval_collator = PaddingCollator(
            pad_token=self.processing_class.pad_token_id,
            padding_side="left",
            special_pad_token_map={
                "input_ids": self.processing_class.pad_token_id,
                "attention_mask": 0,
            },
        )
        self.add_chosen_loss = cfg.add_chosen_loss
        self.add_rejected_loss = cfg.add_rejected_loss
        self.regular_loss_coefficient = cfg.regular_loss_coef
        self.cls_positive_class_diff = cfg.rm_only_pass
        self.eval_batch_tokens = eval_batch_tokens

        self.callback_handler.pop_callback(DefaultFlowCallback)
        self.callback_handler.add_callback(FlowCallback)

    def _get_current_epoch_file(self) -> Optional[Path]:
        if not self._debugging:
            return None
        return self._debug_output_dir / f"epoch_{self.state.epoch}.jsonl.gz"

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
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]

        if self.use_reversed_fail_rewards:
            if rewards_chosen.size(1) != 2:
                raise ValueError(
                    "Reversed fail rewards can only be used with binary rewards"
                )

            diff = (rewards_chosen[:, 0] - rewards_rejected[:, 0]) + (
                rewards_rejected[:, 1] - rewards_chosen[:, 1]
            )
            diff = diff.unsqueeze(-1)

        else:
            if self.cls_positive_class_diff:
                assert rewards_chosen.size(1) == 2
                diff = rewards_chosen[:, 1] - rewards_rejected[:, 1]
            else:
                diff = rewards_chosen - rewards_rejected

        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(  # pylint: disable=not-callable
                diff - inputs["margin"]
            )
        else:
            loss = -nn.functional.logsigmoid(  # pylint: disable=not-callable
                diff
            )

        if self._debugging:
            self._debug_callback.add_substep_examples(
                loss=loss.tolist(),
                rewards_chosen=rewards_chosen.tolist(),
                rewards_rejected=rewards_rejected.tolist(),
                **{
                    k: v.tolist()
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                },
            )
        if self.add_chosen_loss or self.add_rejected_loss:
            rejected_loss = torch.zeros_like(loss)
            chosen_loss = torch.zeros_like(loss)
            if rewards_chosen.size(1) == 1:
                if self.add_chosen_loss:
                    chosen_loss = nn.functional.mse_loss(
                        rewards_chosen,
                        torch.ones_like(rewards_chosen),
                        reduction="none",
                    )
                if self.add_rejected_loss:
                    rejected_loss = nn.functional.mse_loss(
                        rewards_rejected,
                        -1 * torch.ones_like(rewards_rejected),
                        reduction="none",
                    )

            else:
                if self.add_chosen_loss:
                    chosen_loss = nn.functional.cross_entropy(
                        rewards_chosen.view(-1, 2),
                        torch.ones(
                            rewards_chosen.shape[0], dtype=torch.long
                        ).to(rewards_chosen.device),
                        reduction="none",
                    )
                if self.add_rejected_loss:
                    rejected_loss = nn.functional.cross_entropy(
                        rewards_rejected.view(-1, 2),
                        torch.zeros(
                            rewards_rejected.shape[0], dtype=torch.long
                        ).to(rewards_rejected.device),
                        reduction="none",
                    )
            loss += self.regular_loss_coefficient * (
                chosen_loss + rejected_loss
            )

        loss = loss.mean()

        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean(
                (rewards_chosen + rewards_rejected) ** 2
            )

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

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


def make_reward_trainer(
    cfg: RewardTrainingConfig,
    args: TRLRewardConfig,
    model: PreTrainedModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    preprocessed_eval_dataset: Dataset,
    scoring_fn: scoring.ScoringFunction,
    sort_by_length: bool = False,
    **kwargs,
):
    collator = RewardDataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest", pad_to_multiple_of=2
    )

    train_dataset = train_dataset.map(
        lambda ex: {
            "length_chosen": len(ex["input_ids_chosen"]),
            "length_rejected": len(ex["input_ids_rejected"]),
        },
        num_proc=cfg.num_workers,
    )
    if sort_by_length:
        train_dataset = train_dataset.sort("length_chosen", reverse=True)
    chosen_lengths = np.array(train_dataset["length_chosen"])
    reject_lengths = np.array(train_dataset["length_rejected"])
    logger.info(
        f"Total Tokens:\n\tChosen: {chosen_lengths.sum():,}\n\tRejected: {reject_lengths.sum():,}"
    )
    for n, ln_arr in zip(
        ["Chosen", "Rejected"], [chosen_lengths, reject_lengths]
    ):
        logger.info(f"Longest {n} example: {ln_arr.max():,} tokens")
        logger.info(
            f"Average {n} example length: {ln_arr.mean():.2f} "
            f"(+/- {ln_arr.std():0.2f}) tokens"
        )
    train_dataset = train_dataset.remove_columns(
        ["length_chosen", "length_rejected"]
    )
    return RewardScoringTrainer(
        model=model,
        cfg=cfg,
        args=args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        preprocessed_eval_dataset=preprocessed_eval_dataset,
        scoring_fn=scoring_fn,
        eval_batch_tokens=cfg.eval_batch_tokens,
        **kwargs,
    )
