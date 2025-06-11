"""This module contains the logic for scoring a dataset of solutions.

The design approach is that there is a base ScoringFunction class that contains the logic for scoring a dataset.
There are then concrete classes that implement the ScoringFunction for a specific scoring method.

The scoring function is callable with a dataset, a model, and a max_tokens_per_batch.

The scoring function will return a tuple of (score_timings, scored_solutions).

The scoring function will preprocess the dataset, score the dataset, and postprocess the scores.

Key Components:
- ScoringFunction: Abstract base class defining the scoring interface
- BinaryLogitScoring: Scores using binary logits over full sequences
- SingleTokenBinaryLogitScoring: Scores using binary logits on single tokens
- ClassificationScoring: Treats scoring as a classification task
- RewardModelScoring: Uses a reward model approach
- LogProbScoring: Scores based on log probabilities

The module supports different scoring methods:
- binary_logit: Binary logit scoring over sequences
- binary_logprob: Binary log probability scoring over sequences
- st_binary_logit: Single token binary logit scoring
- st_binary_logprob: Single token binary log probability scoring
- classification: Classification-based scoring
- reward_model: Reward model scoring
- logprob: Log probability scoring

Usage:
    scoring_fn = load_scoring_method(
        scoring_cfg=config,
        tokenizer=tokenizer,
        pass_choice_str="pass",
        fail_choice_str="fail",
        eval_completion="The answer is:"
    )

    timings, scores = scoring_fn(
        dataset=dataset,
        model=model,
        max_tokens_per_batch=1024
    )

The module also provides utilities for:
- Grouping and aggregating scores by task
- Computing evaluation metrics like ranking score and best-of-k
- Tracking timing information during scoring
"""

import logging
import os
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.amp
import torch.nn.functional as F
import unidecode
from accelerate import Accelerator
from datasets import Dataset
from omegaconf import MISSING
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from src.metrics import ScoredSolution
from src.metrics import best_of_k
from src.metrics import ranking_score
from src.utils import PaddingCollator
from src.utils import get_dynamic_token_dataloader
from src.utils import is_debugging_enabled
from src.utils import seconds_to_human

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration class for scoring methods.

    Attributes:
        scoring_method (str): The scoring method to use. Must be one of the supported methods.
        disable_single_token_optimization (bool): Whether to disable single token optimization.
        log_softmax_logits (bool): Whether to apply log softmax to logits.
        use_problem_for_mask (bool): Whether to use problem text for masking.
        use_dummy_problem_for_mask (bool): Whether to use dummy problem text for masking.
        include_completion (bool): Whether to include completion text in scoring.
    """

    scoring_method: str = MISSING
    disable_single_token_optimization: bool = False
    log_softmax_logits: bool = False
    use_problem_for_mask: bool = False
    use_dummy_problem_for_mask: bool = False
    include_completion: bool = True

    def __post_init__(self):
        if self.scoring_method not in SCORING_METHODS:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")
        if self.use_problem_for_mask and self.use_dummy_problem_for_mask:
            raise ValueError(
                "Cannot use both use_problem_for_mask and use_dummy_problem_for_mask"
            )


@dataclass
class ScoringTimings:
    """Class for tracking timing information during scoring.

    Attributes:
        inference (float): Time spent on model inference
        get_ds_scores (float): Time spent getting dataset scores
        scoring_function (float): Time spent in scoring function
        evaluation (float): Time spent on evaluation
        overall (float): Overall time spent
        filtering (float): Time spent on filtering
    """

    inference: float
    get_ds_scores: float
    scoring_function: float = float("inf")
    evaluation: float = float("inf")
    overall: float = float("inf")
    filtering: float = float("inf")


def _score_example(
    batch,
    device: torch.device,
    model: PreTrainedModel,
    scoring_fn: Callable,
):
    """Score a single batch of examples.

    Args:
        batch: The batch of examples to score
        device: The device to run inference on
        model: The model to use for scoring
        scoring_fn: The scoring function to apply

    Returns:
        tuple: (batch_scores, batch_losses, elapsed_time)
    """

    start_time = datetime.now(timezone.utc)

    logits = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    ).logits
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    b_score, b_losses = scoring_fn(logits.detach(), batch)

    if b_score.device != torch.device("cpu"):
        raise ValueError("Scores should be on CPU")
    if b_losses is not None and b_losses.device != torch.device("cpu"):
        raise ValueError("Losses should be on CPU")
    if is_debugging_enabled():
        logger.debug(
            f"Scored {batch['input_ids'].shape} batch in {elapsed:0.2f}"
        )
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            percent = (reserved / total) * 100

            # Calculate model size in GB
            model_size = (
                sum(p.numel() * p.element_size() for p in model.parameters())
                / 1024**3
            )

            # Calculate input tensor size in GB
            input_size = (
                batch["input_ids"].numel()
                * batch["input_ids"].element_size()
                / 1024**3
            )

            mem_str = "GPU Memory: " + "\n\t".join(
                [
                    f"{allocated:.1f}GB Allocated",
                    f"{reserved:.1f}GB Reserved",
                    f"{percent:.1f}% Utilized",
                    f"{model_size:.1f}GB Model Size",
                    f"{input_size:.3f}GB Input Size",
                ]
            )
            logger.debug(mem_str)

    del logits
    return b_score, b_losses, elapsed


def _get_ds_scores(
    dataset: Dataset,
    model: PreTrainedModel,
    device: torch.device,
    tokenizer: PreTrainedTokenizer,
    scoring_function: "ScoringFunction",
    max_tokens_per_batch: int,
    pad_side: str = "right",
    special_pad_token_map: Dict[str, int] = None,
) -> Tuple[ScoringTimings, Dict[str, torch.Tensor]]:
    """Get scores for an entire dataset.

    Args:
        dataset: The dataset to score
        model: The model to use for scoring
        device: The device to run on
        tokenizer: The tokenizer to use
        scoring_function: The scoring function to apply
        max_tokens_per_batch: Maximum tokens per batch
        pad_side: Side to pad sequences on
        special_pad_token_map: Map of special padding tokens

    Returns:
        tuple: (scoring_timings, results_dict)
    """
    logger.info(
        f"Scoring {len(dataset):,} examples with {max_tokens_per_batch:,} tokens per batch"
    )
    get_ds_start = datetime.now(timezone.utc)
    model.eval()
    logger.debug(f"Creating dataloader with {pad_side=}")
    dataloader = get_dynamic_token_dataloader(
        dataset=dataset,
        max_tokens_per_batch=max_tokens_per_batch,
        collate_fn=PaddingCollator(
            tokenizer.pad_token_id,
            padding_side=pad_side,
            special_pad_token_map=special_pad_token_map,
        ),
    )

    timings = []
    results = defaultdict(list)

    model.eval()
    with torch.no_grad():
        inference_start_time = datetime.now(timezone.utc)
        finished = 0
        for step, batch in enumerate(dataloader, start=1):
            # logger.debug(
            #     f"Scoring batch {step} with {batch['input_ids'].shape} shape"
            # )
            b_score, b_losses, elapsed = _score_example(
                batch,
                model=model,
                device=device,
                scoring_fn=scoring_function,
            )
            timings.append(elapsed)
            results["scores"].append(b_score)
            results["indices"].append(batch["idx"])
            finished += b_score.size(0)
            if "targets" in batch:
                results["targets"].append(batch["targets"])
            if b_losses is not None:
                results["losses"].append(b_losses)
            if step % 100 == 0:
                elapsed = datetime.now(timezone.utc) - inference_start_time
                logger.log(
                    logging.INFO,
                    f"Finished {finished}/{len(dataset)} in {seconds_to_human(elapsed.total_seconds())}",
                )

                torch.cuda.empty_cache()

        inference_elapsed = (
            datetime.now(timezone.utc) - inference_start_time
        ).total_seconds()
        logger.info(
            f"Finished scoring {len(dataset):,} examples in {seconds_to_human(inference_elapsed)}"
        )
    # We only have 1 timing per batch, but our batches are N examples per. So
    # we need to repeat the elapsed time N times
    logger.debug("Fixing timings")
    timings = torch.tensor(
        [
            t
            for i, t in enumerate(timings)
            for _ in range(results["scores"][i].size(0))
        ]
    )

    results = {k: torch.cat(v) for k, v in results.items()}
    results["timings"] = timings

    get_ds_elapsed = (datetime.now(timezone.utc) - get_ds_start).total_seconds()

    score_timings = ScoringTimings(
        inference=inference_elapsed,
        get_ds_scores=get_ds_elapsed,
    )

    return score_timings, results


class ScoringFunction:
    """Base class for scoring functions.

    This class defines the interface that all scoring functions must implement.
    Subclasses should override the preprocess_batch, score, and postprocess_scores methods.

    Args:
        cfg: The scoring configuration
        tokenizer: The tokenizer to use
        pass_choice_str: String indicating a passing solution
        fail_choice_str: String indicating a failing solution
        eval_completion: String to append for evaluation
        max_length: Maximum sequence length
        special_pad_token_map: Map of special padding tokens
        calculate_loss: Whether to calculate loss
        num_workers: Number of preprocessing workers
        preproc_batch_size: Preprocessing batch size
        postprocess_fn: Optional postprocessing function
    """

    def __init__(
        self,
        cfg: ScoringConfig,
        tokenizer: PreTrainedTokenizer,
        pass_choice_str: str,
        fail_choice_str: str,
        eval_completion: str,
        max_length: int = None,
        special_pad_token_map: Dict[str, int] = None,
        calculate_loss: bool = False,
        num_workers: int = 1,
        preproc_batch_size: int = 5000,
        postprocess_fn: Optional[Callable[[Dict, Dict], Dict]] = None,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_bos = (
            tokenizer.bos_token != tokenizer.pad_token
            and self.tokenizer.bos_token is not None
        )
        self.special_pad_token_map = special_pad_token_map
        self.calculate_loss = calculate_loss
        self.num_workers = num_workers
        self.postprocess_fn = postprocess_fn
        self.pad_side = "right"
        self.preprocess_batch_size = preproc_batch_size
        self.pass_choice_str = pass_choice_str
        self.fail_choice_str = fail_choice_str
        self.eval_completion = eval_completion

    def _preprocess_batch(
        self, batch: Dict[str, List], indices: List[int]
    ) -> Dict[str, List]:
        out = self.preprocess_batch(batch, indices)
        if self.max_length is not None and any(
            i > self.max_length for i in map(len, out["input_ids"])
        ):
            out["input_ids"] = [i[-self.max_length :] for i in out["input_ids"]]
            out["attention_mask"] = [
                a[-self.max_length :] for a in out["attention_mask"]
            ]
        out["length"] = [len(i) for i in out["input_ids"]]
        return out

    def __call__(
        self,
        dataset: Dataset,
        model: PreTrainedModel,
        max_tokens_per_batch: int,
        preprocessed_ds: Optional[Dataset] = None,
        accelerator: Accelerator = None,
        sort_by_length: bool = False,
    ) -> Tuple[ScoringTimings, List[Dict | Tuple[Any, List[Dict]]]]:
        logger.debug("Scoring %d examples", len(dataset))
        start = datetime.now(timezone.utc)
        if preprocessed_ds is None:
            preprocessed_ds = self.preprocess_dataset(dataset, sort_by_length)

        timings, results = _get_ds_scores(
            dataset=preprocessed_ds,
            model=model,
            device=model.device,
            tokenizer=self.tokenizer,
            scoring_function=self.score,
            max_tokens_per_batch=max_tokens_per_batch,
            special_pad_token_map=self.special_pad_token_map,
            pad_side=self.pad_side,
        )
        logger.info("Postprocessing scores")

        out = []
        for res in tqdm(
            self.postprocess_scores(
                **results,
                dataset=dataset,
            ),
            total=len(dataset),
            desc="Postprocessing",
            mininterval=1.0,
        ):
            row = dataset[res["idx"]]
            if self.postprocess_fn is not None:
                to_save = self.postprocess_fn(row, res)
            else:
                to_save = {
                    **res,
                    **{k: v for k, v in row.items() if k not in res},
                }
            out.append(to_save)
        timings.scoring_function = (
            datetime.now(timezone.utc) - start
        ).total_seconds()
        return timings, out

    def preprocess_batch(
        self, batch: Dict[str, List], indices: List[int]
    ) -> Dict[str, List]:
        raise NotImplementedError()

    def preprocess_dataset(
        self, dataset: Dataset, sort_by_length: bool = False
    ) -> Dataset:
        out = dataset.map(
            self._preprocess_batch,
            batched=True,
            remove_columns=[
                k
                for k in dataset.column_names
                if k
                not in {
                    "passed",
                    "task",
                }
            ],
            with_indices=True,
            num_proc=self.num_workers,
            batch_size=self.preprocess_batch_size,
            load_from_cache_file=False,
            desc="Tokenizing For Scoring",
        )

        decoded = unidecode.unidecode(
            self.tokenizer.decode(out["input_ids"][0])
        )
        logger.debug(
            f"Example Decoded Query:\n{decoded}",
        )
        logger.info(
            f"Longest input: {max(map(len, out['input_ids'])):,} tokens"
        )
        if "task" in out.column_names:
            out = out.remove_columns(["task"])

        if sort_by_length:
            logger.info("Sorting by length")
            out = (
                out.map(
                    lambda x: {"length": [len(y) for y in x["input_ids"]]},
                    batched=True,
                    batch_size=self.preprocess_batch_size,
                    num_proc=self.num_workers,
                    desc="Adding Lengths",
                )
                .sort("length", reverse=True)
                .remove_columns("length")
            )
        return out

    def score(
        self, logits: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError()

    def postprocess_scores(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        timings: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        losses: Optional[torch.Tensor] = None,
        dataset: Optional[Dataset] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        raise NotImplementedError()


class BinaryLogitScoring(ScoringFunction):
    """Scoring function that uses binary logits over full sequences.

    This class implements scoring by computing logits for binary classification
    (pass/fail) over complete sequences. It can optionally use full log probabilities.

    Args:
        cfg: The scoring configuration
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        use_full_logprob: Whether to use full log probabilities
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(
        self,
        cfg: ScoringConfig,
        tokenizer: PreTrainedTokenizer,
        max_length: int = None,
        use_full_logprob: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(
            cfg=cfg, tokenizer=tokenizer, max_length=max_length, **kwargs
        )
        self.num_choices = 2
        self.use_full_logprob = use_full_logprob
        self.loss_fct = torch.nn.CrossEntropyLoss(
            reduction="none", ignore_index=-100
        )

        if self.max_length is not None:
            longest_choice = max(
                len(self.tokenizer.encode(self.pass_choice_str)),
                len(self.tokenizer.encode(self.fail_choice_str)),
            )
            logger.debug(
                f"Longest choice: {longest_choice}, removing from max length of {self.max_length}"
            )
            self.max_length -= longest_choice

    def preprocess_batch(
        self, batch: List[Dict[str, List]], indices: List[int] = None
    ) -> List[Dict[str, List]]:
        """Preprocess a batch of examples.

        Args:
            batch: Batch of examples to preprocess
            indices: Optional list of indices for the batch

        Returns:
            dict: Preprocessed batch with tokenized inputs and masks
        """
        out = {
            "input_ids": [],
            "attention_mask": [],
            "choice_mask": [],
            "targets": [],  # used to keep track of the choice for the query.
            "idx": [],  # used to keep track of the original index and group.
            "passed": [],
            "task": [],
        }

        input_lengths = list(
            map(
                len,
                self.tokenizer(
                    batch["query"],
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                )["input_ids"],
            )
        )

        queries_tokenized = self.tokenizer(
            [
                query + self.eval_completion + c
                for query in batch["query"]
                for c in [self.fail_choice_str, self.pass_choice_str]
            ],
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        # Batch them by 2 to get the choices for each query
        queries_tokenized = {
            k: [
                queries_tokenized[k][i : i + 2]
                for i in range(0, len(queries_tokenized[k]), 2)
            ]
            for k in queries_tokenized
        }
        longest_query = 0
        for i, input_len in enumerate(input_lengths):

            input_ids = queries_tokenized["input_ids"][i]
            attention_mask = queries_tokenized["attention_mask"][i]

            # If we need to add a BOS token, increase the input length by 1
            if self.add_bos:
                input_len += 1
            for ids, attn in zip(input_ids, attention_mask):

                if self.add_bos:
                    ids = [self.tokenizer.bos_token_id] + ids
                    attn = [1] + attention_mask

                if self.use_full_logprob:
                    choice_mask = [1] * len(ids)
                else:
                    choice_mask = [0] * input_len + [1] * (len(ids) - input_len)

                out["input_ids"].append(ids)
                out["attention_mask"].append(attn)
                out["choice_mask"].append(choice_mask)
                longest_query = max(longest_query, len(ids))
            out["targets"].extend([0, 1])
            out["idx"].extend([indices[i]] * 2)
            out["passed"].extend([batch["passed"][i]] * 2)
            try:
                out["task"].extend([batch["task"][i]] * 2)
            except KeyError:
                out["task"].extend(["N/A"] * 2)
        return out

    def _preprocess_batch(
        self, batch: Dict[str, List], indices: List[int]
    ) -> Dict[str, List]:
        out = self.preprocess_batch(batch, indices)
        if self.max_length is not None and any(
            i > self.max_length for i in map(len, out["input_ids"])
        ):
            out["input_ids"] = [i[-self.max_length :] for i in out["input_ids"]]
            out["attention_mask"] = [
                a[-self.max_length :] for a in out["attention_mask"]
            ]
            out["choice_mask"] = [
                a[-self.max_length :] for a in out["choice_mask"]
            ]
        return out

    def score(
        self,
        logits: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Score a batch of examples.

        Args:
            logits: Model output logits
            batch: Batch of examples

        Returns:
            tuple: (scores, optional_losses)
        """
        input_ids = batch["input_ids"].clone().to(logits.device)

        # Calculate loss prior to log softmax if needed so we dont need other
        # logic.
        loss = None
        if self.calculate_loss:
            label = batch["input_ids"] * batch["choice_mask"]
            label = label[:, 1:]
            loss_mask = label != 0
            label[~loss_mask] = -100
            label = label.to(logits.device).contiguous()
            loss = self.loss_fct(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                label.view(-1),
            ).cpu()

            # Need to reshape to original input ids shape so we can get loss
            # per sequence, not per token. Cross entropy loss returns
            # (1, Num Tokens total).
            loss = loss.view(label.size(0), -1)
            loss = loss.sum(-1) / loss_mask.sum(-1)
            del label

        if self.cfg.log_softmax_logits:
            logits = F.log_softmax(logits, dim=-1)

        logits = logits[:, :-1, :]

        scores = torch.gather(logits, 2, input_ids[:, 1:].unsqueeze(2)).squeeze(
            2
        )

        scores = scores * batch["choice_mask"][:, 1:].to(logits.device)
        num_tokens = batch["choice_mask"].sum(dim=-1)
        scores = scores.cpu().sum(-1) / num_tokens

        return scores, loss

    def postprocess_scores(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        timings: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        losses: Optional[torch.Tensor] = None,
        dataset: Optional[Dataset] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        logger.info("Postprocessing BinaryLogit scores")
        scores = scores.view(-1, 2)
        preds = scores.argmax(-1).tolist()
        scores = scores.tolist()
        indices = indices.view(-1, 2).tolist()
        targets = targets.view(-1, 2).tolist()
        timings = timings.view(-1, 2).tolist()
        if losses is not None:
            losses = losses.view(-1, 2).tolist()
        else:
            losses = [None] * len(scores)
        for i, idx in enumerate(indices):

            assert idx[0] == idx[1]
            target = targets[i]

            assert target[0] == 0
            assert target[1] == 1

            score_dict = {
                "idx": idx[0],
                "score": scores[i][1],
                "failed_score": scores[i][0],
                "prediction": preds[i],
                "score_time": sum(timings[i]),  # Time per program is 2x.
            }
            if losses[i] is not None:
                label = 1 if dataset[idx[0]]["passed"] else 0

                score_dict["point_loss"] = losses[i][label]
                score_dict["pair_loss"] = (
                    losses[i][label] - losses[i][1 - label]
                )

            yield score_dict


class SingleTokenBinaryLogitScoring(ScoringFunction):
    """Scoring function that uses binary logits on single tokens.

    This class implements scoring by computing logits for binary classification
    on individual tokens rather than full sequences.

    Args:
        *args: Arguments passed to parent class
        use_full_logprob: Whether to use full log probabilities
        max_length: Maximum sequence length
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(
        self,
        *args,
        use_full_logprob: bool = False,
        max_length: int = None,
        **kwargs,
    ):

        logger.info(f"Using {self.__class__.__name__}")
        super().__init__(*args, max_length=max_length, **kwargs)
        self.num_choices = 2
        self.use_full_logprob = use_full_logprob
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        self.choice_tokens = self.tokenizer(
            [self.fail_choice_str, self.pass_choice_str]
        )["input_ids"]

        assert all(len(t) == 1 for t in self.choice_tokens)
        self.choice_tokens = [t[0] for t in self.choice_tokens]
        logger.info(
            f"Pass token ID: {self.pass_choice_str}={self.choice_tokens[1]}"
        )
        logger.info(
            f"Fail token ID: {self.fail_choice_str}={self.choice_tokens[0]}"
        )
        # self.pad_side = "left"

    def preprocess_batch(
        self, batch: Dict[str, List], indices: List[int]
    ) -> Dict[str, List]:
        """Preprocess a batch of examples.

        Args:
            batch: Batch of examples to preprocess
            indices: List of indices for the batch

        Returns:
            dict: Preprocessed batch with tokenized inputs
        """
        out = {
            "idx": indices,  # used to keep track of the original index and group.
        }

        out.update(
            self.tokenizer(
                [
                    q + self.eval_completion
                    for i, q in enumerate(batch["query"])
                ],
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )
        )
        if self.add_bos:
            out["input_ids"] = [
                [self.tokenizer.bos_token_id] + i for i in out["input_ids"]
            ]
            out["attention_mask"] = [[1] + a for a in out["attention_mask"]]

        return out

    def score(
        self,
        logits: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor | None]:
        """Score a batch of examples.

        Args:
            logits: Model output logits
            batch: Batch of examples

        Returns:
            tuple: (scores, optional_losses)
        """
        lengths = batch["attention_mask"].sum(-1) - 1
        loss = None
        # Lengths need to be decremented by 1 to get the last token in the
        # 0-indexed sequence.
        logits = logits[torch.arange(logits.size(0)), lengths.to(logits.device)]

        if self.cfg.log_softmax_logits:
            out_scores = logits.log_softmax(-1)[:, self.choice_tokens].cpu()
        else:
            out_scores = logits[:, self.choice_tokens].cpu()

        if self.calculate_loss:
            # Loss is calculated for the last token only and needs to be log
            # softmax'ed
            if not self.cfg.log_softmax_logits:
                pos_loss = logits.log_softmax(-1)[
                    torch.arange(logits.size(0)),
                    torch.tensor(
                        [self.choice_tokens[1] for _ in batch["passed"]]
                    ),
                ].cpu()
                neg_loss = logits.log_softmax(-1)[
                    torch.arange(logits.size(0)),
                    torch.tensor(
                        [self.choice_tokens[0] for _ in batch["passed"]]
                    ),
                ].cpu()
                loss = torch.cat([neg_loss, pos_loss], dim=1)

            else:
                loss = out_scores

            return out_scores, -1 * loss

        return out_scores, None

    def postprocess_scores(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        timings: torch.Tensor,
        losses: Optional[torch.Tensor] = None,
        dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        indices = indices.view(-1).tolist()
        scores = scores.view(-1, 2)
        preds = scores.argmax(-1).tolist()
        scores = scores.tolist()
        if losses is None:
            losses = [None] * len(scores)
        else:
            losses = losses.tolist()
        for i, idx in enumerate(indices):
            r_dict = {
                "idx": idx,
                "score": scores[i][1],
                "failed_score": scores[i][0],
                "prediction": preds[i],
                "score_time": timings[i].item(),
            }
            if losses[i] is not None:
                label = 1 if dataset[idx]["passed"] else 0

                r_dict["point_loss"] = losses[i][label]
                r_dict["pair_loss"] = losses[i][label] - losses[i][1 - label]
            yield r_dict


class ClassificationScoring(ScoringFunction):
    """Scoring function that treats the task as classification.

    This class implements scoring by treating the task as a standard
    classification problem with cross-entropy loss.

    Args:
        *args: Arguments passed to parent class
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        logger.info(f"Using {self.__class__.__name__}")
        super().__init__(*args, **kwargs)
        self.num_choices = 2

    def preprocess_batch(
        self, batch: Dict[str, List], indices: List[int]
    ) -> Dict[str, List]:
        """Preprocess a batch of examples.

        Args:
            batch: Batch of examples to preprocess
            indices: List of indices for the batch

        Returns:
            dict: Preprocessed batch with tokenized inputs
        """
        out = {
            "idx": indices,  # used to keep track of the original index and group.
        }

        out.update(
            self.tokenizer(
                [q for q in batch["query"]],
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
        )
        if (
            self.add_bos
            and out["input_ids"][0][0] != self.tokenizer.bos_token_id
        ):
            out["input_ids"] = [
                [self.tokenizer.bos_token_id] + i for i in out["input_ids"]
            ]
            out["attention_mask"] = [[1] + a for a in out["attention_mask"]]

        out["input_ids"] = [
            i + [self.tokenizer.eos_token_id] for i in out["input_ids"]
        ]
        out["attention_mask"] = [a + [1] for a in out["attention_mask"]]
        return out

    def score(
        self,
        logits: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor | None]:
        """Score a batch of examples.

        Args:
            logits: Model output logits
            batch: Batch of examples

        Returns:
            tuple: (scores, optional_losses)
        """
        if self.calculate_loss:
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            logits = logits.cpu()
            return logits.cpu(), loss_fct(logits, batch["passed"].view(-1))
        return logits.cpu(), None

    def postprocess_scores(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        timings: torch.Tensor,
        losses: Optional[torch.Tensor] = None,
        **_,
    ) -> Generator[Dict[str, Any], None, None]:
        indices = indices.view(-1).tolist()
        scores = scores.view(-1, 2)
        preds = scores.argmax(-1).tolist()
        scores = scores.tolist()
        if losses is None:
            losses = [None] * len(scores)
        else:
            losses = losses.tolist()
        for i, idx in enumerate(indices):
            r_dict = {
                "idx": idx,
                "score": scores[i][1],
                "failed_score": scores[i][0],
                "prediction": preds[i],
                "score_time": timings[i].item(),
            }
            if losses[i] is not None:
                r_dict["point_loss"] = losses[i]
            yield r_dict


class RewardModelScoring(ScoringFunction):
    """Scoring function that uses a reward model approach.

    This class implements scoring using a reward model that directly
    predicts scores for inputs using MSE loss.

    Args:
        *args: Arguments passed to parent class
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        logger.info(f"Using {self.__class__.__name__}")
        super().__init__(*args, **kwargs)
        self.loss_fct = torch.nn.MSELoss(reduction="none")

    def preprocess_batch(
        self, batch: Dict[str, List], indices: List[int]
    ) -> Dict[str, List]:
        """Preprocess a batch of examples.

        Args:
            batch: Batch of examples to preprocess
            indices: List of indices for the batch

        Returns:
            dict: Preprocessed batch with tokenized inputs
        """
        out = {
            "idx": indices,  # used to keep track of the original index and group.
        }

        out.update(
            self.tokenizer(
                [q for q in batch["query"]],
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
        )
        if (
            self.add_bos
            and out["input_ids"][0][0] != self.tokenizer.bos_token_id
        ):
            out["input_ids"] = [
                [self.tokenizer.bos_token_id] + i for i in out["input_ids"]
            ]
            out["attention_mask"] = [[1] + a for a in out["attention_mask"]]

        out["input_ids"] = [
            i + [self.tokenizer.eos_token_id] for i in out["input_ids"]
        ]
        out["attention_mask"] = [a + [1] for a in out["attention_mask"]]
        return out

    def score(
        self,
        logits: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor | None]:
        """Score a batch of examples.

        Args:
            logits: Model output logits
            batch: Batch of examples

        Returns:
            tuple: (scores, optional_losses)
        """
        if self.calculate_loss:

            logits = logits
            return (
                logits.cpu(),
                self.loss_fct(
                    logits.view(-1), batch["passed"].view(-1).to(logits.device)
                ).cpu(),
            )
        return logits.cpu(), None

    def postprocess_scores(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        timings: torch.Tensor,
        losses: Optional[torch.Tensor] = None,
        **_,
    ) -> Generator[Dict[str, Any], None, None]:
        indices = indices.view(-1).tolist()
        scores = scores.view(-1).tolist()
        timings = timings.view(-1).tolist()
        if losses is None:
            losses = [None] * len(scores)
        else:
            losses = losses.tolist()

        for i, s, l, t in zip(indices, scores, losses, timings):
            r_dict = {"idx": i, "score": s, "score_time": t}
            if l is not None:

                r_dict["point_loss"] = l
            yield r_dict


class LogProbScoring(ScoringFunction):
    """Scoring function that uses log probabilities.

    This class implements scoring by computing log probabilities
    over sequences, with optional masking and completion handling.

    Args:
        *args: Arguments passed to parent class
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def preprocess_batch(
        self, batch: Dict[str, List], indices: List[int]
    ) -> Dict[str, List]:
        """Preprocess a batch of examples.

        Args:
            batch: Batch of examples to preprocess
            indices: List of indices for the batch

        Returns:
            dict: Preprocessed batch with tokenized inputs and masks
        """
        out = {
            "idx": indices,  # used to track the original index
        }

        # Tokenize the queries
        out.update(
            self.tokenizer(
                [
                    q
                    + (
                        batch["completion"][i]
                        if self.cfg.include_completion
                        else ""
                    )
                    for i, q in enumerate(batch["query"])
                ],
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
        )

        key_mask = "query"
        if self.cfg.use_problem_for_mask:
            key_mask = "problem"
        elif self.cfg.use_dummy_problem_for_mask:
            key_mask = "problem_dummy"

        query_toked = self.tokenizer(
            batch[key_mask],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["attention_mask"]
        out["query_mask"] = [
            [0] * len(qt) + [1] * len(out["input_ids"][i][len(qt) :])
            for i, qt in enumerate(query_toked)
        ]

        # Add BOS token if needed
        if (
            self.add_bos
            and out["input_ids"][0][0] != self.tokenizer.bos_token_id
        ):
            out["input_ids"] = [
                [self.tokenizer.bos_token_id] + i for i in out["input_ids"]
            ]
            out["attention_mask"] = [[1] + a for a in out["attention_mask"]]
            out["query_mask"] = [[0] + m for m in out["query_mask"]]

        # Add EOS token to each sequence
        out["input_ids"] = [
            i + [self.tokenizer.eos_token_id] for i in out["input_ids"]
        ]
        out["attention_mask"] = [a + [1] for a in out["attention_mask"]]
        out["query_mask"] = [m + [1] for m in out["query_mask"]]

        return out

    def score(
        self, logits: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Score a batch of examples.

        Args:
            logits: Model output logits
            batch: Batch of examples

        Returns:
            tuple: (scores, optional_losses)
        """
        # Get input IDs and shift for next token prediction
        # log_probs = triton_score(
        #     logits,
        #     batch["input_ids"].to(logits.device),
        #     batch["query_mask"].to(logits.device),
        # )
        # Get input IDs and shift for next token prediction
        input_ids = batch["input_ids"]

        # Shift logits and input_ids for next token prediction
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = batch["query_mask"][:, 1:].contiguous()
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

        # Gather the log probs of the actual next tokens
        log_probs = (
            torch.gather(
                log_probs,
                dim=-1,
                index=shift_labels.to(log_probs.device).unsqueeze(-1),
            )
            .squeeze(-1)
            .cpu()
        )

        # Apply the mask so only the completion tokens are considered
        log_probs = (log_probs * shift_mask).sum(dim=-1) / shift_mask.sum(
            dim=-1
        )
        if self.calculate_loss:
            # For loss calculation, we use the negative log probability
            return log_probs, -log_probs

        return log_probs.cpu(), None

    def postprocess_scores(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        timings: torch.Tensor,
        losses: Optional[torch.Tensor] = None,
        **_,
    ) -> Generator[Dict[str, Any], None, None]:
        indices = indices.view(-1).tolist()
        scores = scores.view(-1).tolist()
        timings = timings.view(-1).tolist()

        if losses is None:
            losses = [None] * len(scores)
        else:
            losses = losses.view(-1).tolist()

        for i, score, loss, t in zip(indices, scores, losses, timings):
            result = {
                "idx": i,
                "score": score,
                "score_time": t,
            }

            if loss is not None:
                result["point_loss"] = loss

            yield result


SCORING_METHODS = {
    "binary_logit": (BinaryLogitScoring, {}),
    "binary_logprob": (BinaryLogitScoring, {"use_full_logprob": True}),
    "st_binary_logit": (SingleTokenBinaryLogitScoring, {}),
    "st_binary_logprob": (
        SingleTokenBinaryLogitScoring,
        {"use_full_logprob": True},
    ),
    "classification": (ClassificationScoring, {}),
    "reward_model": (RewardModelScoring, {}),
    "logprob": (LogProbScoring, {}),
}


def load_scoring_method(
    scoring_cfg: ScoringConfig,
    tokenizer: PreTrainedTokenizer,
    pass_choice_str: str,
    fail_choice_str: str,
    eval_completion: str,
    **scoring_kwargs,
) -> ScoringFunction:
    """Load and instantiate a scoring method.

    Args:
        scoring_cfg: The scoring configuration
        tokenizer: The tokenizer to use
        pass_choice_str: String indicating a passing solution
        fail_choice_str: String indicating a failing solution
        eval_completion: String to append for evaluation
        **scoring_kwargs: Additional keyword arguments for scoring

    Returns:
        ScoringFunction: The instantiated scoring function

    Raises:
        ValueError: If an unknown scoring method is specified
    """

    if scoring_cfg.scoring_method in SCORING_METHODS:
        logger.info("Using scoring method: %s", scoring_cfg.scoring_method)
        score_cls, kwargs = SCORING_METHODS[scoring_cfg.scoring_method]
        logger.info("Scoring kwargs: %s", kwargs)

    else:
        raise ValueError(
            f"Unknown scoring method: {scoring_cfg.scoring_method}"
        )
    return score_cls(
        scoring_cfg,
        tokenizer,
        pass_choice_str=pass_choice_str,
        fail_choice_str=fail_choice_str,
        eval_completion=eval_completion,
        **scoring_kwargs,
        **kwargs,
    )


def group_by_task_id(
    records: List[Dict],
    dataset: Dataset,
    task_id_key: str = "task_id",
) -> Dict[str, List[ScoredSolution]]:
    """Group scoring records by task ID.

    Args:
        records: List of scoring records
        dataset: The dataset containing task information
        task_id_key: Key for task IDs in the dataset

    Returns:
        dict: Records grouped by task ID
    """
    logger.info("Grouping records by task")
    out = defaultdict(lambda: defaultdict(list))
    for record in records:
        row = dataset[record["idx"]]
        task_id = row[task_id_key]
        out[task_id]["score"].append(record["score"])
        out[task_id]["failed_score"].append(record.get("failed_score"))
        out[task_id]["passed_public"].append(row["passed_public"])
        out[task_id]["passed_private"].append(row["passed_private"])
        out[task_id]["passed"].append(row["passed"])
        out[task_id]["point_loss"].append(record.get("point_loss"))
        out[task_id]["pair_loss"].append(record.get("pair_loss"))
        out[task_id]["source"].append(row["source"])

    return out


def _get_records(grouped_scores) -> Dict:
    """Process grouped scores into evaluation metrics.

    Args:
        grouped_scores: Scores grouped by task

    Returns:
        dict: Dictionary of evaluation metrics
    """
    logger.info(f"Processing {len(grouped_scores):,} problems")
    ex = next(iter(grouped_scores.values()))
    has_pair_loss = ex["pair_loss"][0] is not None
    overall_metrics = defaultdict(list)
    for data in grouped_scores.values():
        scores = np.array(data["score"])
        pass_vals = []
        private_pass = []
        public_pass = []
        has_private = False
        pass_scores = []
        fail_scores = []
        for i, s in sorted(
            enumerate(scores),
            reverse=True,
            key=lambda x: (x[1], -x[0]),  # sort by Scores, then by index
        ):
            pass_vals.append(data["passed"][i])
            if data["passed_private"][i] is not None:
                private_pass.append(data["passed_private"][i])
                public_pass.append(data["passed_public"][i])
                has_private = True
            if data["passed"][i]:
                pass_scores.append(s)
            else:
                fail_scores.append(s)
            overall_metrics["point_loss"].append(data["point_loss"][i])
            if has_pair_loss:
                overall_metrics["pair_loss"].append(data["pair_loss"][i])
        pass_vals = np.array(pass_vals)
        pass_scores = np.array(pass_scores)
        fail_scores = np.array(fail_scores)
        assert len(set(data["source"])) == 1

        record = {
            "point_loss": np.mean(data["point_loss"]),
            "rs": ranking_score(pass_vals),
            "b_of_64": best_of_k(pass_vals, 64),
        }

        if has_private:
            private_pass = np.array(private_pass)
            public_pass = np.array(public_pass)
            record["private.rs"] = ranking_score(private_pass)
            record["public.rs"] = ranking_score(public_pass)
            record["private.b_of_64"] = best_of_k(private_pass, 64)
            record["public.b_of_64"] = best_of_k(public_pass, 64)

        for k, v in record.items():
            overall_metrics[k].append(v)

    return overall_metrics


def evaluate_dataset(
    scoring_fn: ScoringFunction,
    accelerator: Accelerator,
    model: PreTrainedModel,
    dataset: Dataset,
    max_tokens_per_batch: int,
    preprocessed_ds: Optional[Dataset] = None,
) -> Dict[str, float]:
    """Score and evaluate a dataset using a scoring function.

    Args:
        scoring_fn: The scoring function to use
        accelerator: The Hugging Face Accelerator
        model: The model to use for scoring
        dataset: The dataset to evaluate
        max_tokens_per_batch: Maximum tokens per batch
        preprocessed_ds: Optional preprocessed dataset

    Returns:
        dict: Dictionary of evaluation metrics
    """

    _, scores = scoring_fn(
        accelerator=accelerator,
        dataset=dataset,
        model=model,
        max_tokens_per_batch=max_tokens_per_batch,
        preprocessed_ds=preprocessed_ds,
    )

    overall_metrics = _get_records(group_by_task_id(scores, dataset))
    out = {}

    for k, v in overall_metrics.items():
        out[k] = np.mean(v)

    return {k: float(v) for k, v in out.items()}
