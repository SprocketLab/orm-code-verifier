import ast
import difflib
import logging
import random
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from code_execution import safe_ast_parse
from omegaconf import MISSING
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from src.utils import CONSOLE

logger = logging.getLogger(__name__)

PRECISION_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


MODEL_NAMES = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-160m-dedupped": "EleutherAI/pythia-160m-dedupped",
    "cg-350m-mu": "Salesforce/codegen-350M-multi",
    "ds-code-1b": "deepseek-ai/deepseek-coder-1.3b-base",
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-14m": "EleutherAI/pythia-14m",
    "pythia-1_4b": "EleutherAI/pythia-1.4b",
    "gpt2": "openai-community/gpt2",
    "gpt-neo-125m": "EleutherAI/gpt-neo-125m",
    "pythia-160m-seed1": "EleutherAI/pythia-160m-seed1",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "codeparrot-small": "codeparrot/codeparrot-small",
    "qwen25-coder-500m": "Qwen/Qwen2.5-Coder-0.5B",
    "qwen25-coder-1_5b": "Qwen/Qwen2.5-Coder-1.5B",
    "qwen25-coder-3b": "Qwen/Qwen2.5-Coder-3B",
    "qwen25-coder-7b": "Qwen/Qwen2.5-Coder-7B",
    "qc-inst-500m": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qc-inst-1_5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qc-inst-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qc-inst-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qc-inst-14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "qwq": "Qwen/QwQ-32B-AWQ",
    "r1-distill-qwen-1_5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

MODEL_TYPE_SHORT_MAP = {
    "causal-lm": "clm",
    "classification": "cls",
    "reward-model": "rm",
    "reward-model-dropout": "rm-dropout",
    "classification-dropout": "cls-dropout",
}


USE_MODEL_NAME = {
    "qwen25-coder": ("Qwen 2.5 Coder", "QWC2.5"),
    "pythia": ("Pythia", "Py"),
}


def get_model_generic_name(model_name) -> Tuple[str, str]:
    for k, v in USE_MODEL_NAME.items():
        if model_name.startswith(k):
            return v

    return model_name, model_name


# Helper function for sorting model sizes
def get_model_size(model_name):
    """Extract and convert model size to float for sorting."""
    size = model_name.lower().split("-")[-1].replace("_", ".")
    if size.endswith("m"):
        modif = 1e6
    elif size.endswith("b"):
        modif = 1e9
    else:
        modif = 1
    size_str = size[:-1]

    return float(size_str) * modif


@dataclass
class ModelConfig:
    name: str = MISSING
    ckpt_path: Optional[str] = None
    model_type: str = "causal-lm"
    init_kwargs: Dict = field(default_factory=dict)
    step: Optional[int] = None
    epoch: Optional[int] = None
    hidden_dropout: Optional[float] = None
    attention_dropout: Optional[float] = None
    classifier_dropout: Optional[float] = None
    generic_name: Optional[str] = None
    short_name: Optional[str] = None
    model_size: Optional[float] = None

    def __post_init__(self):
        if self.model_type not in MODEL_TYPE_SHORT_MAP:
            raise ValueError(f"Invalid model type '{self.model_type}'")

        self.generic_name, self.short_name = get_model_generic_name(self.name)
        self.model_size = get_model_size(self.name)
        if self.ckpt_path is not None:
            self.ckpt_path = Path(self.ckpt_path)
            if not self.ckpt_path.exists():
                raise ValueError(f"{self.ckpt_path} does not exist")
            if not self.ckpt_path.is_dir():
                raise ValueError(f"{self.ckpt_path} must be a directory")
            self.ckpt_path = self.find_checkpoint(self.ckpt_path)

        if "pythia" in self.name:
            if self.hidden_dropout is not None:
                self.init_kwargs["hidden_dropout"] = self.hidden_dropout
            if self.attention_dropout is not None:
                self.init_kwargs["attention_dropout"] = self.attention_dropout
            if "dropout" in self.model_type:
                if self.classifier_dropout is not None:
                    self.init_kwargs["classifier_dropout"] = (
                        self.classifier_dropout
                    )
            elif (
                self.classifier_dropout is not None
                and self.classifier_dropout > 0.0
            ):
                logger.warning(
                    "classifier_dropout is set but model_type is not a dropout model"
                )
                raise ValueError(
                    "classifier_dropout is set but model_type is not a dropout model"
                )
        elif "qwen" in self.name:
            if self.attention_dropout is not None:
                self.init_kwargs["attention_dropout"] = self.attention_dropout

            if (
                self.classifier_dropout is not None
                and self.classifier_dropout > 0.0
            ) or (
                self.hidden_dropout is not None and self.hidden_dropout > 0.0
            ):
                logger.warning(
                    "hidden_dropout and classifier_dropout are set but Qwen models do not support them"
                )
                raise ValueError(
                    "hidden_dropout and classifier_dropout are set but Qwen models do not support them"
                )

    @property
    def short_type(self):
        """Returns the short type name of the model"""
        return MODEL_TYPE_SHORT_MAP[self.model_type]

    def find_checkpoint(self, model_path: Path):
        """Finds the last checkpoint in the given path."""
        if not model_path.name.startswith("checkpoint-"):
            CONSOLE.print(f"Looking for last checkpoint in {model_path}")
            try:
                model_path = max(
                    self.ckpt_path.glob("checkpoint-*"),
                    key=lambda x: int(x.name.split("-")[1]),
                )
            except ValueError as e:
                raise ValueError(f"No checkpoint found in {model_path}") from e
        return model_path

    def is_checkpoint(self):
        """Returns True if the model is a checkpoint"""
        return self.ckpt_path is not None

    @property
    def model_path(self):
        """Returns the model path to be used for loading the model"""
        if self.is_checkpoint():
            return self.ckpt_path
        return MODEL_NAMES.get(self.name, self.name)


class GPT2ClassifierModel(transformers.GPT2PreTrainedModel):
    def __init__(
        self,
        config,
        classifier_dropout: float = 0.0,
        use_attn_mask_lengths: bool = False,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = transformers.GPT2Model(config)
        self.dropout_hidden = nn.Dropout(classifier_dropout)

        self.classification_head = nn.Linear(
            config.vocab_size if self.use_embed else config.hidden_size,
            self.num_labels,
            bias=False,
        )

        self.use_attn_mask_lengths = use_attn_mask_lengths

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[
            Union[Cache, Tuple[Tuple[torch.FloatTensor]]]
        ] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.classification_head(self.dropout_hidden(hidden_states))

        if self.use_attn_mask_lengths:
            sequence_lengths = (attention_mask.sum(-1) - 1).to(logits.device)
        else:
            sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(input_ids.size(0), device=logits.device),
            sequence_lengths,
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GPTNeoClassificationModel(transformers.GPTNeoXPreTrainedModel):
    def __init__(
        self,
        config,
        use_attn_mask_lengths: bool = False,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = transformers.GPTNeoXModel(config)
        self.dropout_hidden = nn.Dropout(config.classifier_dropout)
        self.use_attn_mask_lengths = use_attn_mask_lengths
        self.classification_head = nn.Linear(
            config.hidden_size,
            self.num_labels,
            bias=False,
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[
            Union[Cache, Tuple[Tuple[torch.FloatTensor]]]
        ] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        logits = self.classification_head(self.dropout_hidden(hidden_states))

        if self.use_attn_mask_lengths:
            sequence_lengths = (attention_mask.sum(-1) - 1).to(logits.device)
        else:
            sequence_lengths = (
                torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1)
                - 1
            )
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)

        pooled_logits = logits[
            torch.arange(input_ids.size(0), device=logits.device),
            sequence_lengths,
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_model_and_tokenizer(
    model_cfg: ModelConfig,
    device: str,
    precision: str = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:

    logger.info(
        "Using model of type '%s'(class='%s') from '%s'",
        model_cfg.name,
        model_cfg.model_type,
        model_cfg.model_path,
    )

    model_kwargs = model_cfg.init_kwargs
    if precision is not None:
        logger.info("Using precision '%s'", precision)
        model_kwargs["torch_dtype"] = PRECISION_MAP.get(precision)

    for k, v in model_kwargs.items():
        if isinstance(v, DictConfig):
            model_kwargs[k] = OmegaConf.to_container(v)

    logger.info("init_kwargs=%s", model_kwargs)
    if model_cfg.model_type == "causal-lm":
        model_class = transformers.AutoModelForCausalLM
    elif model_cfg.model_type in {"classification", "reward-model"}:
        if "pythia" in model_cfg.name:
            model_class = GPTNeoClassificationModel
        elif "gpt2" in model_cfg.name:
            model_class = GPT2ClassifierModel
        else:
            model_class = transformers.AutoModelForSequenceClassification
    elif model_cfg.model_type in {
        "classification-dropout",
        "reward-model-dropout",
    }:
        if "pythia" in model_cfg.name:
            model_class = GPTNeoClassificationModel
        else:
            raise ValueError(
                f"Invalid model type '{model_cfg.model_type}' for {model_cfg.name}"
            )
    else:
        raise ValueError(f"Invalid model type '{model_cfg.model_type}'")
    logger.debug("Using model class '%s'", model_class.__name__)

    model = model_class.from_pretrained(
        model_cfg.model_path, **model_kwargs
    ).to(torch.device(device))
    logger.info("Loading tokenizer for '%s'", model_cfg.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_path, use_fast=True
    )
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        logger.debug(
            "Setting pad token to eos token '%s'(id=%d)",
            tokenizer.eos_token,
            tokenizer.eos_token_id,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
