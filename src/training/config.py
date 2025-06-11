import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import MISSING
from transformers import TrainingArguments

from src.modeling import PRECISION_MAP
from src.modeling import ModelConfig
from src.preprocessing import PreprocessorConfig
from src.scoring import ScoringConfig
from src.utils import get_current_repo_hash


@dataclasses.dataclass
class DataConfig:
    pass_level: str = "all"
    require_pf: bool = False
    include_syntax: bool = False
    include_failed: bool = True
    max_prog_chars: int = 10000
    only_dataset: Optional[str] = None

    def __post_init__(self):
        if self.pass_level not in {"public", "private", "generated", "all"}:
            raise ValueError(
                "pass_level must be one of 'public', 'private','generated','all'"
            )


@dataclasses.dataclass
class TrainerConfig:
    model: ModelConfig = MISSING
    scoring: ScoringConfig = MISSING
    preprocessing: PreprocessorConfig = MISSING
    data: DataConfig = MISSING
    setup_name: str = MISSING
    batch_size: int = MISSING
    examples_per_problem: int = 1
    num_train_examples: Optional[int] = None
    max_problem_tokens: int = 1536
    max_input_length: int = 2048
    max_eval_length: int = 2048
    seed: int = 1
    precision: str = "fp16"
    name_suffix: Optional[str] = None
    sort_by_length: bool = False
    current_repo_hash: str = dataclasses.field(
        default_factory=get_current_repo_hash
    )
    num_workers: int = 1
    sub_group: Optional[str] = None
    entity_name: Optional[str] = None
    project_name: Optional[str] = None
    trainer_cls: str = "hf"
    gradient_accumulation_steps: int = 1
    eval_batch_tokens: int = 2048 * 4
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    num_train_epochs: int = 3
    warmup_steps: int = 0
    save_only_model: bool = True
    logging_steps: int = 10
    gradient_checkpointing: bool = False
    warmup_ratio: float = 0.0
    fp16_opt_level: str = "O1"
    half_precision_backend: str = "auto"
    lr_scheduler_type: str = "cosine"
    group_by_length: bool = False
    length_column_name: str = "length"
    optim: str = "adamw_torch_fused"
    auto_find_batch_size: bool = False
    lr_min: float = 0.0
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    map_batch_size: int = 100
    deepspeed: Dict[str, Any] = dataclasses.field(default_factory=dict)

    save_strategy: str = "steps"
    save_steps: float = 0.1
    eval_strategy: str = "steps"
    eval_steps: float = 0.2

    negative_label: int = 0
    positive_label: int = 1

    def __post_init__(self):

        if self.precision not in PRECISION_MAP:
            raise ValueError(f"Invalid precision: {self.precision}")

    def get_name(self):
        out = [
            self.setup_name,
            f"seed{self.seed}",
        ]

        if self.name_suffix:
            out.append(self.name_suffix)
        if self.examples_per_problem > 1:
            out.append(f"ex{self.examples_per_problem}")

        return "_".join(out)

    def make_args(self, **kwargs):
        return TrainingArguments(
            **kwargs,
        )
