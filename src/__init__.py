from dataclasses import dataclass
from dataclasses import field
from typing import Optional

from omegaconf import MISSING

from src import training
from src.evaluation import EvalSuite
from src.evaluation import TaskConfig
from src.modeling import PRECISION_MAP
from src.modeling import ModelConfig
from src.preprocessing import PreprocessorConfig
from src.scoring import ScoringConfig
from src.utils import CONSOLE
from src.utils import JINJA_ENV
from src.utils import get_current_repo_hash


def register_configs(cs):
    cs.store(name="model", node=ModelConfig)
    cs.store(name="scoring", node=ScoringConfig)
    cs.store(name="preprocessing", node=PreprocessorConfig)
    cs.store(name="data", node=training.DataConfig)
    from src.evaluation import register_eval_configs

    register_eval_configs(cs)
    return cs
