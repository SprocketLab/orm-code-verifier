import dataclasses
import logging
from typing import Dict, List, Optional, Set

from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TaskConfig:
    dataset: str = MISSING

    clean_solutions: bool = False
    remove_comments: bool = False
    icl_examples: Optional[str] = None
    icl_prob_style: str = "problem"

    def __post_init__(self):
        self.dataset = self.dataset.lower()


@dataclasses.dataclass
class MBPPConfig(TaskConfig):
    dataset: str = "mbpp"
    problem_test_idx: int = 0


@dataclasses.dataclass
class GSM8KConfig(TaskConfig):
    dataset: str = "gsm8k"


@dataclasses.dataclass
class HumanEvalConfig(TaskConfig):
    dataset: str = "humaneval"
    include_problem_str: bool = True


@dataclasses.dataclass
class CodeContestsConfig(TaskConfig):
    dataset: str = "code_contests"


@dataclasses.dataclass
class EvalSuite:
    name: str = MISSING
    dataset: str = MISSING
    tasks: Dict[str, TaskConfig] = MISSING
    filter_function: Optional[str] = None
    clean_solutions: bool = False
    shuffle: bool = False
    remove_comments: bool = False
    propagate_to_tasks: bool = False
    icl_examples: Dict[str, List[Dict]] = dataclasses.field(
        default_factory=dict
    )
    preproc_batch_size: int = 5000
    filter_batch_size: int = 100

    rstrip_query: bool = True

    def __post_init__(self):

        if self.propagate_to_tasks:
            for k, v in self.tasks.items():
                if self.clean_solutions and not v.clean_solutions:
                    logger.debug(f"Propagating clean_solutions to {k}")
                    v.clean_solutions = True
                if self.remove_comments and not v.remove_comments:
                    logger.debug(f"Propagating remove_comments to {k}")
                    v.remove_comments = True

    def __getitem__(self, key):
        return self.tasks[key]

    def should_clean_solutions(self, task_name):
        return self.clean_solutions or self.tasks[task_name].clean_solutions

    def should_remove_comments(self, task_name):
        return self.remove_comments or self.tasks[task_name].remove_comments

    def get_icl_examples(self, task_name):
        task_cfg = self.tasks[task_name]
        if task_cfg.icl_examples is None:
            return []
        return self.icl_examples.get(task_cfg.icl_examples, [])
