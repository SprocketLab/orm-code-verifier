import dataclasses
import logging
from typing import Dict, List, Optional

from omegaconf import MISSING

from src.utils import JINJA_ENV
from src.utils import clean_solution

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PreprocessorConfig:
    problem: str = MISSING
    program: str = MISSING
    outcome: str = MISSING
    prompt: str = MISSING
    completion: str = MISSING
    pass_choice_str: str = MISSING
    fail_choice_str: str = MISSING
    black_format: bool = False
    remove_comments: bool = False
    problem_start_delim: Optional[str] = None
    problem_end_delim: Optional[str] = None
    solution_start_delim: Optional[str] = None
    solution_end_delim: Optional[str] = None
    instruction: str = ""
    disable_prompt_newline: bool = False


class Preprocessor:
    def __init__(
        self,
        cfg: PreprocessorConfig,
    ):
        self.cfg: PreprocessorConfig = cfg

        self.problem_template = JINJA_ENV.from_string(cfg.problem)
        self.program_template = JINJA_ENV.from_string(cfg.program)
        self.outcome_template = JINJA_ENV.from_string(cfg.outcome)

        self.prompt_template = JINJA_ENV.from_string(cfg.prompt)
        self.completion_template = JINJA_ENV.from_string(cfg.completion)

        self.instruction = JINJA_ENV.from_string(cfg.instruction).render(
            fail_choice=cfg.fail_choice_str, pass_choice=cfg.pass_choice_str
        )

    @property
    def pass_token(self):
        return self.cfg.pass_choice_str

    @property
    def fail_token(self):
        return self.cfg.fail_choice_str

    def render_problem(
        self,
        problem: str,
    ) -> str:

        return self.problem_template.render(
            problem=problem,
        )

    def render_program(
        self,
        program: str,
    ) -> str:
        return self.program_template.render(
            program=program,
            fail_choice=self.cfg.fail_choice_str,
            pass_choice=self.cfg.pass_choice_str,
        )

    def render_outcome(self, passed: bool) -> str:
        return self.outcome_template.render(
            outcome=self.pass_token if passed else self.fail_token,
        )

    def render_icl_example(
        self, problem: str, program: str, outcome: bool
    ) -> str:
        prompt = self.render_prompt(
            problem=problem,
            program=program,
            no_instruction=True,
        )
        completion = self.render_completion(outcome)
        return prompt + completion

    def render_prompt(
        self,
        problem: str,
        program: str,
        icl_examples: List[str] = [],
        no_instruction: bool = False,
    ) -> str:
        """Renders the prompt with the given problem and program."""
        prompt = self.prompt_template.render(
            problem=problem,
            program=program,
            fail_choice=self.cfg.fail_choice_str,
            pass_choice=self.cfg.pass_choice_str,
            instruction="" if no_instruction else self.instruction,
            icl_examples=icl_examples,
        )
        if not prompt.endswith("\n") and not self.cfg.disable_prompt_newline:
            prompt += "\n"
        return prompt

    def render_completion(self, outcome: bool) -> str:

        return self.completion_template.render(
            outcome=self.render_outcome(outcome),
            fail_choice=self.cfg.fail_choice_str,
            pass_choice=self.cfg.pass_choice_str,
        )

    def render_eval_completion(self) -> str:
        return self.completion_template.render(
            outcome="",
            fail_choice=self.cfg.fail_choice_str,
            pass_choice=self.cfg.pass_choice_str,
        )
