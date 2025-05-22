"""Preprocessing module for creating training and evaluation sequences for code verification tasks.

This module provides functionality to format natural language problems, Python solutions, and their
outcomes into structured sequences using Jinja2 templates. It is designed to create consistent
training and evaluation data for code verification tasks.

The module supports:
- Formatting natural language problems and Python solutions
- Adding in-context learning examples for few-shot learning
- Customizable templates for problems, solutions, and outcomes
- Configurable delimiters to mark problem and solution boundaries

Example using the qsol_task template:
    ```python
    cfg = PreprocessorConfig(
        problem="# Question\n{{ problem }}",
        program="# Proposed Solution\n```python\n{{ program }}\n```",
        outcome="{{ outcome }}",
        prompt="{% if instruction %}{{ instruction }}{% endif %}"
              "{% if problem %}{{ problem }}{% endif %}"
              "{{ program }}",
        completion="\n# Is the solution correct? {{ outcome }}",
        pass_choice_str="[Yes]",
        fail_choice_str="[No]"
    )
    preprocessor = Preprocessor(cfg)

    # Example usage
    problem = "Write a function that adds two numbers"
    program = "def add(a, b):\\n    return a + b"
    formatted = preprocessor.render_prompt(problem, program)
    ```
"""

import dataclasses
import logging
from typing import List, Optional

from omegaconf import MISSING

from src.utils import JINJA_ENV

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PreprocessorConfig:
    """Configuration class for the Preprocessor.

    This class defines the structure and formatting templates used by the Preprocessor
    to format problems, programs, and their outcomes. It supports customizable templates
    for creating training and evaluation sequences.

    Attributes:
        problem (str): Jinja template for formatting natural language problem statements
        program (str): Jinja template for formatting Python solution code
        outcome (str): Jinja template for formatting verification outcomes
        prompt (str): Jinja template for the complete sequence structure, including ICL examples
        completion (str): Jinja template for the completion/response structure
        pass_choice_str (str): Token indicating a correct solution (e.g. "[Yes]")
        fail_choice_str (str): Token indicating an incorrect solution (e.g. "[No]")
        black_format (bool): Whether to apply black formatting to Python code (not implemented)
        remove_comments (bool): Whether to remove comments from code (not implemented)
        problem_start_delim (Optional[str]): Optional delimiter to mark where problem text begins
        problem_end_delim (Optional[str]): Optional delimiter to mark where problem text ends
        solution_start_delim (Optional[str]): Optional delimiter to mark where solution code begins
        solution_end_delim (Optional[str]): Optional delimiter to mark where solution code ends
        instruction (str): Additional instruction text to prepend to the sequence
        disable_prompt_newline (bool): Whether to disable adding newline to prompts
    """

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
    """Handles preprocessing and formatting of code verification sequences.

    This class uses Jinja2 templates defined in PreprocessorConfig to format
    natural language problems, Python solutions, and their outcomes into structured
    sequences for training and evaluation.

    The preprocessor supports:
    - Formatting natural language problems into a consistent structure
    - Formatting Python solution code with proper delimiters
    - Adding in-context learning examples for few-shot learning
    - Customizable templates for different sequence formats

    Args:
        cfg (PreprocessorConfig): Configuration object defining templates and formatting options
    """

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
    def pass_token(self) -> str:
        """Returns the token string indicating a passing result."""
        return self.cfg.pass_choice_str

    @property
    def fail_token(self) -> str:
        """Returns the token string indicating a failing result."""
        return self.cfg.fail_choice_str

    def render_problem(
        self,
        problem: str,
    ) -> str:
        """Renders a problem statement using the configured template.

        Args:
            problem (str): The problem statement to format

        Returns:
            str: The formatted problem statement
        """
        return self.problem_template.render(
            problem=problem,
        )

    def render_program(
        self,
        program: str,
    ) -> str:
        """Renders a program using the configured template.

        Args:
            program (str): The program code to format

        Returns:
            str: The formatted program code
        """
        return self.program_template.render(
            program=program,
            fail_choice=self.cfg.fail_choice_str,
            pass_choice=self.cfg.pass_choice_str,
        )

    def render_outcome(self, passed: bool) -> str:
        """Renders the outcome (pass/fail) using the configured template.

        Args:
            passed (bool): Whether the program passed verification

        Returns:
            str: The formatted outcome string
        """
        return self.outcome_template.render(
            outcome=self.pass_token if passed else self.fail_token,
        )

    def render_icl_example(
        self, problem: str, program: str, outcome: bool
    ) -> str:
        """Renders a complete in-context learning example.

        Creates a formatted example combining the problem, program, and outcome
        that can be used as an in-context learning example for few-shot learning.
        The format matches the main sequence format but includes the outcome.

        Args:
            problem (str): The natural language problem statement
            program (str): The Python solution code
            outcome (bool): Whether the solution is correct

        Returns:
            str: The formatted ICL example ready to be used in a sequence
        """
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
        """Renders a complete sequence with optional ICL examples and instructions.

        This method combines the problem statement, program code, and optional components
        into a complete sequence for training or evaluation.

        Args:
            problem (str): The natural language problem statement
            program (str): The Python solution code
            icl_examples (List[str], optional): List of formatted in-context learning
                examples to prepend to the sequence. These help with few-shot learning.
            no_instruction (bool, optional): Whether to exclude the instruction text
                from the sequence.

        Returns:
            str: The complete formatted sequence
        """
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
        """Renders a completion response using the configured template.

        Args:
            outcome (bool): Whether the program passed verification

        Returns:
            str: The formatted completion string
        """
        return self.completion_template.render(
            outcome=self.render_outcome(outcome),
            fail_choice=self.cfg.fail_choice_str,
            pass_choice=self.cfg.pass_choice_str,
        )

    def render_eval_completion(self) -> str:
        """Renders an empty completion template for evaluation purposes.

        Returns:
            str: The formatted completion template with an empty outcome
        """
        return self.completion_template.render(
            outcome="",
            fail_choice=self.cfg.fail_choice_str,
            pass_choice=self.cfg.pass_choice_str,
        )
