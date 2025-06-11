from src.evaluation import prompts
from src.evaluation.configs import CodeContestsConfig
from src.evaluation.configs import EvalSuite
from src.evaluation.configs import GSM8KConfig
from src.evaluation.configs import HumanEvalConfig
from src.evaluation.configs import MBPPConfig
from src.evaluation.configs import TaskConfig
from src.evaluation.entrypoints import evaluate_suite
from src.evaluation.evaluator import Evaluator


# Monkey patch the task.
def fewshot_examples(_):
    import json

    """Loads and returns the few-shot examples for the task if they exist."""
    with open(
        "data/gsm8k_few_shot_prompts.json",
        "r",
        encoding="utf-8",
    ) as file:
        examples = json.load(file)
    return examples


import bigcode_eval.tasks

bigcode_eval.tasks.gsm.Gsm8k.fewshot_examples = fewshot_examples


def register_eval_configs(cs):

    cs.store(name="task_cfg", node=TaskConfig)
    cs.store(name="eval_suite", node=EvalSuite)
    cs.store(name="humaneval_cfg", node=HumanEvalConfig)
    cs.store(name="mbpp_cfg", node=MBPPConfig)
    cs.store(name="gsm8k_cfg", node=GSM8KConfig)
    cs.store(name="codecontests_cfg", node=CodeContestsConfig)
