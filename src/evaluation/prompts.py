import json
import logging
from pathlib import Path
from typing import Dict, List

from code_execution.eval_dataset.apps import NO_FN_NAME
from transformers import PreTrainedTokenizer

from src.utils import clean_solution

logger = logging.getLogger(__name__)
# These are taken from the LiveCodeBench dataset
with Path(__file__).parents[2].joinpath("data", "lcb_icl.json").open("r") as f:
    LCB_ICL = json.loads(f.read())
    for k, v in LCB_ICL.items():
        LCB_ICL[k] = [
            (
                f"# QUESTION:\n{q['question'].strip()}",
                f"```python\n{q['answer'].strip()}\n```",
            )
            for q in v
        ]

# From https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
CC_INST_MSG = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Ensure that when your code is ran, it reads the inputs from STDIN, runs the algorithm and writes output to STDOUT. You must import the necessary modules and write the entire program. **DO NOT** Write any tests or checks for your code. Just write the solution."

GSM8K_INST = "You are an expert Python programmer. You will be given a math word problem and must write the python function `solution()` that returns the correct answer. The function takes no arguments and must return the answer. You will NOT return anything except for the function."

# From https://github.com/evalplus/blob/master/evalplus/provider/utility.py#L26
EVALPLUS_SYS = "You are an intelligent programming assistant to produce Python algorithmic solutions"
EVALPLUS_FORMAT = """Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
{prompt}
```
"""


_CHAT_SPLITTER_ = "[[]]-abc-easy-as-123--a[][]"
EVALPLUS_RESPONSE_PREFIX = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python\n{resp}\n```"

QWEN_SYS_MSG = "You are a helpful assistant."
QWQ_SYS_MSG = "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."
DEEPSEEK_R1_SYS_MSG = (
    "A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)


def make_gsm_icl(
    examples, question_is_doc: bool = False, is_chat: bool = False
):
    icl = []
    for q, a in zip(examples["questions"], examples["solutions"]):
        nq = f"Q: {q}\nWrite the solution in Python.\n"

        if question_is_doc:
            na = f'def solution():\n    """{q}"""\n{a}'
        else:
            na = f"def solution():\n{a}"

        a = f"```python\n{clean_solution(na)}\n```"
        if not is_chat:
            a = "A:\n" + a
        icl.append((nq, a))

    return icl


def apply_chat_template(
    chats: List[Dict],
    tokenizer: PreTrainedTokenizer,
    remove_bos: bool = True,
    add_generation_prefix: bool = False,
):
    prompt = tokenizer.apply_chat_template(
        chats,
        tokenize=False,
        add_generation_prompt=add_generation_prefix,
    ).split(_CHAT_SPLITTER_)[0]
    if (
        tokenizer.bos_token is not None
        and remove_bos
        and prompt.startswith(tokenizer.bos_token)
    ):
        prompt = prompt[len(tokenizer.bos_token) :]
    return prompt


def get_prompt(
    chats: List[Dict],
    tokenizer: PreTrainedTokenizer,
    is_chat: bool = False,
    model_name: str = None,
    add_generation_prefix: bool = False,
):

    assert chats[0]["role"] == "user"
    if not is_chat:
        assert len(chats) == 1
        return chats[0]["content"]

    if model_name.startswith("deepseek-ai/DeepSeek-R1"):
        chats = [{"role": "system", "content": DEEPSEEK_R1_SYS_MSG}, *chats]
    elif model_name.startswith("Qwen/Qwen2.5-Coder"):
        chats = [{"role": "system", "content": QWEN_SYS_MSG}, *chats]
    elif model_name.startswith("Qwen/QwQ"):
        chats = [{"role": "system", "content": QWQ_SYS_MSG}, *chats]

    return apply_chat_template(
        chats,
        tokenizer,
        remove_bos=True,
        add_generation_prefix=add_generation_prefix,
    )


def get_evalplus_prompt(
    doc: Dict,
    is_chat: bool = False,
    add_py_md: bool = True,
):
    prompt = EVALPLUS_FORMAT.format(prompt=doc["prompt"].strip())
    if add_py_md:
        response_prefix = EVALPLUS_RESPONSE_PREFIX.format(resp=_CHAT_SPLITTER_)
    else:
        response_prefix = ""

    if is_chat:
        chats = [
            {"role": "user", "content": prompt},
        ]
        if add_py_md:
            chats.append(
                {"role": "assistant", "content": response_prefix},
            )
    else:
        if add_py_md:
            prompt = f"{prompt}\n\n{response_prefix.split(_CHAT_SPLITTER_)[0]}"
        chats = [{"role": "user", "content": prompt}]
    return chats


def get_gsm_prompt(
    doc,
    icl: List,
    is_chat: bool = False,
    add_py_md: bool = True,
):
    """Gets the GSM8K prompt."""
    if is_chat:
        chats = []
        for q, a in icl:
            chats.append({"role": "user", "content": q})
            chats.append({"role": "assistant", "content": a})

        chats.append(
            {
                "role": "user",
                "content": f"Q: {doc['question']}\nWrite the solution in Python.\n",
            }
        )
        chats[0]["content"] = f"{GSM8K_INST}\n\n{chats[0]['content']}"

        if add_py_md:
            chats.append(
                {
                    "role": "assistant",
                    "content": f"```python\n{_CHAT_SPLITTER_}\n```",
                }
            )

    else:
        icl = "\n\n".join([f"{q}\n\n{a}" for q, a in icl])
        prompt = GSM8K_INST + "\n\n" + icl + "\n\n" + f"Q: {doc['question']}"
        prompt += "\nWrite the solution in Python.\n\nA:\n"
        if add_py_md:
            prompt += "```python\n"
        chats = [{"role": "user", "content": prompt}]
    return chats


def get_cc_prompt(
    doc,
    is_chat: bool = False,
    add_py_md: bool = True,
):
    """Gets the CodeContests prompt."""

    if is_chat:

        question = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
        question += f"Question: {doc['description']}\n\n"
        question += CC_INST_MSG + "\n"
        question += "```python\n# YOUR CODE HERE\n```\n\n"

        chats = [{"role": "user", "content": question}]
        if add_py_md:
            chats.append(
                {
                    "role": "assistant",
                    "content": f"```python\n{_CHAT_SPLITTER_}\n```",
                }
            )
    else:
        prompt = f"### Question\n{doc['description']}\n\n"
        prompt += f"### ANSWER:\n"
        if add_py_md:
            prompt += "```python\n"
        chats = [{"role": "user", "content": prompt}]

    return chats
