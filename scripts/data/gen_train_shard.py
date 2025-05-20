import gzip
import sys
from collections import defaultdict
from pathlib import Path

import click
import ujson
from rich.console import Console
from tqdm import tqdm
from vllm import LLM
from vllm import SamplingParams

print(Path(__file__).parents[2].absolute())
sys.path.append(str(Path(__file__).parents[2].absolute()))
from src import generation
from src.evaluation import prompts
from src.evaluation.utils import EVAL_TASKS
from src.evaluation.utils import STOP_WORDS
from src.evaluation.utils import generate_eval_ds
from src.modeling import MODEL_NAMES
from src.utils import CONSOLE
from src.utils import DEFAULT_OUT_DIR
from src.utils import set_seed

STOP_WORDS = {
    "gsm8k": {
        "\n```",
        "\nif",
        "\n#",
        "\n@",
        "\nprint",
        "<file_sep>",
        "\nclass",
    },
    "code_contests": set(["\n```"]),
}

CONSOLE = Console()


@click.command()
@click.argument(
    "input_shard",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=Path
    ),
)
@click.option(
    "--output_dir", type=Path, default=Path(DEFAULT_OUT_DIR, "train_shards")
)
@click.option("--num_repeat", type=int, default=1)
@click.option("--num_samples", type=int, default=128)
@click.option("--temperature", type=float, default=1.0)
@click.option("--top_p", type=float, default=0.95)
@click.option("--top_k", type=int, default=-1)
@click.option("--max_new_tokens", type=int, default=1024)
@click.option("--swap_space", type=int, default=16)
@click.option("--max_num_seqs", type=int, default=256)
@click.option("--seed", type=int, default=42)
@click.option("--debug_num", type=int, default=None)
@click.option("--quantized", is_flag=True)
@click.option("--gpu_utilization", type=float, default=0.9)
def main(
    input_shard: Path,
    output_dir: Path,
    num_repeat: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    swap_space: int,
    max_num_seqs: int,
    seed: int,
    debug_num: int,
    quantized: bool,
    gpu_utilization: float,
):

    task_name = input_shard.parent.name
    model_name = input_shard.parent.parent.name
    save_dir = output_dir / model_name.replace("/", "_") / task_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if task_name not in STOP_WORDS:
        raise ValueError(f"Task name {task_name} not supported")
    CONSOLE.print(f"Loading shard from {input_shard}")
    with gzip.open(input_shard, "rt", encoding="utf-8") as f:
        queries = list(map(ujson.loads, f))
    if debug_num is not None:
        CONSOLE.print(f"Debug mode, only processing {debug_num} queries")
        queries = queries[:debug_num]
    CONSOLE.print(f"Loaded {len(queries):,} queries from {input_shard}")

    sampling_parameters = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        logprobs=1,
        n=num_samples,
        stop=list(STOP_WORDS[task_name]),
        seed=seed,
    )
    CONSOLE.print(f"Sampling parameters: {sampling_parameters}")
    full_model_name = MODEL_NAMES.get(model_name, model_name)
    if quantized:
        full_model_name += "-GPTQ-Int8"
    CONSOLE.print(f"Using model: {full_model_name}")
    llm = LLM(
        full_model_name,
        seed=seed,
        max_num_seqs=max_num_seqs,
        swap_space=swap_space,
        gpu_memory_utilization=gpu_utilization,
    )

    results = defaultdict(lambda: defaultdict(list))
    query_strings = [q["query"] for q in queries]
    # Repeat the entire batch num_repeat times
    for _ in range(num_repeat):
        # Generate for all queries in a single batch
        outputs = list(
            sorted(
                llm.generate(query_strings, sampling_parameters),
                key=lambda x: int(x.request_id),
            )
        )

        current_idx = 0
        while outputs:
            output = outputs.pop(0)
            results[current_idx]["prompt"].append(output.prompt)
            # Process all outputs
            for o in output.outputs:
                results[current_idx]["predictions"].append(o.text)
                results[current_idx]["cum_log_probs"].append(
                    float(o.cumulative_logprob)
                )
                results[current_idx]["num_tokens"].append(len(o.logprobs))
            current_idx += 1

    example_pred = results[0]["predictions"][0]
    CONSOLE.print(f"Example prediction:\n{example_pred}")

    # Save the results
    output_file = save_dir / f"{input_shard.stem.split('.jsonl')[0]}.jsonl.gz"
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        for i, prob_preds in tqdm(results.items(), desc="Writing results"):
            meta = queries[i]
            assert (
                len(prob_preds["predictions"])
                == len(prob_preds["cum_log_probs"])
                == len(prob_preds["num_tokens"])
            )
            assert len(prob_preds["predictions"]) == num_samples * num_repeat
            assert all(p == meta["query"] for p in prob_preds["prompt"])
            meta["predictions"] = prob_preds["predictions"]
            meta["cum_log_probs"] = prob_preds["cum_log_probs"]
            meta["num_tokens"] = prob_preds["num_tokens"]
            f.write(ujson.dumps(meta) + "\n")


if __name__ == "__main__":
    main()
