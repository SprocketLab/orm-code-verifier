import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ujson
from code_execution.eval_dataset.apps import NO_FN_NAME
from datasets import Dataset
from tqdm import tqdm
from vllm import LLM
from vllm import SamplingParams
from vllm.outputs import RequestOutput

logger = logging.getLogger(__name__)


def seconds_to_human(seconds):
    """Converts a number of seconds into a human-readable format.

    Args:
        seconds (float): the number of seconds to convert

    Returns:
        str: the number of seconds in the format "HH:MM:SS.SS"
    """

    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


class LLMWithProg(LLM):
    """A wrapper for LLM that adds logging capabilities.

    Args:
        *args: Variable length argument list to be passed to the LLM superclass.
        **kwargs: Arbitrary keyword arguments to be passed to the LLM superclass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def _run_engine(self, use_tqdm: bool):
        self.logger.info("Running engine")
        # Initialize tqdm.
        num_requests = self.llm_engine.get_num_unfinished_requests()
        # Run the engine.
        outputs = []
        last_update = 0
        t0 = time.time()
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()

            for output in step_outputs:
                if output.finished:
                    last_update += 1
                    outputs.append(output)

            if last_update >= 10 or len(outputs) == num_requests:
                t1 = time.time()
                rate = (t1 - t0) / len(outputs)
                eta = rate * (num_requests - len(outputs))
                eta = seconds_to_human(eta)
                last_update = len(outputs)
                current_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
                self.logger.info(
                    f"[{current_time}] Finished {len(outputs)}/{num_requests} requests in {seconds_to_human(t1-t0)}. Rem={eta}"
                )
                last_update = 0

        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs


class LLMPredictor:
    """A LLM predictor for generating code based on prompts. Main purposes are for custom serialization and handling of prompts.

    Args:
        model_path: The path to the model checkpoint.
        tp_size: The size of the tensor parallelism.
        sampling_params: The parameters for sampling.
        seed: The seed for random generation.
        visible_devices: The visible devices for the model.
        num_repeat: The number of times to repeat the generation.
        **vllm_kwargs: The keyword arguments for the LLM.

    Returns:
        A predictor object.
    """

    def __init__(
        self,
        model_path: str,
        tp_size: int,
        sampling_params: SamplingParams,
        seed: int,
        visible_devices: str,
        num_repeat: int = 1,
        **vllm_kwargs,
    ):

        print("STARTING_PREDICTOR")
        if visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        self.model_name = model_path
        self.sampling_params = sampling_params
        self.num_finished = 0
        self.num_generated = 0
        self.start_time = datetime.now()
        self.sampling_params = sampling_params
        self.num_repeat = num_repeat
        # Create an LLM.
        self.llm = LLMWithProg(
            model=model_path,
            tensor_parallel_size=tp_size,
            seed=seed,
            **vllm_kwargs,
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        query_map = {}
        if self.num_repeat:
            print(
                f"Breaking {self.sampling_params.n} per into {self.num_repeat} * {self.sampling_params.n} = {self.sampling_params.n * self.num_repeat}"
            )
            queries = []
            for i, q in enumerate(batch["query"]):
                # Repeat each prompt the specified number of times
                for _ in range(self.num_repeat):
                    query_map[len(queries)] = i
                    queries.append(q)
        else:
            # Use each prompt exactly once
            query_map = {i: i for i in range(len(batch["query"]))}
            queries = batch["query"]
        outputs = self.llm.generate(queries, self.sampling_params)

        preds = {}

        i = 0
        while outputs:
            output = outputs.pop(0)
            actual_idx = query_map[i]
            assert queries[i] == output.prompt
            assert batch["query"][actual_idx] == output.prompt

            i += 1
            self.num_finished += 1
            self.num_generated += len(output.outputs)

            pred_dict = {
                "predictions": [],  # List of generated text predictions
                "cum_log_probs": [],  # Cumulative log probabilities for each prediction
                "num_tokens": [],  # Number of tokens in each prediction
            }
            for o in output.outputs:
                pred_dict["predictions"].append(o.text)
                pred_dict["cum_log_probs"].append(float(o.cumulative_logprob))
                pred_dict["num_tokens"].append(len(o.logprobs))

            if actual_idx not in preds:
                # Add the first prediction for this query
                preds[actual_idx] = pred_dict
                preds[actual_idx]["prompt"] = output.prompt
            else:
                # Append additional predictions for this query
                assert preds[actual_idx]["prompt"] == output.prompt
                preds[actual_idx]["predictions"].extend(
                    pred_dict["predictions"]
                )
                preds[actual_idx]["cum_log_probs"].extend(
                    pred_dict["cum_log_probs"]
                )
                preds[actual_idx]["num_tokens"].extend(pred_dict["num_tokens"])

        elapsed = datetime.now() - self.start_time
        print(
            f"Finished {self.num_finished:,} in {seconds_to_human(elapsed.total_seconds())}"
        )
        print(
            f"{self.num_finished:,} prompts with {self.num_generated:,} predictions"
        )

        # Construct the output dictionary by filtering out the "predictions" key from the batch
        out_dict = {
            k: v for k, v in batch.items() if k not in ["predictions", "query"]
        }

        # Prepare lists to store query results
        out_dict["query"] = []
        out_dict["completions"] = []  # List of JSON-encoded predictions
        out_dict["cum_log_probs"] = (
            []
        )  # List of JSON-encoded cumulative log probabilities
        out_dict["num_tokens"] = []  # List of JSON-encoded token counts

        # Iterate over the processed predictions and populate the output dictionary
        for _, p_dict in sorted(preds.items(), key=lambda x: x[0]):
            out_dict["query"].append(p_dict["prompt"])
            out_dict["completions"].append(ujson.dumps(p_dict["predictions"]))
            out_dict["cum_log_probs"].append(
                ujson.dumps(p_dict["cum_log_probs"])
            )
            out_dict["num_tokens"].append(ujson.dumps(p_dict["num_tokens"]))

        return out_dict


def serialize_raw_predictions(
    task_name: str,
    raw_predictions: List[List[RequestOutput]],
    dataset: Dataset,
    expected_num_preds: int,
    prompt_key: str = "prompt",
    include_token_logprobs: bool = False,
    row_keys_add: Optional[List[str]] = None,
):
    """Serializes raw predictions for a given task."""
    logger.debug(f"Serializing raw predictions for '{task_name}'")

    logger.debug("Using enumerate as idx key.")
    gen = enumerate(raw_predictions)

    out = []
    for idx, result in tqdm(
        gen,
        desc="Serializing raw predictions",
        total=len(raw_predictions),
    ):

        row = dataset[idx]

        assert len(set(p.prompt for p in result)) == 1
        assert all(row[prompt_key] in p.prompt for p in result)
        meta = {
            "prompt_tokens": len(result[0].prompt_token_ids),
            "row_id": idx,
            prompt_key: row[prompt_key],
        }

        outputs = [
            (i, o.index, o) for i, r in enumerate(result) for o in r.outputs
        ]

        predictions = []

        for cid, (*_, pred) in enumerate(
            sorted(outputs, key=lambda x: (x[0], x[1]))
        ):
            p_dict = {
                "completion_id": cid,
                "prediction": pred.text,
                "cumulative_logprob": pred.cumulative_logprob,
                "num_tokens": len(pred.token_ids),
            }
            if include_token_logprobs:
                logprobs = []
                for t_dict in pred.logprobs:
                    token_info = min(t_dict.values(), key=lambda x: x.rank)
                    logprobs.append(
                        [token_info.decoded_token, token_info.logprob]
                    )
                p_dict["token_logprobs"] = logprobs

            predictions.append(p_dict)
        assert len(predictions) == expected_num_preds

        if row_keys_add:
            for key in row_keys_add:
                if key == prompt_key:
                    continue
                meta[key] = row[key]
        meta["predictions"] = predictions
        out.append(meta)
    return out
