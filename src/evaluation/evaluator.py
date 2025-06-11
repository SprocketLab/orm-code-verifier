import dataclasses
import gzip
import logging
from collections import defaultdict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import ujson
from accelerate import Accelerator
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from src import metrics
from src import scoring
from src import utils
from src.evaluation.configs import EvalSuite
from src.evaluation.filter_functions import run_filter_on_dataset
from src.preprocessing import Preprocessor
from src.preprocessing import PreprocessorConfig

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        seed: int,
        scoring_cfg: scoring.ScoringConfig,
        preprocessor_cfg: PreprocessorConfig,
        num_workers: int,
        max_tokens_per_batch: int,
        sort_by_length: bool = False,
    ):
        self.seed = seed
        self.scoring_cfg = scoring_cfg
        self.num_workers = num_workers
        self.max_tokens_per_batch = max_tokens_per_batch
        self.preprocessor_cfg = preprocessor_cfg
        self.sort_by_length = sort_by_length

    def _postprocess_score(self, row: Dict, result: Dict):
        # This idx is used to get the scoring dataset row, so we dont need
        # it anymore.
        row.pop("problem")
        row.pop("problem_dummy")
        row.pop("query")
        row.pop("completion")
        result.pop("idx")
        row["exec_elapsed"] = row.pop("elapsed")
        return {**result, **row}

    def group_scores(
        self,
        scores: List[Dict],
        filtered_out: Optional[Dataset],
    ) -> Dict[str, List[metrics.ScoredSolution]]:
        logger.info("Grouping scores")
        out = defaultdict(list)
        for record in tqdm(scores, desc="Grouping Scores", miniters=100):
            out[record.pop("_identifier")].append(
                metrics.ScoredSolution(
                    sid=record["sid"],
                    completion_id=record["completion_id"],
                    score=record["score"],
                    score_time=record["score_time"],
                    passed=record["passed"],
                    passed_public=record["passed_public"],
                    passed_plus=record["passed_plus"],
                    num_tests_passed=record["num_tests_passed"],
                    count=record["count"],
                    logprob=record["logprob"],
                )
            )

        if filtered_out is None:
            return out

        logger.info(f"Adding back {len(filtered_out)} filtered out solutions")
        for row in filtered_out:
            problem_idx = row["_identifier"]
            for sol in row["solutions"]:

                out[problem_idx].append(
                    metrics.ScoredSolution(
                        sid=sol["sid"],
                        completion_id=sol["completion_id"],
                        passed=sol["passed"],
                        passed_public=sol["passed_public"],
                        passed_plus=sol["passed_plus"],
                        num_tests_passed=sol["num_tests_passed"],
                        logprob=sol["logprob"],
                        count=sol["count"],
                        score=float("-inf"),
                        filtered_out=True,
                        score_time=0.0,
                    )
                )
        return out

    def __call__(
        self,
        dataset: Dataset,
        accelerator: Optional[Accelerator],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        suite: EvalSuite,
        preproc_batch_size: int = 1,
        debug_num: int = None,
    ) -> Tuple[scoring.ScoringTimings, List[Dict], Dataset]:

        start = datetime.now(timezone.utc)
        rng = utils.set_seed(self.seed)

        if suite.shuffle:
            logger.info("Shuffling dataset")
            dataset = dataset.shuffle(rng=rng)

        if debug_num is not None:
            logger.warning(f"Debugging with {debug_num} examples")
            dataset = dataset.map(lambda x: {"len": len(x["query"])})
            dataset = dataset.sort("len", reverse=True)
            dataset = dataset.select(range(debug_num))
            dataset = dataset.remove_columns("len")
        preprocessor = Preprocessor(self.preprocessor_cfg)
        score_function = scoring.load_scoring_method(
            self.scoring_cfg,
            tokenizer=tokenizer,
            pass_choice_str=self.preprocessor_cfg.pass_choice_str,
            fail_choice_str=self.preprocessor_cfg.fail_choice_str,
            eval_completion=preprocessor.render_eval_completion(),
            num_workers=self.num_workers,
            max_length=tokenizer.model_max_length,
            preproc_batch_size=preproc_batch_size,
            postprocess_fn=self._postprocess_score,
        )

        scoring_ds = score_function.preprocess_dataset(
            dataset, sort_by_length=self.sort_by_length
        )
        logger.debug(
            f"Example Query:\n{tokenizer.decode(scoring_ds['input_ids'][0])}"
        )
        logger.debug(f"Scoring ds keys: {scoring_ds.column_names}")
        timings, scores = score_function(
            dataset=dataset,
            accelerator=accelerator,
            model=model,
            max_tokens_per_batch=self.max_tokens_per_batch,
            preprocessed_ds=scoring_ds,
        )

        timings.evaluation = (
            datetime.now(timezone.utc) - start
        ).total_seconds()

        return timings, scores

    def save_scores(
        self,
        scores: Dict[int, List[metrics.ScoredSolution]],
        raw_dataset: Dataset,
        out_file: Path,
    ):
        logger.info(f"Saving {len(scores):,} problems to '{out_file.name}'")
        with gzip.open(out_file, "wt") as f:
            for problem_idx, scored_solutions in tqdm(
                scores.items(), total=len(scores), desc="Saving scores"
            ):
                out_dict = {
                    k: v
                    for k, v in raw_dataset[problem_idx].items()
                    if k not in {"problem", "meta"}
                }
                timings = []
                solutions = [None] * len(scored_solutions)
                for i, solution in enumerate(scored_solutions):
                    timings.append(solution.score_time)
                    solutions[i] = dataclasses.asdict(solution)
                    solutions[i].pop("solution")

                out_dict["solutions"] = solutions
                out_dict["net_elapsed"] = sum(timings)
                out_dict["mean_elapsed"] = float(np.mean(timings))
                out_dict["median_elapsed"] = float(np.median(timings))
                out_dict["std_elapsed"] = float(np.std(timings))

                f.write(ujson.dumps(out_dict) + "\n")
