import gzip
import json
import logging
import os
from pathlib import Path
from typing import Optional, Set, Tuple

import click
import torch
import wandb
import yaml
from accelerate import Accelerator
from hydra import compose
from hydra import initialize
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from jinja2 import Environment
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict

from src import CONSOLE
from src import evaluation
from src import modeling
from src import register_configs as default_register_configs
from src import utils
from src.preprocessing import PreprocessorConfig
from src.scoring import ScoringConfig

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
JINJA_ENV = Environment()


TRAIN_CFG_NAME = "train_config.yaml"
TRAIN_EXAMPLES_NAME = "train_examples.csv.gz"
os.environ["ACCELERATE_DOWNCAST_BF16"] = "true"


@click.group()
@click.argument("generator_model", type=str)
@click.argument("sampling_setup", type=str)
@click.option("--device", type=str, default=None)
@click.option("--debug", is_flag=True)
@click.option("--verbose", is_flag=True)
@click.option("--overwrite", is_flag=True)
@click.option("--tags", "-t", multiple=True)
@click.option(
    "--output_dir",
    type=click.Path(dir_okay=True, path_type=Path),
    default=Path(utils.DEFAULT_OUT_DIR, "evaluation"),
)
@click.option("--debug_num_probs", type=int, default=None)
@click.option("--group_name", "-group", type=str, default=None)
@click.option("--max_tokens_per_batch", type=int, default=4096)
@click.option("--sub_group", type=str, default=None)
@click.option("--seed", type=int, default=1)
@click.option("--precision", type=str, default="fp16")
@click.option("--debug_num", type=int, default=None)
@click.option("--num_workers", type=int, default=1)
@click.option("--disable_wandb", is_flag=True)
@click.option("--preproc_batch_size", type=int, default=5000)
@click.option("--sort_by_length", is_flag=True)
@click.pass_context
def cli(
    ctx,
    generator_model: str,
    sampling_setup: str,
    device: str,
    precision: str,
    **kwargs,
):
    ctx.ensure_object(dict)
    if device not in {"mps", "cpu"}:
        print(f"{device=}")
        if device.isdigit():
            device = f"cuda:{device}"

        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]

    if precision not in modeling.PRECISION_MAP:
        raise ValueError(f"Invalid precision: {precision}")
    CONSOLE.print(f"{device=}")
    ctx.obj["device"] = device
    ctx.obj["precision"] = precision
    ctx.obj["generator_model"] = generator_model
    ctx.obj["sampling_setup"] = sampling_setup
    os.environ["PRECISION"] = precision
    ctx.obj.update(kwargs)

    cs = ConfigStore.instance()
    default_register_configs(cs)


DATASETS_SHORT_NAME = {
    "humaneval": "he",
    "mbpp": "mbpp",
}


def update_config(
    old_cfg: DictConfig,
    raw_cfg: DictConfig,
    new_cfg: object,
    dict_keys: Set[str] = None,
    ignore_keys: Set[str] = None,
):
    dict_keys = dict_keys or set()
    ignore_keys = ignore_keys or set()
    for k, v in old_cfg.items():
        if k in ignore_keys:
            continue
        if not hasattr(new_cfg, k):
            CONSOLE.print(f"WARNING: '{k}' is not longer part of config.")
            continue

        current_val = getattr(new_cfg, k, None)
        if k in dict_keys:
            if v != current_val:
                CONSOLE.print(
                    f"Overwriting model init_kwargs from '{current_val}' to '{v}'"
                )
                getattr(new_cfg, k).update(v)
            continue
        if current_val != v:
            CONSOLE.print(
                f"Overwriting model config value '{k}' from '{current_val}' to '{v}'",
                markup=False,
            )
            setattr(new_cfg, k, v)
            raw_cfg[k] = v
    return raw_cfg, new_cfg


def run_evaluation(
    ctx: click.Context,
    run_name: str,
    out_dir: Path,
    eval_suite: evaluation.EvalSuite,
    model_cfg: modeling.ModelConfig,
    scoring_cfg: ScoringConfig,
    preprocessor_cfg: PreprocessorConfig,
    cfg_dict: DictConfig,
    is_base: bool = False,
):
    eval_suite.preproc_batch_size = ctx.obj["preproc_batch_size"]
    wandb_run = None
    if not ctx.obj["disable_wandb"]:
        if is_base:
            group_name = "baseline"
        elif ctx.obj["group_name"] is not None:
            group_name = ctx.obj["group_name"]
        else:
            group_name = "checkpoint"
        wandb_run = utils.setup_wandb(
            cfg=cfg_dict,
            job_type="evaluation",
            run_name=run_name,
            group=group_name,
            run_dir=out_dir,
            tags=[t.strip() for t in ctx.obj["tags"]],
        )

    accelerator = None
    if ctx.obj["precision"] == "fp32":

        logger.info("Disabling attention implementation")
        model_cfg.init_kwargs["attn_implementation"] = None
    else:
        logger.info("Using mixed precision")
        accelerator = Accelerator(
            device_placement=False, mixed_precision=ctx.obj["precision"]
        )
    device = ctx.obj["device"]
    logger.info(f"Visible device: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"Loading model and tokenizer to device {device}")
    logger.info(f"ctx={ctx.obj}")
    model, tokenizer = modeling.load_model_and_tokenizer(
        model_cfg=model_cfg,
        device=device,
    )

    if accelerator is not None:
        model = accelerator.prepare_model(model, evaluation_mode=True)
    results = evaluation.evaluate_suite(
        generator_model=ctx.obj["generator_model"],
        sampling_setup=ctx.obj["sampling_setup"],
        seed=ctx.obj["seed"],
        accelerator=accelerator,
        scoring_cfg=scoring_cfg,
        suite=eval_suite,
        preprocessor_cfg=preprocessor_cfg,
        model=model,
        tokenizer=tokenizer,
        num_workers=ctx.obj["num_workers"],
        max_tokens_per_batch=ctx.obj["max_tokens_per_batch"],
        debug_num_probs=ctx.obj["debug_num_probs"],
        out_directory=out_dir,
        debug_num=ctx.obj["debug_num"],
        preproc_batch_size=eval_suite.preproc_batch_size,
        sort_by_length=ctx.obj["sort_by_length"],
    )

    if wandb_run is not None:
        wandb_run.log(results, step=1)
        result_artifact = wandb.Artifact(wandb_run.name, type="eval_results")
        result_artifact.add_file(out_dir / "results.jsonl.gz")
        result_artifact.add_file(out_dir / "config.yaml")
        result_artifact.add_file(out_dir / "evaluation.log")
        if (out_dir / TRAIN_CFG_NAME).exists():
            result_artifact.add_file(out_dir / TRAIN_CFG_NAME)
        if (out_dir / TRAIN_EXAMPLES_NAME).exists():
            result_artifact.add_file(out_dir / TRAIN_EXAMPLES_NAME)
        result_artifact.save()
        wandb_run.finish()
    logger.info(f"Git hash: {utils.get_current_repo_hash()}")
    results.update(
        {
            "run_name": run_name,
            "hash": utils.get_current_repo_hash(),
        }
    )
    with out_dir.joinpath("results.json").open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    del model


@cli.command("base")
@click.argument("model_name", type=str)
@click.argument("cfg_name", type=str)
@click.argument("eval_suite", type=str)
@click.option("--variant_name", type=str, default=None)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def evaluate_base(
    ctx: click.Context,
    model_name: str,
    cfg_name: str,
    eval_suite: str,
    variant_name: Optional[str],
    overrides: Tuple[str],
):
    run_name = f"baseline-{cfg_name}-{eval_suite}-{ctx.obj['generator_model']}-{ctx.obj['sampling_setup']}"
    if variant_name is not None:
        run_name += f"-{variant_name}"
    with initialize(config_path="configs", version_base=None):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                *overrides,
                f"suite={eval_suite}",
                f"model.name={model_name}",
                f"+evaluation={cfg_name}",
            ],
        )

    with open_dict(cfg):
        cfg.precision = ctx.obj["precision"]
        cfg.generator_model = ctx.obj["generator_model"]
        cfg.sampling_setup = ctx.obj["sampling_setup"]

    logger.info(f"Evaluating base model {model_name} on")
    logger.info(f"\t{eval_suite=}")
    logger.info(f"\tgenerator_model={cfg.generator_model}")
    logger.info(f"\tsampling_setup={cfg.sampling_setup}")

    run_name, out_dir = utils.setup_env(
        "evaluation",
        out_dir=ctx.obj["output_dir"],
        run_name=run_name,
        cfg=cfg,
        debug=ctx.obj["debug"],
        verbose=ctx.obj["verbose"],
        overwrite=ctx.obj["overwrite"],
        group_name=ctx.obj["group_name"],
    )
    suite = instantiate(cfg.suite, _convert_="object")
    logger.debug(f"{suite=}")
    model_cfg = instantiate(cfg.model, _convert_="object")
    logger.debug(f"{model_cfg=}")
    scoring_cfg = instantiate(cfg.scoring, _convert_="object")
    logger.debug(f"{scoring_cfg=}")
    preprocessor_cfg = instantiate(cfg.preprocessing, _convert_="object")
    logger.debug(f"{preprocessor_cfg=}")

    run_evaluation(
        ctx=ctx,
        run_name=run_name,
        out_dir=out_dir,
        eval_suite=suite,
        model_cfg=model_cfg,
        scoring_cfg=scoring_cfg,
        preprocessor_cfg=preprocessor_cfg,
        cfg_dict=cfg,
        is_base=True,
    )


@cli.command("checkpoint")
@click.argument(
    "model_dir", type=click.Path(exists=True, dir_okay=True, path_type=Path)
)
@click.argument("eval_suite", type=str)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--use_config",
    "-cfg",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--variant_name", type=str, default=None)
@click.pass_context
def evaluate_checkpoint(
    ctx: click.Context,
    model_dir: Path,
    eval_suite: str,
    overrides: Tuple[str],
    variant_name: str | None,
    use_config: Optional[Path],
):
    _ = use_config
    print(f"{model_dir=}")

    if model_dir.stem.startswith("checkpoint-"):
        train_dir = model_dir.parent
    else:
        train_dir = model_dir
        try:
            model_dir = max(
                model_dir.glob("checkpoint-*"),
                key=lambda x: int(x.name.split("-")[1]),
            )
        except ValueError as e:
            raise ValueError(f"No checkpoint found in {model_dir}") from e

    run_name = (
        train_dir.name,
        model_dir.stem.split("-")[-1],
        ctx.obj["generator_model"],
        ctx.obj["sampling_setup"],
        eval_suite,
    )
    run_name = "-".join(run_name)
    with (train_dir / "config.yaml").open("r") as f:
        train_cfg = OmegaConf.create(yaml.safe_load(f))

    with initialize(config_path="configs/suite", version_base=None):
        eval_suite_cfg = compose(
            config_name=f"{eval_suite}.yaml",
            overrides=[*overrides, "dataset=gabeorlanski/code_ranking_eval"],
        )
    suite = instantiate(eval_suite_cfg, _convert_="object")
    with open_dict(train_cfg):
        train_cfg["suite"] = eval_suite_cfg
        train_cfg["precision"] = ctx.obj["precision"]
        if "attn_dropout" in train_cfg["model"]["init_kwargs"]:
            train_cfg["model"]["init_kwargs"]["attn_dropout"] = None
        train_cfg["generator_model"] = ctx.obj["generator_model"]
        train_cfg["sampling_setup"] = ctx.obj["sampling_setup"]
    logger.info(f"Evaluating checkpoint {model_dir} on")
    logger.info(f"\t{eval_suite=}")
    logger.info(f"\tgenerator_model={train_cfg.generator_model}")
    logger.info(f"\tsampling_setup={train_cfg.sampling_setup}")

    run_name, out_dir = utils.setup_env(
        "evaluation",
        out_dir=ctx.obj["output_dir"],
        run_name=run_name,
        cfg=train_cfg,
        debug=ctx.obj["debug"],
        verbose=ctx.obj["verbose"],
        overwrite=ctx.obj["overwrite"],
        group_name=ctx.obj["group_name"],
    )
    if variant_name is not None:
        run_name += f"-{variant_name}"

    logger.info("Saving train config to '%s'", out_dir / "train_config.yaml")
    with open(out_dir / TRAIN_CFG_NAME, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(train_cfg, resolve=True, sort_keys=False))
    if (train_dir / "train_examples.csv").exists():

        logger.info(f"Copying train examples file to '{out_dir}'")
        with open(train_dir / "train_examples.csv", "r", encoding="utf-8") as f:
            with gzip.open(out_dir / TRAIN_EXAMPLES_NAME, "wt") as out_f:
                out_f.write(f.read())
    model_cfg = modeling.ModelConfig(**OmegaConf.to_container(train_cfg.model))
    model_cfg.ckpt_path = model_dir
    logger.debug(f"{model_cfg=}")
    scoring_cfg = ScoringConfig(**OmegaConf.to_container(train_cfg.scoring))
    logger.debug(f"{scoring_cfg=}")
    preprocessor_cfg = PreprocessorConfig(
        **OmegaConf.to_container(train_cfg.preprocessing)
    )

    logger.debug(f"{preprocessor_cfg=}")
    run_evaluation(
        ctx=ctx,
        run_name=run_name,
        out_dir=out_dir,
        eval_suite=suite,
        model_cfg=model_cfg,
        scoring_cfg=scoring_cfg,
        preprocessor_cfg=preprocessor_cfg,
        cfg_dict=train_cfg,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
