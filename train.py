import argparse
import csv
import functools
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed
import wandb
from datasets import Dataset
from hydra import compose
from hydra import initialize
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import open_dict
from transformers import PreTrainedTokenizer

from src import CONSOLE
from src import modeling
from src import register_configs as default_register_configs
from src import scoring
from src import training
from src import utils
from src.preprocessing import Preprocessor

logger = logging.getLogger(__name__)

CFG_DIR = Path(__file__).parent / "configs"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("experiment", type=Path)
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        help="Output directory",
        default=Path(utils.DEFAULT_OUT_DIR),
    )
    parser.add_argument(
        "--config_file",
        "-cfg",
        type=Path,
        help="Path to the config file",
        default=Path("configs", "train.yaml"),
    )
    parser.add_argument(
        "--group_name",
        "-group",
        type=str,
        help="Group variant to use for wandb",
        default=None,
    )
    parser.add_argument(
        "--real_batch_size",
        type=int,
        help="Real batch size",
        default=64,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=4,
    )
    parser.add_argument(
        "--device", type=str, help="Device to use", default="cpu"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing run"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug_train_examples",
        "-train_examples",
        type=int,
        default=None,
        help="Number of training examples to use for debugging",
    )
    parser.add_argument(
        "--debug_val_examples",
        "-val_examples",
        type=int,
        default=None,
        help="Number of validation examples to use for debugging",
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
    )
    parser.add_argument("--disable_cpu_offload", action="store_true")
    parser.add_argument("--sort_by_length", action="store_true")
    parser.add_argument("--max_val_per_prob", "-vpp", default=None)
    parser.add_argument("--tags", "-t", nargs="+", default=[])
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--disable_deepspeed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.resume and args.overwrite:
        raise ValueError("Cannot resume and overwrite")

    # overrides = [f"+training={args.train_setup}"] + overrides
    if args.max_val_per_prob is not None and args.max_val_per_prob < 2:
        raise ValueError("max_val_per_prob must be at least 2")
    if args.device not in {"mps", "cpu"}:
        if args.device.isdigit():
            args.device = f"cuda:{args.device}"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]

    CONSOLE.print(f"{args.device=}")
    return args, overrides


def register_configs(cs: ConfigStore):
    default_register_configs(cs)
    cs.store(name="hf", node=training.HFTrainerConfig, group="trainers")
    cs.store(name="dpo", node=training.DPOTrainingConfig, group="trainers")
    cs.store(name="rm", node=training.RewardTrainingConfig, group="trainers")


def get_train_val_datasets(
    rng,
    cfg: training.TrainerConfig,
    tokenizer: PreTrainedTokenizer,
    flags,
) -> Tuple[Dataset, Dataset]:
    logger.info("Making train and validation datasets")

    train_dataset = training.load_training_dataset(
        cfg=cfg,
        tokenizer=tokenizer,
        debug_num_problems=flags.debug_train_examples,
    )

    validation_dataset = training.make_validation_dataset(
        cfg=cfg,
        per_prob=flags.max_val_per_prob,
        tokenizer=tokenizer,
    )

    train_dataset.shuffle(seed=cfg.seed)
    if cfg.num_train_examples:
        logger.warning(
            f"Using only {cfg.num_train_examples}/{len(train_dataset)} training examples"
        )
        train_dataset = train_dataset.select(
            rng.choice(len(train_dataset), cfg.num_train_examples)
        )

    logger.info(f"Shuffling {len(validation_dataset)} validation examples")
    validation_dataset = validation_dataset.shuffle(seed=cfg.seed)
    return train_dataset, validation_dataset


def main(
    flags,
    overrides,
):
    cs = ConfigStore.instance()
    register_configs(cs)
    rel_path = flags.config_file.relative_to(Path("configs"))
    setup_name = flags.experiment.stem
    if flags.variant:
        setup_name = f"{setup_name}_{flags.variant}"

    overrides = [
        f"+experiments={flags.experiment.stem}",
        f"setup_name={setup_name}",
    ] + (overrides or [])
    overrides.extend(
        [
            f"batch_size={flags.batch_size}",
            f"gradient_accumulation_steps={int(flags.real_batch_size / flags.batch_size)    }",
        ]
    )

    CONSOLE.print(f"{overrides=}")

    with initialize(config_path="configs", version_base=None):
        raw_cfg = compose(config_name=str(rel_path), overrides=overrides)

    with open_dict(raw_cfg):
        if (
            "pythia" in raw_cfg.model.name
            and "causal" in raw_cfg.model.model_type
        ):
            raw_cfg.scoring.log_softmax_logits = True
    cfg: training.TrainerConfig = instantiate(raw_cfg, _convert_="object")

    run_name, out_dir = utils.setup_env(
        job_type="training",
        cfg=raw_cfg,
        out_dir=flags.output_dir,
        run_name=cfg.get_name(),
        debug=flags.debug,
        overwrite=flags.overwrite,
        verbose=flags.verbose,
        resume=flags.resume,
        delete_existing=True,
        group_name=flags.group_name or setup_name,
    )

    rng = utils.set_seed(cfg.seed)

    wandb_run = None
    if flags.enable_wandb:
        logger.info("Starting wandb run")
        if flags.group_name:
            group_name = flags.group_name
        else:
            group_name = setup_name
        wandb_run = utils.setup_wandb(
            cfg=raw_cfg,
            job_type="training",
            run_name=run_name,
            group=group_name,
            run_dir=out_dir,
            tags=flags.tags,
        )
    if cfg.deepspeed:
        ds_config = cfg.deepspeed
        ds_config["gradient_accumulation_steps"] = (
            cfg.gradient_accumulation_steps
        )
        ds_config["train_batch_size"] = (
            cfg.batch_size * cfg.gradient_accumulation_steps
        )

        ds_config["train_micro_batch_size_per_gpu"] = cfg.batch_size
    else:
        ds_config = None
    training_args = cfg.make_args(
        output_dir=out_dir,
        save_only_model=True,
        fp16=cfg.precision == "fp16",
        bf16=cfg.precision == "bf16",
        seed=cfg.seed,
        eval_strategy=cfg.eval_strategy,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        warmup_steps=cfg.warmup_steps,
        warmup_ratio=cfg.warmup_ratio,
        save_strategy=cfg.eval_strategy,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.eval_steps,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        data_seed=cfg.seed,
        group_by_length=cfg.group_by_length,
        length_column_name=cfg.length_column_name,
        half_precision_backend=cfg.half_precision_backend,
        fp16_opt_level=cfg.fp16_opt_level,
        lr_scheduler_type=cfg.lr_scheduler_type,
        include_num_input_tokens_seen=False,
        include_tokens_per_second=False,
        run_name=run_name,
        report_to="wandb" if flags.enable_wandb else [],
        dataloader_drop_last=True,
        optim=cfg.optim,
        auto_find_batch_size=cfg.auto_find_batch_size,
        full_determinism=False,
        deepspeed=ds_config,
    )

    model, tokenizer = modeling.load_model_and_tokenizer(
        cfg.model,
        device="cuda:0" if flags.device not in {"mps", "cpu"} else flags.device,
    )

    train_ds, validation_ds = get_train_val_datasets(
        rng=rng,
        cfg=cfg,
        tokenizer=tokenizer,
        flags=flags,
    )
    if isinstance(cfg, training.RewardTrainingConfig):
        logger.info("Using reward trainer")
        make_trainer_fn = training.make_reward_trainer
        example_list, train_ds = training.make_rm_dataset(
            train_dataset=train_ds,
            rng=rng,
            cfg=cfg,
            tokenizer=tokenizer,
            debug_num_problems=flags.debug_train_examples,
        )
    elif isinstance(cfg, training.DPOTrainingConfig):
        logger.info("Using DPO trainer")
        make_trainer_fn = training.make_dpo_trainer
        example_list, train_ds = training.make_dpo_dataset(
            train_dataset=train_ds,
            rng=rng,
            cfg=cfg,
            tokenizer=tokenizer,
            debug_num_problems=flags.debug_train_examples,
        )
    elif isinstance(cfg, training.HFTrainerConfig):
        if cfg.model.model_type in {
            "causal-lm",
        }:
            example_list, train_ds = training.make_clm_dataset(
                train_dataset=train_ds,
                rng=rng,
                cfg=cfg,
                tokenizer=tokenizer,
                debug_num_problems=flags.debug_train_examples,
            )
        else:
            example_list, train_ds = training.make_cls_dataset(
                train_dataset=train_ds,
                rng=rng,
                cfg=cfg,
                tokenizer=tokenizer,
                debug_num_problems=flags.debug_train_examples,
            )

        make_trainer_fn = training.make_hf_trainer
    else:
        raise ValueError(f"Invalid trainer: {type(cfg)}")
    preprocessor = Preprocessor(cfg.preprocessing)
    scoring_fn = scoring.load_scoring_method(
        cfg.scoring,
        tokenizer=tokenizer,
        fail_choice_str=cfg.preprocessing.fail_choice_str,
        pass_choice_str=cfg.preprocessing.pass_choice_str,
        eval_completion=preprocessor.render_eval_completion(),
        max_length=tokenizer.model_max_length,
        calculate_loss=True,
        num_workers=cfg.num_workers,
    )
    preprocessed_val_ds = scoring_fn.preprocess_dataset(validation_ds)

    train_example_file = out_dir / "train_examples.csv"
    examples_by_source, example_list = example_list
    logger.info("Examples by Source:")
    for k, v in sorted(examples_by_source.items(), key=lambda x: x[0]):
        logger.info(f"\t{k}: {v:,}")
    logger.info(f"Writing training examples to {train_example_file}")
    with open(train_example_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        if isinstance(
            cfg, (training.RewardTrainingConfig, training.DPOTrainingConfig)
        ):
            columns = ["chosen", "rejected"]
        else:
            columns = ["chosen", "passed"]
        writer.writerow(columns)
        for r in example_list:
            writer.writerow(r)

    if flags.debug_train_examples:
        logger.warning(
            f"Using only {flags.debug_train_examples} training examples"
        )
        train_ds = train_ds.select(range(flags.debug_train_examples))
    if flags.debug_val_examples:
        logger.warning(
            f"Using only {flags.debug_val_examples} validation examples"
        )
        validation_ds = validation_ds.select(range(flags.debug_val_examples))
        preprocessed_val_ds = preprocessed_val_ds.select(
            range(flags.debug_val_examples)
        )

    current_hash = utils.get_current_repo_hash()
    logger.info(f"Git hash: {current_hash}")

    logger.info("Starting training")
    trainer = make_trainer_fn(
        cfg=cfg,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        tokenizer=tokenizer,
        preprocessed_eval_dataset=preprocessed_val_ds,
        scoring_fn=scoring_fn,
        sort_by_length=flags.sort_by_length,
    )

    trainer.train(flags.resume)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        if wandb_run:
            run_id = wandb_run.id
        else:
            run_id = None
        json.dump(
            {
                "hash": current_hash,
                "run_id": run_id,
                "step_results": trainer.step_results,
                "examples_by_source": examples_by_source,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    if flags.enable_wandb:
        artifact = wandb.Artifact(
            f"{cfg.model.name}-{run_name}",
            type="train_output",
        )
        artifact.add_file(out_dir / "results.json")
        artifact.add_file(train_example_file)
        artifact.add_file(out_dir / "config.yaml")
        wandb_run.finish()
    logger.info("Training complete")


if __name__ == "__main__":
    main(*parse_args())
    exit()
