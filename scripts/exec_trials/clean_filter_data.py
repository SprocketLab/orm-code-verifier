"""Code to clean up the raw filter results into something usable."""

import pathlib
import click
import itertools
import pandas as pd
import ujson
import numpy as np
from collections import defaultdict
import gzip
import asyncio
from tqdm import tqdm

MODELS = {
    "qc-inst-7b",
    "qc-inst-14b",
    "qc-inst-1_5b",
    "qc-inst-3b",
    "qc-inst-500m",
}
SAMPLING_SETUPS = {
    "t1.0_n128",
    "t1.0_n256",
    "t0.2_n128",
    "t0.4_n128",
    "t0.6_n128",
    "t0.8_n128",
}

DATASETS = {
    "gsm8k",
    "code_contests",
    "humaneval",
    "mbpp",
}

NUM_TESTS = {1, 3, 10}


# Helper function to load JSON synchronously for use with asyncio.to_thread
def _load_json_sync(file_path_str):
    with open(file_path_str, "r") as f:
        return ujson.load(f)


async def process_single_instance(
    filter_dir_path,
    save_name_val,
    model_val,
    sampling_setup_val,
    dataset_val,
    filter_type_val,
):
    """
    Processes a single filter instance asynchronously.
    Reads data, calculates metrics, and returns records and timing info.
    """
    # For clarity within the function
    model = model_val
    sampling_setup = sampling_setup_val
    dataset = dataset_val
    save_name = save_name_val

    identifier = (
        f"[{model}/{sampling_setup}/{dataset}/{save_name}/{filter_type_val}]"
    )

    if not filter_dir_path.exists():
        print(f"{identifier} Filter directory {filter_dir_path} does not exist")
        return [], None
    if not (filter_dir_path / "overall.json").exists():
        print(
            f"{identifier} Filter directory {filter_dir_path} does not have overall.json"
        )
        return [], None
    if not (filter_dir_path / "results.parquet").exists():
        print(
            f"{identifier} Filter directory {filter_dir_path} does not have results.parquet"
        )
        return [], None

    try:
        overall_results = await asyncio.to_thread(
            _load_json_sync, filter_dir_path / "overall.json"
        )
        # Assuming pd.read_parquet is thread-safe or releases GIL for I/O
        df = await asyncio.to_thread(
            pd.read_parquet, filter_dir_path / "results.parquet"
        )
    except Exception as e:
        print(f"{identifier} Error processing {filter_dir_path}: {e}")
        return [], None

    instance_records = []
    sid_results = (
        df.groupby(["task_id", "sid"])[["passed", "timeout"]]
        .max()
        .reset_index()
    )
    for _, row in sid_results.iterrows():
        instance_records.append(
            {
                "task_id": row["task_id"],
                "sid": row["sid"],
                "generator": model,
                "sampling_setup": sampling_setup,
                "dataset": dataset,
                "filter_name": save_name,
                "passed": row["passed"],
                "timeout": row["timeout"],
            }
        )

    if filter_type_val == "exec" and dataset == "code_contests":
        time_values = [
            t["execution_time"]
            + t["postprocessing_time"]
            + t["preprocessing_time"]
            for t in overall_results.values()
            if isinstance(t, dict)
            and all(
                k in t
                for k in [
                    "execution_time",
                    "postprocessing_time",
                    "preprocessing_time",
                ]
            )
        ]
    else:
        time_values = [
            t["net_time"]
            for t in overall_results.values()
            if isinstance(t, dict) and "net_time" in t
        ]

    if not time_values:
        print(
            f"{identifier} Warning: No time values found for {filter_dir_path}. Times: {overall_results.values()}"
        )
        overall_time = np.nan  # Use NaN if no time data
    else:
        overall_time = np.mean(time_values)

    instance_timing_info = {
        "generator": model,
        "sampling_setup": sampling_setup,
        "dataset": dataset,
        "filter_name": save_name,
        "time": overall_time,
    }

    print(f"{identifier} Overall time: {overall_time}")
    print(
        f"{identifier} Filter directory {filter_dir_path} has {len(df)} rows, processed into {len(instance_records)} records."
    )

    return instance_records, instance_timing_info


async def _process_all_filters_async(
    filter_data_path: pathlib.Path,
    filters_config: list,
    models_config: set,
    sampling_setups_config: set,
    datasets_config: set,
):
    """
    Asynchronously processes all filter combinations.
    """
    all_tasks = []
    for filter_type, *other_args in filters_config:
        if filter_type == "exec":
            exec_filter_path = filter_data_path / "exec_filter"
            name_func = (
                lambda ds, gm, ss: exec_filter_path
                / f"{ds}_{gm}"
                / f"{ss}.t={other_args[0]}"
            )
            current_save_name = f"e3s{other_args[0]}t"
        else:
            name_func = (
                lambda ds, gm, ss: filter_data_path
                / filter_type
                / f"{ds}_{gm}"
                / f"{ss}.n=16"
            )
            current_save_name = (
                "is_valid_syntax"
                if filter_type == "syntax"
                else "has_no_lint_errors"
            )

        for model, sampling_setup, dataset in itertools.product(
            models_config, sampling_setups_config, datasets_config
        ):
            filter_dir = name_func(dataset, model, sampling_setup)
            task = asyncio.create_task(
                process_single_instance(
                    filter_dir,
                    current_save_name,
                    model,
                    sampling_setup,
                    dataset,
                    filter_type,
                )
            )
            all_tasks.append(task)

    # Use tqdm_asyncio.gather as per user import
    print(f"Gathering results for {len(all_tasks)} tasks...")
    return await asyncio.gather(
        *all_tasks,
        return_exceptions=True,
    )


@click.command()
@click.argument(
    "filter_data_path",
    type=click.Path(
        exists=True, path_type=pathlib.Path, dir_okay=True, file_okay=False
    ),
)
@click.argument(
    "output_path",
    type=click.Path(path_type=pathlib.Path, dir_okay=True, file_okay=False),
)
def clean_filter_data(
    filter_data_path: pathlib.Path, output_path: pathlib.Path
):
    """Clean the filter data by removing duplicate rows and saving the cleaned data."""

    filters_config = [("exec", n_t) for n_t in NUM_TESTS]
    filters_config.extend([("pylint",), ("syntax",)])

    # Run the asynchronous processing part
    gathered_results = asyncio.run(
        _process_all_filters_async(
            filter_data_path, filters_config, MODELS, SAMPLING_SETUPS, DATASETS
        )
    )

    records = []
    filter_timings = []

    for result_records, result_timing_info in gathered_results:
        if result_records:
            records.extend(result_records)
        if result_timing_info:
            filter_timings.append(result_timing_info)

    if not records:
        print("No records were processed. Exiting.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    records_df = pd.DataFrame(records)

    if filter_timings:
        filter_timings_df = pd.DataFrame(filter_timings)
        filter_timings_df.to_csv(
            output_path / "filter_timings_summary.csv", index=False
        )
        print(
            f"Saved filter timings summary to {output_path / 'filter_timings_summary.csv'}"
        )

    grouped = records_df.groupby(["generator", "sampling_setup", "dataset"])
    for (generator, sampling_setup, dataset), gdf in tqdm(
        grouped,
        total=len(grouped),
        desc="Processing filter data",
    ):
        new_records = []
        sub_out_dir = output_path / f"{generator}.{sampling_setup}"
        sub_out_dir.mkdir(parents=True, exist_ok=True)
        for (task_id, sid), pdf in gdf.groupby(["task_id", "sid"]):
            pdf.set_index("filter_name", inplace=True)
            new_record = {
                "generator": generator,
                "sampling_setup": sampling_setup,
                "dataset": dataset,
                "task_id": task_id,
                "sid": sid,
            }
            for filter_name in pdf.index:
                new_record[filter_name] = pdf.loc[filter_name, "passed"]
                new_record[f"{filter_name}_timeout"] = pdf.loc[
                    filter_name, "timeout"
                ]
            new_records.append(new_record)
        new_records = pd.DataFrame(new_records)
        new_records.to_csv(
            sub_out_dir / f"{dataset}.csv.gz",
            index=False,
            compression="gzip",
        )


if __name__ == "__main__":
    clean_filter_data()
