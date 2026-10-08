"""Build the predeclared OpenDelta SST-2 experiment matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ds340w_repro.config_schema import (
    COURSE_DATA_MANIFEST_SHA256,
    DATASET_LOADER_REPOSITORY,
    DATASET_LOADER_REVISION,
    LORA_MODIFIED_MODULES,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    ResolvedConfig,
    SOURCE_LOCK_SHA256,
    TEST_INDICES_SHA256,
    UNFROZEN_MODULES,
    validate_nonfinal_config,
)


PAPER = "paper/supplement"
UPSTREAM = "pinned-code default"
ACCOMMODATION = "permitted resource/configuration accommodation"
SST2_ARCHIVE_SHA256 = (
    "d67e16fb55739c1b32cdce9877596db1c127dc322d93c082281f64057c16deaa"
)


def _base_values(
    *,
    model_path: Path,
    output_dir: Path,
    physical_batch_size: int,
    evaluation_batch_size: int,
    effective_batch_size: int,
    learning_rate: float,
    rank: int,
    max_steps: int = 20_000,
    eval_steps: int = 100,
    save_steps: int = 100,
    max_train_samples: int | None = None,
    max_val_samples: int = 436,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, str]]:
    if effective_batch_size % physical_batch_size:
        raise ValueError("Effective batch must be divisible by physical batch")

    values: dict[str, Any] = {
        "model_name_or_path": str(model_path),
        "tokenizer_name": str(model_path),
        "model_revision": MODEL_REVISION,
        "model_source_repository": MODEL_REPOSITORY,
        "model_source_revision": MODEL_REVISION,
        "dataset_loader_repository": DATASET_LOADER_REPOSITORY,
        "dataset_loader_revision": DATASET_LOADER_REVISION,
        "dataset_archive_sha256": SST2_ARCHIVE_SHA256,
        "course_data_manifest_sha256": COURSE_DATA_MANIFEST_SHA256,
        "source_lock_sha256": SOURCE_LOCK_SHA256,
        "test_indices_sha256": TEST_INDICES_SHA256,
        "task_name": "sst2",
        "dataset_config_name": "en",
        "eval_dataset_name": "sst2",
        "eval_dataset_config_name": "en",
        "test_dataset_name": "sst2",
        "test_dataset_config_name": "en",
        "do_train": True,
        "do_eval": True,
        "do_test": False,
        "split_validation_test": False,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "max_test_samples": None,
        "max_source_length": 128,
        "max_target_length": 128,
        "val_max_target_length": 128,
        "test_max_target_length": 128,
        "pad_to_max_length": False,
        "ignore_pad_token_for_loss": True,
        "preprocessing_num_workers": None,
        "data_sample_seed": 42,
        "num_beams": 1,
        "output_dir": str(output_dir),
        "overwrite_cache": True,
        "overwrite_output_dir": True,
        "max_steps": max_steps,
        "num_train_epochs": 3.0,
        "per_device_train_batch_size": physical_batch_size,
        "per_device_eval_batch_size": evaluation_batch_size,
        "effective_batch_size": effective_batch_size,
        "gradient_accumulation_steps": effective_batch_size
        // physical_batch_size,
        "learning_rate": learning_rate,
        "optim": "adamw_hf",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "logging_steps": min(10, max_steps),
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "average_metrics",
        "greater_is_better": True,
        "predict_with_generate": True,
        "generation_max_length": 128,
        "generation_num_beams": 1,
        "seed": seed,
        "data_seed": seed,
        "full_determinism": True,
        "effective_model_dropout": 0.0,
        "fp16": False,
        "bf16": False,
        "tf32": False,
        "dataloader_num_workers": 0,
        "report_to": [],
        "push_to_hub": False,
        "compute_time": True,
        "compute_memory": True,
        "delta_type": "lora",
        "lora_r": rank,
        "lora_alpha": 2 * rank,
        "lora_dropout": 0.0,
        "modified_modules": list(LORA_MODIFIED_MODULES),
        "unfrozen_modules": list(UNFROZEN_MODULES),
        "common_structure": True,
    }

    paper_fields = {
        "task_name",
        "eval_dataset_name",
        "learning_rate",
        "effective_batch_size",
        "lora_r",
        "lora_alpha",
        "max_steps",
    }
    accommodation_fields = {
        "model_name_or_path",
        "tokenizer_name",
        "model_revision",
        "model_source_repository",
        "model_source_revision",
        "dataset_loader_repository",
        "dataset_loader_revision",
        "dataset_archive_sha256",
        "course_data_manifest_sha256",
        "source_lock_sha256",
        "test_indices_sha256",
        "do_test",
        "split_validation_test",
        "max_train_samples",
        "max_val_samples",
        "max_test_samples",
        "output_dir",
        "overwrite_cache",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "full_determinism",
        "fp16",
        "bf16",
        "tf32",
        "dataloader_num_workers",
        "report_to",
        "push_to_hub",
        "modified_modules",
        "unfrozen_modules",
        "common_structure",
    }
    authorities = {
        key: (
            "approved confirmation design"
            if key in {"seed", "data_seed"} and seed != 42
            else PAPER
            if key in paper_fields
            else ACCOMMODATION
            if key in accommodation_fields
            else UPSTREAM
        )
        for key in values
    }
    return values, authorities


def _resolved(experiment_id: str, **kwargs: Any) -> ResolvedConfig:
    values, authorities = _base_values(**kwargs)
    config = ResolvedConfig(experiment_id, values, authorities)
    validate_nonfinal_config(config)
    return config


def build_baseline_config(
    model_path: Path,
    output_root: Path,
    physical_batch_size: int,
    evaluation_batch_size: int,
) -> ResolvedConfig:
    """Return the approved first paper-grid cell at rank 8, LR 1e-4, batch 16."""

    experiment_id = "baseline-r8-lr1e-4-eb16"
    return _resolved(
        experiment_id,
        model_path=model_path,
        output_dir=output_root / experiment_id,
        physical_batch_size=physical_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        effective_batch_size=16,
        learning_rate=1e-4,
        rank=8,
    )


def build_rank_matrix(
    model_path: Path,
    output_root: Path,
    physical_batch_size: int,
    evaluation_batch_size: int,
    selected_learning_rate: float,
    selected_effective_batch_size: int,
    rank_eight_checkpoint: Path,
) -> list[ResolvedConfig]:
    if selected_learning_rate != 1e-4 or selected_effective_batch_size != 16:
        raise ValueError("Rank sweep must use the approved fixed baseline setting")
    matrix = []
    for rank in (2, 4, 8, 16):
        experiment_id = f"rank-r{rank}"
        config = _resolved(
            experiment_id,
            model_path=model_path,
            output_dir=output_root / experiment_id,
            physical_batch_size=physical_batch_size,
            evaluation_batch_size=evaluation_batch_size,
            effective_batch_size=selected_effective_batch_size,
            learning_rate=selected_learning_rate,
            rank=rank,
        )
        if rank == 8:
            config = config.with_value(
                "reuse_checkpoint", str(rank_eight_checkpoint), PAPER
            )
        matrix.append(config)
    return matrix


def build_confirmation_matrix(
    model_path: Path,
    output_root: Path,
    physical_batch_size: int,
    evaluation_batch_size: int,
    selected_rank: int,
) -> list[ResolvedConfig]:
    """Repeat each distinct selected setting with the two reserved seeds."""

    if selected_rank not in (2, 4, 8, 16):
        raise ValueError("Selected rank must be one of the predeclared rank values")
    configs = []
    for rank in dict.fromkeys((8, selected_rank)):
        for seed in (17, 23):
            experiment_id = f"confirmation-r{rank}-seed{seed}"
            config = _resolved(
                experiment_id,
                model_path=model_path,
                output_dir=output_root / experiment_id,
                physical_batch_size=physical_batch_size,
                evaluation_batch_size=evaluation_batch_size,
                effective_batch_size=16,
                learning_rate=1e-4,
                rank=rank,
                seed=seed,
            )
            configs.append(config)
    return configs


def build_smoke_config(
    model_path: Path,
    output_dir: Path,
    evaluation_batch_size: int = 1,
) -> ResolvedConfig:
    """Return a one-step, one-example non-result viability configuration."""

    return _resolved(
        "week1-smoke-r8",
        model_path=model_path,
        output_dir=output_dir,
        physical_batch_size=1,
        evaluation_batch_size=evaluation_batch_size,
        effective_batch_size=1,
        learning_rate=1e-3,
        rank=8,
        max_steps=1,
        eval_steps=1,
        save_steps=1,
        max_train_samples=1,
        max_val_samples=1,
    ).with_value("max_val_samples", 1, ACCOMMODATION)


def _with_accommodations(
    config: ResolvedConfig, values: dict[str, Any]
) -> ResolvedConfig:
    for key, value in values.items():
        config = config.with_value(key, value, ACCOMMODATION)
    validate_nonfinal_config(config)
    return config


def build_evaluation_calibration_config(
    model_path: Path,
    output_dir: Path,
    evaluation_batch_size: int,
) -> ResolvedConfig:
    """Build one full-Test, evaluation-only preflight configuration."""

    config = _resolved(
        f"preflight-eval-eb{evaluation_batch_size}",
        model_path=model_path,
        output_dir=output_dir,
        physical_batch_size=1,
        evaluation_batch_size=evaluation_batch_size,
        effective_batch_size=1,
        learning_rate=1e-3,
        rank=16,
        max_steps=1,
        eval_steps=1,
        save_steps=1,
        max_train_samples=1,
        max_val_samples=436,
    )
    return _with_accommodations(
        config,
        {
            "do_train": False,
            "evaluation_strategy": "no",
            "save_strategy": "no",
            "load_best_model_at_end": False,
        },
    )


def build_training_calibration_config(
    model_path: Path,
    output_dir: Path,
    physical_batch_size: int,
    evaluation_batch_size: int,
) -> ResolvedConfig:
    """Build one ten-step worst-case LoRA training calibration."""

    config = _resolved(
        f"preflight-train-pb{physical_batch_size}-eb{evaluation_batch_size}",
        model_path=model_path,
        output_dir=output_dir,
        physical_batch_size=physical_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        effective_batch_size=32,
        learning_rate=1e-3,
        rank=16,
        max_steps=10,
        eval_steps=10,
        save_steps=10,
        max_val_samples=436,
    )
    return _with_accommodations(
        config,
        {
            "do_eval": False,
            "evaluation_strategy": "no",
            "save_strategy": "no",
            "load_best_model_at_end": False,
        },
    )


def build_pilot_config(
    model_path: Path,
    output_dir: Path,
    physical_batch_size: int,
    evaluation_batch_size: int,
) -> ResolvedConfig:
    """Build the discarded 200-step rank-16 timing pilot."""

    return _resolved(
        f"preflight-pilot-r16-pb{physical_batch_size}-eb{evaluation_batch_size}",
        model_path=model_path,
        output_dir=output_dir,
        physical_batch_size=physical_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        effective_batch_size=32,
        learning_rate=1e-3,
        rank=16,
        max_steps=200,
        eval_steps=100,
        save_steps=100,
        max_val_samples=436,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--model-path", type=Path, default=Path("/verified/model"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--physical-batch-size", type=int, default=1)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    arguments = parser.parse_args()

    config = build_baseline_config(
        arguments.model_path,
        arguments.output_root,
        arguments.physical_batch_size,
        arguments.evaluation_batch_size,
    )
    if arguments.verify:
        validate_nonfinal_config(config)
    print(
        json.dumps(
            {"experiment_id": config.experiment_id, "sha256": config.sha256},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
