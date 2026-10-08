from pathlib import Path

import pytest

from ds340w_repro.build_matrix import (
    build_baseline_config,
    build_evaluation_calibration_config,
    build_pilot_config,
    build_rank_matrix,
    build_smoke_config,
    build_training_calibration_config,
)
from ds340w_repro.config_schema import ResolvedConfig, validate_nonfinal_config
from ds340w_repro.run_opendelta import resolve_runner_payload


MODEL_PATH = Path("/verified/model/snapshot")


def test_baseline_is_the_approved_fixed_paper_grid_cell() -> None:
    config = build_baseline_config(
        model_path=MODEL_PATH,
        output_root=Path("/ignored/artifacts"),
        physical_batch_size=2,
        evaluation_batch_size=4,
    )

    assert config.experiment_id == "baseline-r8-lr1e-4-eb16"
    assert (
        config.values["learning_rate"],
        config.values["effective_batch_size"],
        config.values["lora_r"],
        config.values["lora_alpha"],
    ) == (1e-4, 16, 8, 16)
    assert config.values["effective_model_dropout"] == 0.0
    assert config.values["source_lock_sha256"] == (
        "4aae9bca3d409d7b075f6440bf4f7b3ce3023f8fbd24391ec1fa514148cb6b0d"
    )
    assert config.values["test_indices_sha256"] == (
        "83d6bdad006418cd841167d938f835561dc13b9ccd464ad21ebe7dabeb0b2394"
    )
    assert config.values["course_data_manifest_sha256"] == (
        "400143658894e4460f67d8e96dc2e69225150d1ed9ec15ec3da3cef0f37bff66"
    )
    assert (
        config.values["per_device_train_batch_size"]
        * config.values["gradient_accumulation_steps"]
        == config.values["effective_batch_size"]
    )
    validate_nonfinal_config(config)


def test_rank_matrix_uses_alpha_twice_rank_and_reuses_rank_eight() -> None:
    matrix = build_rank_matrix(
        model_path=MODEL_PATH,
        output_root=Path("/ignored/artifacts"),
        physical_batch_size=2,
        evaluation_batch_size=4,
        selected_learning_rate=1e-4,
        selected_effective_batch_size=16,
        rank_eight_checkpoint=Path("/ignored/artifacts/best-r8"),
    )

    assert [(row.values["lora_r"], row.values["lora_alpha"]) for row in matrix] == [
        (2, 4),
        (4, 8),
        (8, 16),
        (16, 32),
    ]
    assert matrix[2].values["reuse_checkpoint"] == "/ignored/artifacts/best-r8"
    assert all(row.values["learning_rate"] == 1e-4 for row in matrix)

    with pytest.raises(ValueError, match="fixed baseline"):
        build_rank_matrix(
            MODEL_PATH, Path("runs"), 2, 4, 5e-4, 16, Path("rank8")
        )


def test_normalization_and_checksum_are_stable() -> None:
    first = build_baseline_config(MODEL_PATH, Path("/ignored/artifacts"), 2, 4)
    second = ResolvedConfig.from_json(first.to_json())

    assert first.to_json() == second.to_json()
    assert first.sha256 == second.sha256
    assert set(first.authority_by_field) == set(first.values)


def test_nonfinal_validation_rejects_unseen_or_unsafe_numeric_modes() -> None:
    base = build_baseline_config(MODEL_PATH, Path("/ignored/artifacts"), 2, 4)

    for key, value in (
        ("do_test", True),
        ("fp16", True),
        ("bf16", True),
        ("tf32", True),
        ("private_unseen_path", "/private/indices.json"),
    ):
        changed = base.with_value(key, value, authority="forbidden test fixture")
        with pytest.raises(ValueError):
            validate_nonfinal_config(changed)


def test_every_nonfinal_runner_config_forces_preprocessing_cache_overwrite() -> None:
    configs = [
        build_baseline_config(MODEL_PATH, Path("runs"), 2, 4),
        *build_rank_matrix(
            MODEL_PATH,
            Path("runs"),
            2,
            4,
            1e-4,
            16,
            Path("rank8"),
        ),
        build_smoke_config(MODEL_PATH, Path("smoke")),
        build_evaluation_calibration_config(MODEL_PATH, Path("eval"), 4),
        build_training_calibration_config(MODEL_PATH, Path("train"), 2, 4),
        build_pilot_config(MODEL_PATH, Path("pilot"), 2, 4),
    ]

    assert all(config.values["overwrite_cache"] is True for config in configs)
    assert all(
        config.runner_payload()["overwrite_cache"] is True for config in configs
    )
    assert all(
        config.authority_by_field["overwrite_cache"]
        == "permitted resource/configuration accommodation"
        for config in configs
    )


def test_nonfinal_validation_rejects_reusable_preprocessing_cache() -> None:
    base = build_baseline_config(MODEL_PATH, Path("/ignored/artifacts"), 2, 4)

    changed = base.with_value(
        "overwrite_cache", False, authority="forbidden test fixture"
    )

    with pytest.raises(ValueError, match="overwrite_cache"):
        validate_nonfinal_config(changed)

    values = dict(base.values)
    authorities = dict(base.authority_by_field)
    values.pop("overwrite_cache")
    authorities.pop("overwrite_cache")
    missing = ResolvedConfig(base.experiment_id, values, authorities)

    with pytest.raises(ValueError, match="overwrite_cache"):
        validate_nonfinal_config(missing)


def test_relative_runner_paths_are_resolved_from_the_project_root(
    tmp_path: Path,
) -> None:
    config = build_baseline_config(
        Path("cached/model"), Path("artifacts/runs"), 2, 4
    )

    payload = resolve_runner_payload(config, tmp_path)

    assert payload["model_name_or_path"] == str(
        (tmp_path / "cached/model").resolve()
    )
    assert payload["tokenizer_name"] == str(
        (tmp_path / "cached/model").resolve()
    )
    assert payload["output_dir"] == str(
        (tmp_path / "artifacts/runs" / config.experiment_id).resolve()
    )
