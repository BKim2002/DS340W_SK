#!/usr/bin/env python3
"""Validate a completed Week 1 smoke run and emit sanitized evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable
import warnings

from ds340w_repro.config_schema import ResolvedConfig
from ds340w_repro.run_opendelta import resolve_runner_payload


EXPECTED_UPSTREAM_COMMIT = "20e426f716ea53618657cd34bbfc4b347b98cd74"
EXPECTED_UPSTREAM_URL = "https://github.com/thunlp/OpenDelta.git"
EXPECTED_RUNNER = "examples/legacies/examples_seq2seq/run_seq2seq.py"
EXPECTED_MODEL_REPOSITORY = "google/t5-base-lm-adapt"
EXPECTED_MODEL_REVISION = "bf48a6b7de15c6afe1e27e1a51aa1ea4626f96bf"
EXPECTED_DATASET_REPOSITORY = "nyu-mll/glue"
EXPECTED_DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
EXPECTED_DATASET_LOADER_REPOSITORY = "https://github.com/huggingface/datasets.git"
EXPECTED_DATASET_LOADER_REVISION = "dff6c92c80fbfac456349a531c0a4521c2b0c8f8"
EXPECTED_MODIFIED_MODULES = [
    "attn@.q@",
    "attn@.v@",
    "crossattn@.q@",
    "crossattn@.v@",
]
EXPECTED_UNFROZEN_MODULES = ["deltas", "layer_norm@"]
EXPECTED_COUNTS = {
    "injected_projections": 72,
    "lora_parameters": 884_736,
    "layer_norm_parameters": 47_616,
    "total_trainable_parameters": 932_352,
}
ALLOWED_PATCH_CLASSIFICATIONS = {
    "compatibility",
    "configuration",
    "data-handling",
    "provenance",
    "observation-only",
}
EVIDENCE_FILES = (
    "resolved-config.json",
    "runner-config.json",
    "execution-tree.json",
    "process-result.json",
    "train_results.json",
    "eval_results.json",
    "performance_results.json",
    "all_results.json",
    "trainer_state.json",
    "checkpoint-1/trainer_state.json",
    "checkpoint-1/optimizer.pt",
    "checkpoint-1/scheduler.pt",
    "checkpoint-1/pytorch_model.bin",
    "pytorch_model.bin",
    "delta/config.json",
    "delta/pytorch_model.bin",
    "reload-verification.json",
)
ARTIFACT_ROLES = {
    "resolved-config.json": "resolved-config",
    "runner-config.json": "original-runner-config",
    "execution-tree.json": "source-provenance",
    "process-result.json": "process-completion",
    "train_results.json": "training-metrics",
    "eval_results.json": "generated-evaluation-metrics",
    "performance_results.json": "runtime-memory-metrics",
    "all_results.json": "combined-metrics",
    "trainer_state.json": "final-trainer-state",
    "checkpoint-1/trainer_state.json": "step-one-trainer-state",
    "checkpoint-1/optimizer.pt": "step-one-optimizer-state",
    "checkpoint-1/scheduler.pt": "step-one-scheduler-state",
    "checkpoint-1/pytorch_model.bin": "step-one-trainable-weights",
    "pytorch_model.bin": "selected-trainable-weights",
    "delta/config.json": "native-delta-config",
    "delta/pytorch_model.bin": "native-delta-weights",
    "reload-verification.json": "native-delta-reload-check",
}


class SmokeEvidenceError(ValueError):
    """Raised when a claimed smoke invariant is absent or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeEvidenceError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing required evidence file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SmokeEvidenceError(f"Invalid JSON evidence: {path.name}") from error
    _require(isinstance(value, dict), f"Expected a JSON object: {path.name}")
    return value


def _finite_number(payload: dict[str, Any], key: str, *, minimum: float | None = None) -> float:
    value = payload.get(key)
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"Missing numeric metric {key}",
    )
    number = float(value)
    _require(math.isfinite(number), f"Metric {key} is not finite")
    if minimum is not None:
        _require(number >= minimum, f"Metric {key} must be at least {minimum}")
    return number


def _keys_containing(value: Any, needle: str, prefix: str = "") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            location = f"{prefix}.{key}" if prefix else str(key)
            if needle in str(key).lower():
                matches.append(location)
            matches.extend(_keys_containing(nested, needle, location))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            matches.extend(_keys_containing(nested, needle, f"{prefix}[{index}]"))
    return matches


def _artifact_record(artifact_root: Path, relative_path: str) -> dict[str, Any]:
    path = artifact_root / relative_path
    _require(path.is_file(), f"Missing required artifact: {relative_path}")
    return {
        "relative_path": relative_path,
        "role": ARTIFACT_ROLES[relative_path],
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_checkpoint_counts(path: Path) -> tuple[dict[str, int], set[str]]:
    try:
        import torch
    except ImportError as error:
        raise SmokeEvidenceError(
            "PyTorch is required to verify checkpoint tensor evidence"
        ) from error

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="TypedStorage is deprecated.*",
                category=UserWarning,
            )
            state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise SmokeEvidenceError(
            "Locked PyTorch must support safe weights_only checkpoint loading"
        ) from error
    _require(isinstance(state, dict), f"Checkpoint is not a state dictionary: {path.name}")
    _require(state, f"Checkpoint state dictionary is empty: {path.name}")

    projection_sides: dict[str, set[str]] = {}
    lora_parameters = 0
    layer_norm_parameters = 0
    keys: set[str] = set()
    unexpected: list[str] = []
    for key, tensor in state.items():
        _require(isinstance(key, str), "Checkpoint contains a non-string key")
        _require(hasattr(tensor, "numel"), f"Checkpoint value is not a tensor: {key}")
        keys.add(key)
        if ".lora.lora_" in key:
            prefix, side = key.rsplit(".lora.lora_", 1)
            _require(side in {"A", "B"}, f"Unexpected LoRA tensor suffix: {key}")
            projection_sides.setdefault(prefix, set()).add(side)
            lora_parameters += int(tensor.numel())
        elif "layer_norm" in key:
            layer_norm_parameters += int(tensor.numel())
        else:
            unexpected.append(key)
    _require(not unexpected, "Checkpoint contains tensors outside LoRA/LayerNorm scope")
    _require(
        all(sides == {"A", "B"} for sides in projection_sides.values()),
        "A LoRA projection is missing its A or B tensor",
    )
    counts = {
        "injected_projections": len(projection_sides),
        "lora_parameters": lora_parameters,
        "layer_norm_parameters": layer_norm_parameters,
        "total_trainable_parameters": lora_parameters + layer_norm_parameters,
    }
    return counts, keys


def _verify_provenance(project_root: Path, artifact_root: Path) -> dict[str, Any]:
    upstream = _load_json(project_root / "data/manifests/upstream-source.json")
    source_lock = _load_json(project_root / "data/manifests/source-lock.json")
    series = _load_json(project_root / "patches/series.json")
    execution = _load_json(artifact_root / "execution-tree.json")

    _require(upstream.get("remote_url") == EXPECTED_UPSTREAM_URL, "Unexpected upstream URL")
    _require(upstream.get("commit") == EXPECTED_UPSTREAM_COMMIT, "Unexpected upstream commit")
    _require(upstream.get("runner_relative_path") == EXPECTED_RUNNER, "Unexpected runner path")
    pristine_runner = project_root / "vendor/OpenDelta" / EXPECTED_RUNNER
    _require(pristine_runner.is_file(), "Pinned pristine runner is unavailable")
    pre_hash = _sha256(pristine_runner)
    _require(upstream.get("runner_sha256") == pre_hash, "Pristine runner hash mismatch")

    _require(series.get("upstream_commit") == EXPECTED_UPSTREAM_COMMIT, "Patch series commit mismatch")
    patch_records: list[dict[str, str]] = []
    patch_hashes: list[str] = []
    for entry in series.get("patches", []):
        _require(isinstance(entry, dict), "Patch series contains an invalid entry")
        classification = entry.get("classification")
        _require(
            classification in ALLOWED_PATCH_CLASSIFICATIONS,
            f"Disallowed patch classification: {classification}",
        )
        relative_path = entry.get("path")
        _require(isinstance(relative_path, str), "Patch path is missing")
        patch_path = (project_root / relative_path).resolve(strict=True)
        _require(
            patch_path.is_relative_to(project_root),
            "Patch path escapes the project root",
        )
        patch_hash = _sha256(patch_path)
        _require(entry.get("sha256") == patch_hash, f"Patch hash mismatch: {relative_path}")
        patch_hashes.append(patch_hash)
        patch_records.append(
            {
                "classification": str(classification),
                "relative_path": relative_path.replace("\\", "/"),
                "sha256": patch_hash,
            }
        )
    _require(patch_records, "Patch series is empty")

    _require(execution.get("upstream_commit") == EXPECTED_UPSTREAM_COMMIT, "Execution commit mismatch")
    _require(execution.get("pre_patch_runner_sha256") == pre_hash, "Execution pre-patch hash mismatch")
    _require(execution.get("applied_patch_sha256") == patch_hashes, "Applied patch order/hash mismatch")
    runtime_root = Path(str(execution.get("root", ""))).resolve(strict=True)
    runtime_runner = Path(str(execution.get("runner", ""))).resolve(strict=True)
    _require(runtime_runner == runtime_root / EXPECTED_RUNNER, "Execution runner path mismatch")
    post_hash = _sha256(runtime_runner)
    _require(execution.get("post_patch_runner_sha256") == post_hash, "Post-patch runner hash mismatch")

    opendelta = source_lock.get("opendelta", {})
    _require(opendelta.get("resolved_revision") == EXPECTED_UPSTREAM_COMMIT, "Source lock OpenDelta revision mismatch")
    return {
        "repository_url": EXPECTED_UPSTREAM_URL,
        "commit": EXPECTED_UPSTREAM_COMMIT,
        "runner_relative_path": EXPECTED_RUNNER,
        "pre_patch_runner_sha256": pre_hash,
        "post_patch_runner_sha256": post_hash,
        "patches": patch_records,
    }


def _verify_inputs(source_lock: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    model = source_lock.get("model", {})
    tokenizer = source_lock.get("tokenizer", {})
    dataset = source_lock.get("dataset", {})
    mirror = dataset.get("provenance_mirror", {})
    raw_archive = dataset.get("raw_archive", {})
    checks = (
        (model.get("repo_id"), EXPECTED_MODEL_REPOSITORY, "model repository"),
        (model.get("requested_revision"), EXPECTED_MODEL_REVISION, "requested model revision"),
        (model.get("resolved_revision"), EXPECTED_MODEL_REVISION, "resolved model revision"),
        (tokenizer.get("repo_id"), EXPECTED_MODEL_REPOSITORY, "tokenizer repository"),
        (tokenizer.get("resolved_revision"), EXPECTED_MODEL_REVISION, "tokenizer revision"),
        (mirror.get("repo_id"), EXPECTED_DATASET_REPOSITORY, "dataset repository"),
        (mirror.get("revision"), EXPECTED_DATASET_REVISION, "dataset revision"),
        (dataset.get("configuration"), "sst2", "dataset configuration"),
        (dataset.get("repository_url"), EXPECTED_DATASET_LOADER_REPOSITORY, "dataset loader repository"),
        (dataset.get("resolved_revision"), EXPECTED_DATASET_LOADER_REVISION, "dataset loader revision"),
        (values.get("model_source_repository"), EXPECTED_MODEL_REPOSITORY, "configured model repository"),
        (values.get("model_source_revision"), EXPECTED_MODEL_REVISION, "configured model source revision"),
        (values.get("model_revision"), EXPECTED_MODEL_REVISION, "configured model revision"),
        (values.get("dataset_loader_repository"), EXPECTED_DATASET_LOADER_REPOSITORY, "configured dataset loader repository"),
        (values.get("dataset_loader_revision"), EXPECTED_DATASET_LOADER_REVISION, "configured dataset loader revision"),
        (values.get("dataset_archive_sha256"), raw_archive.get("sha256"), "configured dataset archive hash"),
    )
    for actual, expected, label in checks:
        _require(actual == expected, f"Unexpected {label}")
    return {
        "model": {
            "repository": EXPECTED_MODEL_REPOSITORY,
            "revision": EXPECTED_MODEL_REVISION,
        },
        "dataset": {
            "repository": EXPECTED_DATASET_REPOSITORY,
            "configuration": "sst2",
            "revision": EXPECTED_DATASET_REVISION,
            "loader_repository": EXPECTED_DATASET_LOADER_REPOSITORY,
            "loader_revision": EXPECTED_DATASET_LOADER_REVISION,
            "archive_sha256": raw_archive["sha256"],
        },
    }


def _verify_configuration(
    project_root: Path, artifact_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_path = artifact_root / "resolved-config.json"
    runner_path = artifact_root / "runner-config.json"
    execution = _load_json(artifact_root / "execution-tree.json")
    resolved = _load_json(resolved_path)
    runner = _load_json(runner_path)
    try:
        config = ResolvedConfig.from_json(resolved_path.read_text(encoding="utf-8"))
        expected_runner = resolve_runner_payload(config, project_root)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SmokeEvidenceError(
            "Resolved config cannot produce the canonical runner payload"
        ) from error
    values = config.values
    authorities = config.authority_by_field
    _require(isinstance(values, dict), "Resolved config values are missing")
    _require(isinstance(authorities, dict), "Resolved config authorities are missing")
    _require(set(values) == set(authorities), "Resolved config authorities do not cover every field")
    _require(execution.get("resolved_config_sha256") == _sha256(resolved_path), "Resolved config hash mismatch")
    _require(execution.get("runner_config_sha256") == _sha256(runner_path), "Runner config hash mismatch")
    _require(
        runner == expected_runner,
        "Runner config does not exactly match resolved config",
    )
    _require(runner.get("do_test") is False, "Runner config do_test must be false")
    _require(values.get("do_test") is False, "Resolved config do_test must be false")
    _require(values.get("do_train") is True, "Smoke must enable training")
    _require(values.get("do_eval") is True, "Smoke must enable evaluation")
    _require(values.get("predict_with_generate") is True, "Smoke must use generated evaluation")
    _require(values.get("max_steps") == 1, "Smoke must run exactly one optimizer step")
    _require(values.get("lora_r") == 8, "Smoke must use LoRA rank 8")
    _require(values.get("lora_alpha") == 16, "Smoke must use LoRA alpha 16")
    _require(values.get("lora_dropout") == 0.0, "Smoke LoRA dropout must be zero")
    _require(
        values.get("effective_model_dropout") == 0.0,
        "Smoke effective model dropout must be zero",
    )
    _require(values.get("modified_modules") == EXPECTED_MODIFIED_MODULES, "LoRA target modules drifted")
    _require(values.get("unfrozen_modules") == EXPECTED_UNFROZEN_MODULES, "Unfrozen modules drifted")
    input_hashes = {
        "source_lock_sha256": _sha256(
            project_root / "data/manifests/source-lock.json"
        ),
        "test_indices_sha256": _sha256(
            project_root / "data/manifests/sst2-test-indices.json"
        ),
        "course_data_manifest_sha256": _sha256(
            project_root / "data/manifests/sst2-course-data.json"
        ),
    }
    for field, actual_hash in input_hashes.items():
        _require(
            values.get(field) == actual_hash,
            f"Resolved config {field} does not match the project file",
        )
    forbidden = _keys_containing(resolved, "unseen") + _keys_containing(runner, "unseen")
    _require(not forbidden, f"Non-final configuration contains unseen fields: {forbidden}")
    return values, {
        "experiment_id": config.experiment_id,
        "resolved_config_sha256": _sha256(resolved_path),
        "runner_config_sha256": _sha256(runner_path),
        "runner_payload_matches_resolved_config": True,
        "do_test": False,
        "contains_unseen_fields": False,
        "max_steps": 1,
        "rank": 8,
        "alpha": 16,
        "effective_model_dropout": 0.0,
        **input_hashes,
    }


def _verify_process(
    artifact_root: Path, values: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    execution = _load_json(artifact_root / "execution-tree.json")
    process = _load_json(artifact_root / "process-result.json")
    _require(process.get("schema_version") == 1, "Unsupported process-result schema")
    _require(process.get("exit_code") == 0, "Original runner did not exit successfully")
    _require(process.get("entrypoint") == EXPECTED_RUNNER, "Process did not invoke the original runner path")
    _require(
        process.get("argv") == ["python", "run_seq2seq.py", "runner-config.json"],
        "Recorded runner argv is unexpected",
    )
    resolved_hash = _sha256(artifact_root / "resolved-config.json")
    _require(
        process.get("resolved_config_sha256") == resolved_hash,
        "Process result references a different resolved config",
    )
    execution_model_binding = execution.get("model_input_binding")
    process_model_binding = process.get("model_input_binding")
    for label, binding in (
        ("Execution", execution_model_binding),
        ("Process", process_model_binding),
    ):
        _require(isinstance(binding, dict), f"{label} model input binding is absent")
        _require(
            binding.get("verified") is True,
            f"Model input binding was not verified ({label.lower()})",
        )
        _require(
            binding.get("source_lock_sha256") == values.get("source_lock_sha256"),
            f"Model input binding source-lock hash mismatch ({label.lower()})",
        )
    _require(
        execution_model_binding == process_model_binding,
        "Model input binding differs between execution and process evidence",
    )

    execution_course_binding = execution.get("course_data_binding")
    process_course_binding = process.get("course_data_binding")
    for label, binding in (
        ("Execution", execution_course_binding),
        ("Process", process_course_binding),
    ):
        _require(isinstance(binding, dict), f"{label} course-data binding is absent")
        _require(
            binding.get("verified") is True,
            f"{label} course-data binding was not verified",
        )
        _require(
            binding.get("course_data_manifest_sha256")
            == values.get("course_data_manifest_sha256"),
            f"{label} course-data binding manifest hash mismatch",
        )
        _require(
            binding.get("visible_partitions") == ["Training", "Test"],
            "Course-data binding exposes unexpected partitions",
        )
        _require(
            binding.get("final_unseen_exposed") is False,
            "Course-data binding exposed final unseen data",
        )
    _require(
        execution_course_binding == process_course_binding,
        "Course-data binding differs between execution and process evidence",
    )

    execution_commit = execution.get("project_git_commit")
    process_commit = process.get("project_git_commit")
    _require(
        isinstance(execution_commit, str)
        and len(execution_commit) == 40
        and all(character in "0123456789abcdef" for character in execution_commit),
        "Execution project Git commit is not a full commit hash",
    )
    _require(
        process_commit == execution_commit,
        "Project Git commit differs between execution and process evidence",
    )
    _require(
        execution.get("project_worktree_clean") is True,
        "Execution did not record a clean project worktree",
    )
    _require(
        process.get("project_worktree_clean") is True,
        "Process did not record a clean project worktree",
    )
    enforcement = process.get("nonfinal_access_enforcement")
    _require(isinstance(enforcement, dict), "Non-final access enforcement evidence is absent")
    _require(enforcement.get("do_test") is False, "Process result did not enforce do_test=false")
    _require(
        enforcement.get("private_root_exposed_to_child") is False,
        "Private unseen root was exposed to the smoke subprocess",
    )
    _require(
        enforcement.get("final_unseen_file_exposed_to_child") is False,
        "Final unseen file was exposed to the smoke subprocess",
    )
    _require(
        enforcement.get("unseen_fields_in_config") == [],
        "Process result recorded unseen configuration fields",
    )
    try:
        started = datetime.fromisoformat(str(process["started_at_utc"]))
        finished = datetime.fromisoformat(str(process["finished_at_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        raise SmokeEvidenceError("Process timestamps are absent or invalid") from error
    _require(started.tzinfo is not None and finished.tzinfo is not None, "Process timestamps must include UTC offsets")
    _require(finished >= started, "Process finish timestamp precedes its start")
    wall_seconds = _finite_number(process, "wall_seconds", minimum=0.0)
    _require(wall_seconds > 0.0, "wall_seconds must be greater than zero")
    return process, {
        "exit_code": 0,
        "runner_relative_path": EXPECTED_RUNNER,
        "private_root_exposed_to_child": False,
        "final_unseen_file_exposed_to_child": False,
        "project_git_commit": execution_commit,
        "project_worktree_clean": True,
        "model_input_binding": {
            "verified": True,
            "source_lock_sha256": values["source_lock_sha256"],
        },
        "course_data_binding": {
            "verified": True,
            "course_data_manifest_sha256": values[
                "course_data_manifest_sha256"
            ],
            "visible_partitions": ["Training", "Test"],
            "final_unseen_exposed": False,
        },
        "started_at_utc": process["started_at_utc"],
        "finished_at_utc": process["finished_at_utc"],
        "wall_seconds": wall_seconds,
    }


def _verify_run_outputs(artifact_root: Path, values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trainer = _load_json(artifact_root / "trainer_state.json")
    checkpoint_trainer = _load_json(artifact_root / "checkpoint-1/trainer_state.json")
    train = _load_json(artifact_root / "train_results.json")
    evaluation = _load_json(artifact_root / "eval_results.json")
    performance = _load_json(artifact_root / "performance_results.json")
    combined = _load_json(artifact_root / "all_results.json")

    _require(trainer.get("global_step") == 1, "Trainer did not finish exactly one step")
    _require(trainer.get("max_steps") == 1, "Trainer max_steps is not one")
    _require(checkpoint_trainer.get("global_step") == 1, "Checkpoint is not from step one")
    history = trainer.get("log_history", [])
    _require(
        any(item.get("step") == 1 and "loss" in item for item in history if isinstance(item, dict)),
        "Step-one training loss is absent",
    )
    _require(
        any(item.get("step") == 1 and "eval_loss" in item for item in history if isinstance(item, dict)),
        "Step-one generated evaluation is absent",
    )

    train_loss = _finite_number(train, "train_loss", minimum=0.0)
    train_runtime = _finite_number(train, "train_runtime", minimum=0.0)
    _require(train_runtime > 0.0, "train_runtime must be greater than zero")
    _require(train.get("train_samples") == 1, "Smoke must contain one training sample")
    eval_loss = _finite_number(evaluation, "eval_loss", minimum=0.0)
    eval_accuracy = _finite_number(evaluation, "eval_accuracy", minimum=0.0)
    _require(eval_accuracy <= 1.0, "Smoke evaluation accuracy is outside [0, 1]")
    eval_runtime = _finite_number(evaluation, "eval_runtime", minimum=0.0)
    _require(eval_runtime > 0.0, "eval_runtime must be greater than zero")
    peak_memory = _finite_number(performance, "peak_memory", minimum=0.0)
    _require(peak_memory > 0.0, "peak_memory must be greater than zero")
    training_time_minutes = _finite_number(performance, "total_time in minutes ", minimum=0.0)
    _require(
        training_time_minutes > 0.0,
        "total_time in minutes must be greater than zero",
    )
    _require(combined.get("train_loss") == train.get("train_loss"), "Combined train metrics drifted")
    _require(combined.get("eval_loss") == evaluation.get("eval_loss"), "Combined eval metrics drifted")
    _require(combined.get("peak_memory") == performance.get("peak_memory"), "Combined memory metric drifted")
    _require(values.get("max_val_samples") == 1, "Smoke must evaluate one Test example")

    return (
        {
            "global_step": 1,
            "train_samples": 1,
            "train_loss": train_loss,
            "train_runtime_seconds": train_runtime,
        },
        {
            "course_partition": "Test",
            "evaluated_samples": 1,
            "generation_path_exercised": True,
            "accuracy": eval_accuracy,
            "loss": eval_loss,
            "runtime_seconds": eval_runtime,
        },
        {
            "upstream_peak_memory_gb": peak_memory,
            "upstream_training_time_minutes": training_time_minutes,
        },
    )


def _verify_checkpoint_scope(artifact_root: Path) -> dict[str, int]:
    checkpoint_paths = (
        artifact_root / "checkpoint-1/pytorch_model.bin",
        artifact_root / "pytorch_model.bin",
        artifact_root / "delta/pytorch_model.bin",
    )
    reference_keys: set[str] | None = None
    for path in checkpoint_paths:
        _require(path.is_file(), f"Missing parameter checkpoint: {path.name}")
        counts, keys = _load_checkpoint_counts(path)
        _require(counts == EXPECTED_COUNTS, f"Rank-8 trainable counts mismatch in {path.name}: {counts}")
        if reference_keys is None:
            reference_keys = keys
        else:
            _require(keys == reference_keys, "Saved checkpoint tensor keys do not match")

    delta_config = _load_json(artifact_root / "delta/config.json")
    _require(delta_config.get("delta_type") == "lora", "Native delta checkpoint is not LoRA")
    _require(delta_config.get("lora_r") == 8, "Native delta checkpoint rank mismatch")
    _require(delta_config.get("lora_alpha") == 16, "Native delta checkpoint alpha mismatch")
    _require(delta_config.get("modified_modules") == EXPECTED_MODIFIED_MODULES, "Native delta target modules mismatch")
    _require(delta_config.get("unfrozen_modules") == EXPECTED_UNFROZEN_MODULES, "Native delta unfrozen modules mismatch")
    _require(delta_config.get("backbone_checkpoint_name") == EXPECTED_MODEL_REVISION, "Native delta backbone revision mismatch")
    return dict(EXPECTED_COUNTS)


def _verify_reload(artifact_root: Path) -> dict[str, Any]:
    reload_evidence = _load_json(artifact_root / "reload-verification.json")
    _require(reload_evidence.get("schema_version") == 1, "Unsupported reload evidence schema")
    _require(reload_evidence.get("checkpoint_relative_path") == "delta", "Reload did not use native delta checkpoint")
    stable_id = reload_evidence.get("stable_id")
    _require(isinstance(stable_id, str) and stable_id.startswith("validation:"), "Reload stable Test ID is missing")
    _require(reload_evidence.get("retained_prediction") == reload_evidence.get("reloaded_prediction"), "Reloaded prediction differs")
    _require(reload_evidence.get("predictions_match") is True, "Reload prediction check did not pass")
    saved_loss = _finite_number(reload_evidence, "retained_eval_loss")
    reloaded_loss = _finite_number(reload_evidence, "reloaded_eval_loss")
    reported_difference = _finite_number(reload_evidence, "abs_eval_loss_difference", minimum=0.0)
    tolerance = _finite_number(reload_evidence, "tolerance", minimum=0.0)
    actual_difference = abs(saved_loss - reloaded_loss)
    _require(math.isclose(reported_difference, actual_difference, rel_tol=0.0, abs_tol=1e-12), "Reload loss difference was reported incorrectly")
    _require(actual_difference <= tolerance <= 1e-4, "Reload loss differs beyond the allowed tolerance")
    _require(
        reload_evidence.get("checkpoint_tensor_keys_match") is True,
        "Native delta and retained checkpoint tensor keys differ",
    )
    maximum_tensor_difference = _finite_number(
        reload_evidence,
        "checkpoint_tensor_max_abs_difference",
        minimum=0.0,
    )
    _require(
        maximum_tensor_difference == 0.0,
        "Native delta and retained checkpoint tensor values differ",
    )
    _require(
        reload_evidence.get("resolved_config_sha256")
        == _sha256(artifact_root / "resolved-config.json"),
        "Reload evidence references a different resolved config",
    )
    _require(reload_evidence.get("passed") is True, "Reload verification is not marked passed")
    return {
        "checkpoint_relative_path": "delta",
        "predictions_match": True,
        "loss_difference": actual_difference,
        "tolerance": tolerance,
        "checkpoint_tensors_match": True,
        "checkpoint_tensor_max_abs_difference": maximum_tensor_difference,
        "passed": True,
    }


def verify_week1_smoke(project_root: Path, artifact_root: Path) -> dict[str, Any]:
    """Return a path-sanitized manifest for one completed non-result smoke."""

    project_root = project_root.resolve(strict=True)
    artifact_root = artifact_root.resolve(strict=True)
    upstream = _verify_provenance(project_root, artifact_root)
    values, configuration = _verify_configuration(project_root, artifact_root)
    source_lock = _load_json(project_root / "data/manifests/source-lock.json")
    inputs = _verify_inputs(source_lock, values)
    _, process = _verify_process(artifact_root, values)
    training, evaluation, measurement = _verify_run_outputs(artifact_root, values)
    counts = _verify_checkpoint_scope(artifact_root)
    reload_result = _verify_reload(artifact_root)
    artifact_records = [_artifact_record(artifact_root, name) for name in EVIDENCE_FILES]

    return {
        "schema_version": 1,
        "artifact_kind": "week1-non-result-smoke",
        "status": "verified",
        "scope_note": "Execution smoke only; these metrics are not a reproduction result.",
        "upstream": upstream,
        "inputs": inputs,
        "configuration": configuration,
        "process": process,
        "data_access": {
            "do_test": False,
            "contains_unseen_fields": False,
            "private_root_exposed_to_child": False,
            "final_unseen_file_exposed_to_child": False,
            "visible_partitions": ["Training", "Test"],
            "course_data_binding_verified": True,
            "final_unseen_access_prevented": True,
        },
        "training": training,
        "evaluation": evaluation,
        "measurement": measurement,
        "trainable_parameters": counts,
        "checkpoint_reload": reload_result,
        "artifacts": artifact_records,
        "evidence_gaps": [],
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--write-manifest", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        manifest = verify_week1_smoke(arguments.project_root, arguments.artifact_root)
        _write_manifest(arguments.write_manifest, manifest)
    except (OSError, SmokeEvidenceError) as error:
        print(f"week1 smoke verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "verified", "manifest": str(arguments.write_manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
