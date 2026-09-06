from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import torch

from ds340w_repro.config_schema import ResolvedConfig
from ds340w_repro.run_opendelta import resolve_runner_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify_week1_smoke.py"
UPSTREAM_COMMIT = "20e426f716ea53618657cd34bbfc4b347b98cd74"
MODEL_REVISION = "bf48a6b7de15c6afe1e27e1a51aa1ea4626f96bf"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
DATASET_LOADER_REVISION = "dff6c92c80fbfac456349a531c0a4521c2b0c8f8"
RUNNER_RELATIVE_PATH = "examples/legacies/examples_seq2seq/run_seq2seq.py"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _checkpoint_state() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for projection in range(72):
        prefix = f"block.{projection}.attention.q.lora"
        state[f"{prefix}.lora_A"] = torch.zeros((8, 768))
        state[f"{prefix}.lora_B"] = torch.zeros((768, 8))
    state["encoder.final_layer_norm.weight"] = torch.zeros(47_616)
    return state


def _make_valid_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    artifact = tmp_path / "private-machine" / "smoke-output"
    runtime = tmp_path / "private-machine" / "runtime"
    output = project / "data" / "manifests" / "week1-smoke.json"

    pristine_runner = project / "vendor" / "OpenDelta" / RUNNER_RELATIVE_PATH
    pristine_runner.parent.mkdir(parents=True)
    pristine_runner.write_bytes(b"original author runner\n")
    patched_runner = runtime / RUNNER_RELATIVE_PATH
    patched_runner.parent.mkdir(parents=True)
    patched_runner.write_bytes(b"patched author runner\n")

    patch_path = project / "patches" / "0001-test.patch"
    patch_path.parent.mkdir(parents=True)
    patch_path.write_bytes(b"synthetic compatibility patch\n")
    _write_json(
        project / "patches" / "series.json",
        {
            "schema_version": 1,
            "upstream_commit": UPSTREAM_COMMIT,
            "patches": [
                {
                    "path": "patches/0001-test.patch",
                    "sha256": _digest(patch_path),
                    "classification": "compatibility",
                    "rationale": "synthetic fixture",
                    "touched_paths": [RUNNER_RELATIVE_PATH],
                }
            ],
        },
    )
    _write_json(
        project / "data" / "manifests" / "upstream-source.json",
        {
            "commit": UPSTREAM_COMMIT,
            "is_clean": True,
            "remote_url": "https://github.com/thunlp/OpenDelta.git",
            "runner_relative_path": RUNNER_RELATIVE_PATH,
            "runner_sha256": _digest(pristine_runner),
        },
    )
    source_lock_path = project / "data" / "manifests" / "source-lock.json"
    _write_json(
        source_lock_path,
        {
            "schema_version": 1,
            "opendelta": {
                "repository_url": "https://github.com/thunlp/OpenDelta.git",
                "requested_revision": UPSTREAM_COMMIT,
                "resolved_revision": UPSTREAM_COMMIT,
            },
            "model": {
                "repo_id": "google/t5-base-lm-adapt",
                "requested_revision": MODEL_REVISION,
                "resolved_revision": MODEL_REVISION,
            },
            "tokenizer": {
                "repo_id": "google/t5-base-lm-adapt",
                "requested_revision": MODEL_REVISION,
                "resolved_revision": MODEL_REVISION,
            },
            "dataset": {
                "dataset_name": "glue",
                "configuration": "sst2",
                "provenance_mirror": {
                    "repo_id": "nyu-mll/glue",
                    "revision": DATASET_REVISION,
                },
                "repository_url": "https://github.com/huggingface/datasets.git",
                "requested_revision": DATASET_LOADER_REVISION,
                "resolved_revision": DATASET_LOADER_REVISION,
                "raw_archive": {"sha256": "a" * 64},
            },
        },
    )
    test_indices_path = (
        project / "data" / "manifests" / "sst2-test-indices.json"
    )
    _write_json(test_indices_path, {"schema_version": 1, "indices": [718]})
    course_data_manifest_path = (
        project / "data" / "manifests" / "sst2-course-data.json"
    )
    _write_json(
        course_data_manifest_path,
        {"schema_version": 1, "partitions": ["Training", "Test"]},
    )

    values = {
        "model_name_or_path": "verified-model",
        "tokenizer_name": "verified-model",
        "output_dir": str(artifact),
        "do_train": True,
        "do_eval": True,
        "do_test": False,
        "predict_with_generate": True,
        "max_steps": 1,
        "max_train_samples": 1,
        "max_val_samples": 1,
        "model_revision": MODEL_REVISION,
        "model_source_repository": "google/t5-base-lm-adapt",
        "model_source_revision": MODEL_REVISION,
        "dataset_loader_repository": "https://github.com/huggingface/datasets.git",
        "dataset_loader_revision": DATASET_LOADER_REVISION,
        "dataset_archive_sha256": "a" * 64,
        "source_lock_sha256": _digest(source_lock_path),
        "test_indices_sha256": _digest(test_indices_path),
        "course_data_manifest_sha256": _digest(course_data_manifest_path),
        "task_name": "sst2",
        "eval_dataset_name": "sst2",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "effective_model_dropout": 0.0,
        "modified_modules": [
            "attn@.q@",
            "attn@.v@",
            "crossattn@.q@",
            "crossattn@.v@",
        ],
        "unfrozen_modules": ["deltas", "layer_norm@"],
    }
    resolved = ResolvedConfig(
        experiment_id="week1-smoke-r8",
        values=values,
        authority_by_field={key: "test fixture" for key in values},
    )
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "resolved-config.json").write_text(
        resolved.to_json(), encoding="utf-8", newline="\n"
    )
    _write_json(
        artifact / "runner-config.json",
        resolve_runner_payload(resolved, project),
    )
    model_input_binding = {
        "file_count": 7,
        "source_lock_sha256": _digest(source_lock_path),
        "verified": True,
    }
    course_data_binding = {
        "course_data_manifest_sha256": _digest(course_data_manifest_path),
        "final_unseen_exposed": False,
        "test_sha256": "b" * 64,
        "training_sha256": "c" * 64,
        "verified": True,
        "visible_partitions": ["Training", "Test"],
    }
    project_git_commit = "d" * 40
    _write_json(
        artifact / "execution-tree.json",
        {
            "root": str(runtime),
            "runner": str(patched_runner),
            "upstream_commit": UPSTREAM_COMMIT,
            "pre_patch_runner_sha256": _digest(pristine_runner),
            "post_patch_runner_sha256": _digest(patched_runner),
            "applied_patch_sha256": [_digest(patch_path)],
            "resolved_config_sha256": _digest(artifact / "resolved-config.json"),
            "runner_config_sha256": _digest(artifact / "runner-config.json"),
            "model_input_binding": model_input_binding,
            "course_data_binding": course_data_binding,
            "project_git_commit": project_git_commit,
            "project_worktree_clean": True,
        },
    )
    _write_json(
        artifact / "process-result.json",
        {
            "schema_version": 1,
            "entrypoint": RUNNER_RELATIVE_PATH,
            "argv": ["python", "run_seq2seq.py", "runner-config.json"],
            "exit_code": 0,
            "started_at_utc": "2026-09-06T16:19:38+00:00",
            "finished_at_utc": "2026-09-06T16:19:56+00:00",
            "wall_seconds": 20.5,
            "resolved_config_sha256": _digest(artifact / "resolved-config.json"),
            "model_input_binding": model_input_binding,
            "course_data_binding": course_data_binding,
            "project_git_commit": project_git_commit,
            "project_worktree_clean": True,
            "nonfinal_access_enforcement": {
                "do_test": False,
                "private_root_exposed_to_child": False,
                "final_unseen_file_exposed_to_child": False,
                "unseen_fields_in_config": [],
            },
        },
    )
    _write_json(
        artifact / "trainer_state.json",
        {
            "global_step": 1,
            "max_steps": 1,
            "log_history": [
                {"step": 1, "loss": 10.0},
                {
                    "step": 1,
                    "eval_accuracy": 0.0,
                    "eval_loss": 8.0,
                    "eval_runtime": 5.0,
                },
            ],
        },
    )
    _write_json(
        artifact / "checkpoint-1" / "trainer_state.json",
        {"global_step": 1, "max_steps": 1},
    )
    _write_json(
        artifact / "train_results.json",
        {"train_loss": 10.0, "train_runtime": 6.0, "train_samples": 1},
    )
    _write_json(
        artifact / "eval_results.json",
        {
            "eval_accuracy": 0.0,
            "eval_loss": 8.0,
            "eval_runtime": 5.0,
            "eval_samples_per_second": 0.2,
        },
    )
    _write_json(
        artifact / "all_results.json",
        {
            "train_loss": 10.0,
            "train_runtime": 6.0,
            "eval_accuracy": 0.0,
            "eval_loss": 8.0,
            "eval_runtime": 5.0,
            "peak_memory": 1.0,
            "total_time in minutes ": 0.1,
        },
    )
    _write_json(
        artifact / "performance_results.json",
        {"peak_memory": 1.0, "total_time in minutes ": 0.1},
    )
    _write_json(
        artifact / "delta" / "config.json",
        {
            "backbone_checkpoint_name": MODEL_REVISION,
            "delta_type": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "modified_modules": values["modified_modules"],
            "unfrozen_modules": values["unfrozen_modules"],
        },
    )
    state = _checkpoint_state()
    torch.save(state, artifact / "checkpoint-1" / "pytorch_model.bin")
    torch.save(state, artifact / "delta" / "pytorch_model.bin")
    torch.save(state, artifact / "pytorch_model.bin")
    (artifact / "checkpoint-1" / "optimizer.pt").write_bytes(b"optimizer\n")
    (artifact / "checkpoint-1" / "scheduler.pt").write_bytes(b"scheduler\n")
    _write_json(
        artifact / "reload-verification.json",
        {
            "schema_version": 1,
            "checkpoint_relative_path": "delta",
            "stable_id": "validation:718",
            "retained_prediction": "0",
            "reloaded_prediction": "0",
            "retained_eval_loss": 8.0,
            "reloaded_eval_loss": 8.0000009,
            "abs_eval_loss_difference": 0.0000009,
            "tolerance": 0.0001,
            "predictions_match": True,
            "checkpoint_tensor_keys_match": True,
            "checkpoint_tensor_max_abs_difference": 0.0,
            "resolved_config_sha256": _digest(artifact / "resolved-config.json"),
            "passed": True,
        },
    )
    return project, artifact, output


def _run_verifier(project: Path, artifact: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--project-root",
            str(project),
            "--artifact-root",
            str(artifact),
            "--write-manifest",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_verifies_smoke_and_writes_sanitized_manifest(tmp_path: Path) -> None:
    project, artifact, output = _make_valid_fixture(tmp_path)

    result = _run_verifier(project, artifact, output)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert manifest["artifact_kind"] == "week1-non-result-smoke"
    assert manifest["training"]["global_step"] == 1
    assert manifest["evaluation"]["generation_path_exercised"] is True
    assert manifest["trainable_parameters"] == {
        "injected_projections": 72,
        "layer_norm_parameters": 47_616,
        "lora_parameters": 884_736,
        "total_trainable_parameters": 932_352,
    }
    assert manifest["data_access"]["final_unseen_access_prevented"] is True
    assert manifest["data_access"]["visible_partitions"] == ["Training", "Test"]
    assert (
        manifest["data_access"]["final_unseen_file_exposed_to_child"] is False
    )
    assert manifest["configuration"]["runner_payload_matches_resolved_config"] is True
    assert manifest["configuration"]["effective_model_dropout"] == 0.0
    assert manifest["configuration"]["source_lock_sha256"] == _digest(
        project / "data" / "manifests" / "source-lock.json"
    )
    assert manifest["configuration"]["test_indices_sha256"] == _digest(
        project / "data" / "manifests" / "sst2-test-indices.json"
    )
    assert manifest["configuration"]["course_data_manifest_sha256"] == _digest(
        project / "data" / "manifests" / "sst2-course-data.json"
    )
    assert manifest["process"]["wall_seconds"] == 20.5
    assert manifest["process"]["project_git_commit"] == "d" * 40
    assert manifest["process"]["project_worktree_clean"] is True
    assert manifest["process"]["model_input_binding"] == {
        "source_lock_sha256": manifest["configuration"]["source_lock_sha256"],
        "verified": True,
    }
    assert manifest["process"]["course_data_binding"] == {
        "course_data_manifest_sha256": manifest["configuration"][
            "course_data_manifest_sha256"
        ],
        "final_unseen_exposed": False,
        "verified": True,
        "visible_partitions": ["Training", "Test"],
    }
    assert manifest["checkpoint_reload"]["passed"] is True
    assert manifest["checkpoint_reload"]["loss_difference"] < 1e-4
    assert manifest["checkpoint_reload"]["checkpoint_tensors_match"] is True
    artifacts_by_path = {
        item["relative_path"]: item for item in manifest["artifacts"]
    }
    assert artifacts_by_path["delta/config.json"]["role"] == "native-delta-config"
    assert (
        artifacts_by_path["delta/pytorch_model.bin"]["role"]
        == "native-delta-weights"
    )
    assert artifacts_by_path["delta/pytorch_model.bin"]["size_bytes"] > 0
    serialized = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "source_indices" not in serialized
    assert "retained_prediction" not in serialized
    assert "reloaded_prediction" not in serialized


def test_cli_rejects_zero_peak_memory_as_missing_measurement(tmp_path: Path) -> None:
    project, artifact, output = _make_valid_fixture(tmp_path)
    _write_json(
        artifact / "performance_results.json",
        {"peak_memory": 0.0, "total_time in minutes ": 0.1},
    )
    combined = json.loads((artifact / "all_results.json").read_text(encoding="utf-8"))
    combined["peak_memory"] = 0.0
    _write_json(artifact / "all_results.json", combined)

    result = _run_verifier(project, artifact, output)

    assert result.returncode == 1
    assert "peak_memory must be greater than zero" in result.stderr
    assert not output.exists()


def test_cli_rejects_private_unseen_root_exposure(tmp_path: Path) -> None:
    project, artifact, output = _make_valid_fixture(tmp_path)
    process = json.loads(
        (artifact / "process-result.json").read_text(encoding="utf-8")
    )
    process["nonfinal_access_enforcement"]["private_root_exposed_to_child"] = True
    _write_json(artifact / "process-result.json", process)

    result = _run_verifier(project, artifact, output)

    assert result.returncode == 1
    assert "Private unseen root was exposed" in result.stderr
    assert not output.exists()


def test_cli_rejects_runner_payload_mismatch(tmp_path: Path) -> None:
    project, artifact, output = _make_valid_fixture(tmp_path)
    runner_path = artifact / "runner-config.json"
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    runner["unexpected_argument"] = True
    _write_json(runner_path, runner)
    execution_path = artifact / "execution-tree.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["runner_config_sha256"] = _digest(runner_path)
    _write_json(execution_path, execution)

    result = _run_verifier(project, artifact, output)

    assert result.returncode == 1
    assert "Runner config does not exactly match resolved config" in result.stderr
    assert not output.exists()


def test_cli_rejects_model_binding_with_wrong_source_lock(tmp_path: Path) -> None:
    project, artifact, output = _make_valid_fixture(tmp_path)
    for name in ("execution-tree.json", "process-result.json"):
        path = artifact / name
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["model_input_binding"]["source_lock_sha256"] = "e" * 64
        _write_json(path, evidence)

    result = _run_verifier(project, artifact, output)

    assert result.returncode == 1
    assert "Model input binding source-lock hash mismatch" in result.stderr
    assert not output.exists()


def test_cli_rejects_course_data_partition_exposure(tmp_path: Path) -> None:
    project, artifact, output = _make_valid_fixture(tmp_path)
    for name in ("execution-tree.json", "process-result.json"):
        path = artifact / name
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["course_data_binding"]["visible_partitions"] = [
            "Training",
            "Test",
            "Validation (Unseen)",
        ]
        _write_json(path, evidence)

    result = _run_verifier(project, artifact, output)

    assert result.returncode == 1
    assert "Course-data binding exposes unexpected partitions" in result.stderr
    assert not output.exists()


def test_cli_rejects_final_unseen_course_data_exposure(tmp_path: Path) -> None:
    project, artifact, output = _make_valid_fixture(tmp_path)
    for name in ("execution-tree.json", "process-result.json"):
        path = artifact / name
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["course_data_binding"]["final_unseen_exposed"] = True
        _write_json(path, evidence)

    result = _run_verifier(project, artifact, output)

    assert result.returncode == 1
    assert "Course-data binding exposed final unseen data" in result.stderr
    assert not output.exists()


def test_cli_rejects_final_unseen_file_exposure_to_child(tmp_path: Path) -> None:
    project, artifact, output = _make_valid_fixture(tmp_path)
    process_path = artifact / "process-result.json"
    process = json.loads(process_path.read_text(encoding="utf-8"))
    process["nonfinal_access_enforcement"][
        "final_unseen_file_exposed_to_child"
    ] = True
    _write_json(process_path, process)

    result = _run_verifier(project, artifact, output)

    assert result.returncode == 1
    assert "Final unseen file was exposed" in result.stderr
    assert not output.exists()


def test_cli_rejects_mismatched_project_git_commit(tmp_path: Path) -> None:
    project, artifact, output = _make_valid_fixture(tmp_path)
    process_path = artifact / "process-result.json"
    process = json.loads(process_path.read_text(encoding="utf-8"))
    process["project_git_commit"] = "f" * 40
    _write_json(process_path, process)

    result = _run_verifier(project, artifact, output)

    assert result.returncode == 1
    assert "Project Git commit differs" in result.stderr
    assert not output.exists()
