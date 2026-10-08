"""Prepare and invoke a patched copy of the authors' original runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from ds340w_repro.config_schema import ResolvedConfig, validate_nonfinal_config
from ds340w_repro.data.create_sst2_split import OFFICIAL_VALIDATION_ROWS
from ds340w_repro.data.materialize_sst2 import (
    OFFICIAL_TRAINING_ROWS,
    TEST_FILENAME,
    TRAINING_FILENAME,
    verify_course_data,
)
from ds340w_repro.preflight import (
    ComputeGateError,
    require_retained_training_authorization,
    validated_preflight_phase,
)
from ds340w_repro.provenance import verify_upstream


ALLOWED_PATCH_CLASSIFICATIONS = {
    "compatibility",
    "configuration",
    "data-handling",
    "provenance",
    "observation-only",
}


@dataclass(frozen=True)
class ExecutionTreeReport:
    root: Path
    runner: Path
    upstream_commit: str
    pre_patch_runner_sha256: str
    post_patch_runner_sha256: str
    applied_patch_sha256: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_project_git_binding(project_root: Path) -> dict[str, object]:
    """Return the exact committed source state, refusing a dirty worktree."""

    project_root = project_root.resolve(strict=True)
    commit_result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = commit_result.stdout.strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Project Git HEAD is not a full commit hash")
    status_result = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if status_result.stdout:
        raise ValueError("Project worktree must be clean before execution")
    return {
        "project_git_commit": commit,
        "project_worktree_clean": True,
    }


def _touched_paths(patch_text: str) -> set[str]:
    return {
        line.removeprefix("+++ b/")
        for line in patch_text.splitlines()
        if line.startswith("+++ b/")
    }


def resolve_runner_payload(
    config: ResolvedConfig, project_root: Path
) -> dict[str, object]:
    """Resolve filesystem arguments once, relative to the repository root."""

    project_root = project_root.resolve(strict=True)
    payload = config.runner_payload()
    for field in ("model_name_or_path", "tokenizer_name", "output_dir"):
        candidate = Path(str(payload[field])).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        payload[field] = str(candidate.resolve())
    return payload


def verify_model_snapshot_binding(
    config: ResolvedConfig, project_root: Path, cache_root: Path
) -> dict[str, object]:
    """Bind configured model paths to every byte recorded in source-lock.json."""

    project_root = project_root.resolve(strict=True)
    cache_root = cache_root.resolve(strict=True)
    source_lock_path = project_root / "data" / "manifests" / "source-lock.json"
    source_lock_bytes = source_lock_path.read_bytes()
    source_lock_sha256 = sha256(source_lock_bytes).hexdigest()
    if source_lock_sha256 != config.values.get("source_lock_sha256"):
        raise ValueError("Source-lock checksum does not match configuration")
    source_lock = json.loads(source_lock_bytes)
    model_lock = source_lock.get("model")
    if not isinstance(model_lock, dict):
        raise ValueError("Source lock has no model record")
    cache_relative_path = Path(str(model_lock["cache_relative_path"]))
    cache_parts = cache_relative_path.parts
    if (
        cache_relative_path.is_absolute()
        or ".." in cache_parts
        or len(cache_parts) != 4
        or cache_parts[0] != "hub"
        or not cache_parts[1].startswith("models--")
        or cache_parts[2] != "snapshots"
        or cache_parts[3] != model_lock.get("resolved_revision")
    ):
        raise ValueError("Unsafe model snapshot path in source lock")
    model_cache_root = (cache_root / cache_parts[0] / cache_parts[1]).resolve(
        strict=True
    )
    expected_snapshot = (cache_root / cache_relative_path).resolve(strict=True)
    if not expected_snapshot.is_relative_to(model_cache_root):
        raise ValueError("Unsafe model snapshot path in source lock")
    payload = resolve_runner_payload(config, project_root)
    configured_paths = {
        Path(str(payload[field])).resolve(strict=True)
        for field in ("model_name_or_path", "tokenizer_name")
    }
    if configured_paths != {expected_snapshot}:
        raise ValueError(
            "Model and tokenizer must use the approved locked snapshot"
        )

    records = model_lock.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Source lock model file list is empty")
    for record in records:
        relative_path = Path(str(record["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Unsafe model file in source lock")
        path = expected_snapshot / relative_path
        try:
            resolved_target = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError("Unsafe model file in source lock") from error
        if not path.is_file() or not resolved_target.is_relative_to(model_cache_root):
            raise ValueError("Unsafe model file in source lock")
        if path.stat().st_size != record.get("size"):
            raise ValueError(f"Model size mismatch: {path.name}")
        if _file_sha256(path) != record.get("sha256"):
            raise ValueError(f"Model checksum mismatch: {path.name}")
    return {
        "file_count": len(records),
        "source_lock_sha256": source_lock_sha256,
        "verified": True,
    }


def verify_course_data_binding(
    config: ResolvedConfig,
    project_root: Path,
    cache_root: Path,
    *,
    expected_training_rows: int = OFFICIAL_TRAINING_ROWS,
    expected_validation_rows: int = OFFICIAL_VALIDATION_ROWS,
) -> tuple[dict[str, object], Path, Path]:
    """Verify and return the only two dataset files exposed to the runner."""

    project_root = project_root.resolve(strict=True)
    course_data_root = (cache_root / "course-data").resolve(strict=True)
    manifest_path = project_root / "data" / "manifests" / "sst2-course-data.json"
    test_indices_path = project_root / "data" / "manifests" / "sst2-test-indices.json"
    manifest_sha256 = _file_sha256(manifest_path)
    if manifest_sha256 != config.values.get("course_data_manifest_sha256"):
        raise ValueError("Course-data manifest checksum does not match configuration")
    if _file_sha256(test_indices_path) != config.values.get("test_indices_sha256"):
        raise ValueError("Course Test-index checksum does not match configuration")
    report = verify_course_data(
        output_root=course_data_root,
        manifest_path=manifest_path,
        test_indices_path=test_indices_path,
        expected_training_rows=expected_training_rows,
        expected_validation_rows=expected_validation_rows,
    )
    if report["manifest_sha256"] != manifest_sha256:
        raise ValueError("Course-data verifier returned a different manifest checksum")
    training_path = (course_data_root / TRAINING_FILENAME).resolve(strict=True)
    test_path = (course_data_root / TEST_FILENAME).resolve(strict=True)
    evidence: dict[str, object] = {
        "course_data_manifest_sha256": manifest_sha256,
        "final_unseen_exposed": False,
        "test_sha256": report["test_sha256"],
        "training_sha256": report["training_sha256"],
        "verified": report["valid"],
        "visible_partitions": ["Training", "Test"],
    }
    return evidence, training_path, test_path


def enforce_compute_gate(
    config: ResolvedConfig,
    gate_manifest_path: Path,
    resolved_config_path: Path,
) -> None:
    """Allow exact smoke/preflight shapes; stage-gate retained training."""

    if sha256(resolved_config_path.read_bytes()).hexdigest() != config.sha256:
        raise ValueError("Resolved configuration file is not canonical")
    values = config.values
    bounded_smoke = (
        config.experiment_id == "week1-smoke-r8"
        and values.get("max_steps") == 1
        and values.get("max_train_samples") == 1
        and values.get("max_val_samples") == 1
        and values.get("effective_batch_size") == 1
        and values.get("lora_r") == 8
        and values.get("lora_alpha") == 16
    )
    if bounded_smoke:
        return
    if validated_preflight_phase(config) is not None:
        return

    baseline_ids = {"baseline-r8-lr1e-4-eb16"}
    if config.experiment_id in baseline_ids:
        stage = "baseline"
    elif config.experiment_id in {"rank-r2", "rank-r4", "rank-r16"}:
        stage = "rank"
    else:
        raise ComputeGateError(
            f"Unsupported experiment at OpenDelta execution boundary: "
            f"{config.experiment_id}"
        )
    manifest = json.loads(gate_manifest_path.read_text(encoding="utf-8"))
    require_retained_training_authorization(
        manifest,
        stage=stage,
        resolved_config_sha256=config.sha256,
    )


def prepare_execution_tree(project_root: Path, destination: Path) -> ExecutionTreeReport:
    """Copy pristine OpenDelta source and apply the declared patch series."""

    project_root = project_root.resolve(strict=True)
    upstream = project_root / "vendor" / "OpenDelta"
    provenance = verify_upstream(upstream)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Execution tree already exists: {destination}")

    manifest = json.loads(
        (project_root / "patches" / "series.json").read_text(encoding="utf-8")
    )
    if manifest["upstream_commit"] != provenance.commit:
        raise ValueError("Patch series and upstream commit do not match")

    archive = subprocess.run(
        ["git", "-C", str(upstream), "archive", "--format=tar", provenance.commit],
        check=True,
        capture_output=True,
    ).stdout
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source_tar:
        for member in source_tar.getmembers():
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination):
                raise ValueError(f"Unsafe path in upstream archive: {member.name}")
        source_tar.extractall(destination)
    # The pinned archive contains CRLF source files. A disposable Git metadata
    # directory supplies the same text normalization as the pristine checkout.
    subprocess.run(
        ["git", "-C", str(destination), "init", "--quiet"], check=True
    )
    subprocess.run(
        ["git", "-C", str(destination), "config", "core.autocrlf", "true"],
        check=True,
    )
    applied_hashes: list[str] = []
    for entry in manifest["patches"]:
        if entry["classification"] not in ALLOWED_PATCH_CLASSIFICATIONS:
            raise ValueError(f"Disallowed patch classification: {entry['classification']}")
        patch = (project_root / entry["path"]).resolve(strict=True)
        digest = _file_sha256(patch)
        if digest != entry["sha256"]:
            raise ValueError(f"Patch checksum mismatch: {patch}")
        patch_text = patch.read_text(encoding="utf-8")
        if _touched_paths(patch_text) != set(entry["touched_paths"]):
            raise ValueError(f"Patch touched-path declaration mismatch: {patch}")
        subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", str(patch)],
            cwd=destination,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch)],
            cwd=destination,
            check=True,
            capture_output=True,
            text=True,
        )
        applied_hashes.append(digest)

    runner = destination / provenance.runner_relative_path
    return ExecutionTreeReport(
        root=destination,
        runner=runner,
        upstream_commit=provenance.commit,
        pre_patch_runner_sha256=provenance.runner_sha256,
        post_patch_runner_sha256=_file_sha256(runner),
        applied_patch_sha256=tuple(applied_hashes),
    )


def invoke_original_runner(
    *,
    project_root: Path,
    resolved_config_path: Path,
    runtime_dir: Path,
    python_executable: Path = Path(sys.executable),
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | list[str]:
    """Validate configuration, then execute only the patched upstream entry point."""

    project_root = project_root.resolve(strict=True)
    resolved_config_path = resolved_config_path.resolve(strict=True)
    config = ResolvedConfig.from_json(resolved_config_path.read_text(encoding="utf-8"))
    validate_nonfinal_config(config)
    enforce_compute_gate(
        config,
        project_root / "data" / "manifests" / "compute-gate.json",
        resolved_config_path,
    )
    project_git_binding = verify_project_git_binding(project_root)
    environment = os.environ.copy()
    cache_root = Path(
        environment.get("DS340W_CACHE_ROOT", Path.home() / ".cache" / "ds340w")
    )
    model_input_binding = verify_model_snapshot_binding(
        config, project_root, cache_root
    )
    course_data_binding, training_path, test_path = verify_course_data_binding(
        config, project_root, cache_root
    )
    report = prepare_execution_tree(project_root, runtime_dir)

    runner_payload = resolve_runner_payload(config, project_root)
    output_dir = Path(str(runner_payload["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_config = (output_dir / "runner-config.json").resolve()
    runner_config.write_text(
        json.dumps(runner_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    execution_report = output_dir / "execution-tree.json"
    serialized_report = asdict(report)
    serialized_report["root"] = str(report.root)
    serialized_report["runner"] = str(report.runner)
    serialized_report["applied_patch_sha256"] = list(report.applied_patch_sha256)
    serialized_report["resolved_config_sha256"] = config.sha256
    serialized_report["runner_config_sha256"] = _file_sha256(runner_config)
    serialized_report["model_input_binding"] = model_input_binding
    serialized_report["course_data_binding"] = course_data_binding
    serialized_report.update(project_git_binding)
    execution_report.write_text(
        json.dumps(serialized_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    command = [str(python_executable), str(report.runner), str(runner_config)]
    if dry_run:
        return command

    environment.pop("DS340W_PRIVATE_ROOT", None)
    environment.pop("DS340W_SST2_FINAL_FILE", None)
    python_paths = [str(report.root), str(report.root / "examples" / "legacies")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "42",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_DATASETS_CACHE": str(cache_root / "datasets"),
            "DS340W_SST2_TRAIN_FILE": str(training_path),
            "DS340W_SST2_TEST_FILE": str(test_path),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    if validated_preflight_phase(config) == "pilot":
        environment["DS340W_PREFLIGHT_TIMING_PATH"] = str(
            output_dir / "preflight-timing.json"
        )
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=output_dir,
        env=environment,
        text=True,
        check=False,
    )
    finished_at = datetime.now(timezone.utc)
    process_result = {
        "schema_version": 1,
        "entrypoint": "examples/legacies/examples_seq2seq/run_seq2seq.py",
        "argv": ["python", "run_seq2seq.py", "runner-config.json"],
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "wall_seconds": time.perf_counter() - started_clock,
        "exit_code": completed.returncode,
        "resolved_config_sha256": config.sha256,
        "model_input_binding": model_input_binding,
        "course_data_binding": course_data_binding,
        **project_git_binding,
        "nonfinal_access_enforcement": {
            "do_test": config.values["do_test"],
            "private_root_exposed_to_child": False,
            "final_unseen_file_exposed_to_child": False,
            "unseen_fields_in_config": [
                key for key in config.values if "unseen" in key.lower()
            ],
        },
    }
    (output_dir / "process-result.json").write_text(
        json.dumps(process_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return completed


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    result = invoke_original_runner(
        project_root=arguments.project_root,
        resolved_config_path=arguments.config,
        runtime_dir=arguments.runtime_dir,
        python_executable=arguments.python,
        dry_run=arguments.dry_run,
    )
    if isinstance(result, list):
        print(json.dumps(result))
    else:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    _main()
