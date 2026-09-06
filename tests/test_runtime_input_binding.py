from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest

from ds340w_repro.build_matrix import build_baseline_matrix, build_smoke_config
from ds340w_repro.data.create_sst2_split import create_course_split
from ds340w_repro.data.materialize_sst2 import materialize_course_data
from ds340w_repro.preflight import ComputeGateError
from ds340w_repro.run_opendelta import (
    enforce_compute_gate,
    verify_project_git_binding,
    verify_course_data_binding,
    verify_model_snapshot_binding,
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    project = tmp_path / "project"
    cache = tmp_path / "cache"
    snapshot = cache / "hub/models--example/snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    for name, payload in (("config.json", b"{}\n"), ("pytorch_model.bin", b"weights")):
        (snapshot / name).write_bytes(payload)
    source_lock = {
        "schema_version": 1,
        "model": {
            "cache_relative_path": snapshot.relative_to(cache).as_posix(),
            "resolved_revision": "a" * 40,
            "files": [
                {
                    "relative_path": path.name,
                    "size": path.stat().st_size,
                    "sha256": _digest(path),
                }
                for path in sorted(snapshot.iterdir())
            ],
        },
    }
    manifest_path = project / "data/manifests/source-lock.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(source_lock), encoding="utf-8")
    config = build_smoke_config(snapshot, project / "artifacts/smoke").with_value(
        "source_lock_sha256", _digest(manifest_path), "fixture"
    )
    return project, cache, snapshot, config


def test_model_and_tokenizer_paths_are_bound_to_hash_verified_snapshot(
    tmp_path: Path,
) -> None:
    project, cache, _, config = _fixture(tmp_path)

    report = verify_model_snapshot_binding(config, project, cache)

    assert report["verified"] is True
    assert report["file_count"] == 2
    assert len(report["source_lock_sha256"]) == 64


def test_claimed_revision_cannot_mask_an_arbitrary_model_path(tmp_path: Path) -> None:
    project, cache, _, config = _fixture(tmp_path)
    rogue = tmp_path / "rogue"
    rogue.mkdir()
    (rogue / "config.json").write_text("{}", encoding="utf-8")
    changed = config.with_value(
        "model_name_or_path", str(rogue), "forged fixture"
    ).with_value("tokenizer_name", str(rogue), "forged fixture")

    with pytest.raises(ValueError, match="approved locked snapshot"):
        verify_model_snapshot_binding(changed, project, cache)


def test_snapshot_tampering_is_rejected_immediately_before_execution(
    tmp_path: Path,
) -> None:
    project, cache, snapshot, config = _fixture(tmp_path)
    (snapshot / "pytorch_model.bin").write_bytes(b"tamper!")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_model_snapshot_binding(config, project, cache)


def test_stop_gate_allows_only_the_bounded_week1_smoke(tmp_path: Path) -> None:
    gate = tmp_path / "compute-gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "decision": {
                    "status": "STOP",
                    "retained_training_authorized": False,
                    "reasons": ["pilot_incomplete"],
                },
            }
        ),
        encoding="utf-8",
    )
    smoke = build_smoke_config(Path("model"), Path("smoke"))
    baseline = build_baseline_matrix(
        Path("model"), Path("runs"), 1, 1
    )[0]
    smoke_path = tmp_path / "smoke.json"
    baseline_path = tmp_path / "baseline.json"
    smoke.write(smoke_path)
    baseline.write(baseline_path)

    assert enforce_compute_gate(smoke, gate, smoke_path) is None
    with pytest.raises(ComputeGateError, match="STOP"):
        enforce_compute_gate(baseline, gate, baseline_path)


def test_runtime_sees_only_hash_bound_training_and_test_files(tmp_path: Path) -> None:
    project, cache, _, config = _fixture(tmp_path)
    indices_path = project / "data/manifests/sst2-test-indices.json"
    test_indices = list(create_course_split(validation_size=6).test)
    indices_path.write_text(
        json.dumps(
            {
                "row_count": len(test_indices),
                "source_indices": test_indices,
                "source_split": "validation",
            }
        ),
        encoding="utf-8",
    )

    class Rows:
        def __init__(self, count: int, prefix: str):
            self.rows = [
                {"idx": index, "label": index % 2, "sentence": f"{prefix} {index}"}
                for index in range(count)
            ]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            return self.rows[index]

    manifest_path = project / "data/manifests/sst2-course-data.json"
    materialize_course_data(
        {"train": Rows(4, "train"), "validation": Rows(6, "valid")},
        output_root=cache / "course-data",
        manifest_path=manifest_path,
        test_indices_path=indices_path,
        expected_training_rows=4,
        expected_validation_rows=6,
    )
    bound = config.with_value(
        "course_data_manifest_sha256", _digest(manifest_path), "fixture"
    ).with_value("test_indices_sha256", _digest(indices_path), "fixture")

    evidence, training_path, test_path = verify_course_data_binding(
        bound,
        project,
        cache,
        expected_training_rows=4,
        expected_validation_rows=6,
    )

    assert evidence["visible_partitions"] == ["Training", "Test"]
    assert evidence["final_unseen_exposed"] is False
    assert training_path.name == "sst2-training.jsonl"
    assert test_path.name == "sst2-test.jsonl"


def test_execution_is_bound_to_a_clean_committed_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "--quiet", str(project)], check=True)
    subprocess.run(
        ["git", "-C", str(project), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project), "config", "user.name", "Test Fixture"],
        check=True,
    )
    tracked = project / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", "tracked.py"], check=True)
    subprocess.run(
        ["git", "-C", str(project), "commit", "--quiet", "-m", "fixture"],
        check=True,
    )

    report = verify_project_git_binding(project)

    assert report["project_worktree_clean"] is True
    assert len(report["project_git_commit"]) == 40

    tracked.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worktree must be clean"):
        verify_project_git_binding(project)
