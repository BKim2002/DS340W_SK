"""Guard against retraining or silently accepting mismatched completed runs."""

import json
from pathlib import Path

import pytest

from scripts import run_scaling_sweep as sweep
from ds340w_repro.scaling_experiment import build_scaling_config
from ds340w_repro.scaling_experiment import _historical_probe_evidence

ROOT = Path(__file__).resolve().parents[1]


def test_retained_execution_records_do_not_count_as_additional_gpu_work(tmp_path):
    for relative in ("old-probe", "retained/scaling-r2/evidence", "runs/scaling-r2"):
        path = tmp_path / "artifacts" / relative / "process-result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"wall_seconds": 7.0}))
    evidence = _historical_probe_evidence(tmp_path)
    assert len(evidence) == 1
    assert evidence[0]["path"] == "artifacts/old-probe/process-result.json"
    assert evidence[0]["wall_seconds"] == 7.0


def test_controller_handoff_requires_the_expected_runner(monkeypatch):
    monkeypatch.setattr(sweep.Path, "exists", lambda path: True)
    monkeypatch.setattr(sweep.Path, "read_bytes", lambda path: b"python\0unrelated-job.py\0")
    with pytest.raises(ValueError, match="expected OpenDelta runner"):
        sweep.wait_for_existing_training(12345, "scaling-r4-seed42")


@pytest.mark.parametrize("case", ("absent", "complete", "wrong_hash", "failed", "partial"))
def test_resume_accepts_only_completed_matching_training(tmp_path, monkeypatch, case):
    config = build_scaling_config(ROOT, 2)
    config.write(tmp_path / f"configs/resolved/scaling/{config.experiment_id}.json")
    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    if case == "absent":
        assert sweep.completed_training(config.experiment_id) is False
        return
    directory = tmp_path / f"artifacts/runs/{config.experiment_id}"
    directory.mkdir(parents=True)
    (directory / "process-result.json").write_text(json.dumps({
        "exit_code": 1 if case == "failed" else 0,
        "resolved_config_sha256": "f" * 64 if case == "wrong_hash" else config.sha256,
    }))
    (directory / "trainer_state.json").write_text(json.dumps({
        "global_step": 19999 if case == "partial" else 20000,
    }))
    if case == "complete":
        assert sweep.completed_training(config.experiment_id) is True
    else:
        with pytest.raises(ValueError, match="incomplete or its configuration differs"):
            sweep.completed_training(config.experiment_id)
